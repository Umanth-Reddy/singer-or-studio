"""C1 step 4: apply the pre-registered causal-test rule (docs/04_pilot_log.md, C1).

  ~/ps/.venv-ssl/bin/python -I pilots/c1_judge.py
Needs data/{f2,g1}/emb_c1{base,A,B,Bnorev}.npz. Effect = per-singer mean error over the fixed
strength set S (E1: union of VocalSet failing strengths of CVSM-ART and SSL) minus clean error.
Writes pilots/c1_verdict.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
from g1_judge import failing_of  # noqa: E402
from m1_train_eval import DATA, HERE, load, trial_errors  # noqa: E402

ARMS = ("c1base", "c1A", "c1B", "c1Bnorev")


def per_singer(pilot: str, model: str, S: list[str]) -> tuple[dict, dict]:
    manifest = json.loads((DATA / pilot / "manifest.json").read_text())
    vec = load(pilot, model)
    sex = {r["singer"]: r["sex"] for r in manifest}
    emb = {(r["singer"], r["song"], n): vec[p[0]] for r in manifest for n, p in r["crops"].items() if p[0] in vec}
    clean = trial_errors(emb, sex, "clean")
    per = {v: trial_errors(emb, sex, v) for v in S}
    return clean, per


def ci(d: np.ndarray) -> list[float]:
    rng = np.random.default_rng(0)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return [float(np.percentile(b, q) * 100) for q in (2.5, 97.5)]


def main() -> None:
    vocalset = {r["model"]: r for r in json.loads((HERE / "f2_results_final.json").read_text())}
    S = sorted(set(failing_of(vocalset["cvsm_art"])) | set(failing_of(vocalset["ssl"])))
    reverb = [v for v in S if v.startswith("reverb@")]
    out = {"S": S, "datasets": {}}
    for pilot in ("f2", "g1"):
        res = {}
        for arm in ARMS:
            clean, per = per_singer(pilot, arm, S)
            singers = sorted(clean)
            res[arm] = {"singers": singers,
                        "clean": np.array([clean[s] for s in singers]),
                        "effect": np.array([np.mean([per[v][s] for v in S]) - clean[s] for s in singers]),
                        "reverb_effect": np.array([np.mean([per[v][s] for v in reverb]) - clean[s] for s in singers])}
        a, b, nr = res["c1A"], res["c1B"], res["c1Bnorev"]
        summary = {arm: {"clean_pct": float(r["clean"].mean() * 100), "effect_pts": float(r["effect"].mean() * 100),
                         "reverb_effect_pts": float(r["reverb_effect"].mean() * 100)} for arm, r in res.items()}
        d = b["effect"] - a["effect"]
        verdict = {"B_minus_A_effect_pts": float(d.mean() * 100), "B_minus_A_ci95": ci(d),
                   "uninformative": summary["c1A"]["clean_pct"] > 25}
        verdict["pass"] = bool(not verdict["uninformative"]
                               and summary["c1B"]["effect_pts"] <= 0.5 * summary["c1A"]["effect_pts"]
                               and verdict["B_minus_A_ci95"][1] < 0
                               and summary["c1B"]["clean_pct"] <= summary["c1A"]["clean_pct"] + 3)
        dr = nr["reverb_effect"] - a["reverb_effect"]
        verdict["Bnorev_minus_A_reverb_pts"] = float(dr.mean() * 100)
        verdict["Bnorev_minus_A_reverb_ci95"] = ci(dr)
        out["datasets"][pilot] = {"summary": summary, "verdict": verdict}
    (HERE / "c1_verdict.json").write_text(json.dumps(out, indent=1))
    for pilot, r in out["datasets"].items():
        print(f"== {pilot} ({'VocalSet' if pilot == 'f2' else 'GTSinger'})")
        for arm, s in r["summary"].items():
            print(f"   {arm:9s} clean {s['clean_pct']:5.1f}%  effect {s['effect_pts']:+5.1f} pts  reverb effect {s['reverb_effect_pts']:+5.1f}")
        v = r["verdict"]
        print(f"   B - A effect {v['B_minus_A_effect_pts']:+.1f} CI {[round(x, 1) for x in v['B_minus_A_ci95']]}"
              f" | uninformative {v['uninformative']} | PASS {v['pass']}")
        print(f"   Bnorev - A on reverb {v['Bnorev_minus_A_reverb_pts']:+.1f} CI {[round(x, 1) for x in v['Bnorev_minus_A_reverb_ci95']]}")
    print("\nC1 PASS (both datasets):", all(r["verdict"]["pass"] for r in out["datasets"].values()))


if __name__ == "__main__":
    main()
