"""Apply the pre-registered replication rule (docs/04_pilot_log.md, G1 and E1).

  ~/ps/.venv-ssl/bin/python -I pilots/g1_judge.py [pilot]   # g1 (default) or f2
Needs data/<pilot>/emb_*.npz and pilots/<pilot>_results.json (f2_analyze.py <pilot>).
Original encoders use their own VocalSet failing strengths (G1); extra encoders use the
fixed union S of CVSM-ART and SSL failing strengths (E1). Writes pilots/<pilot>_verdict.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
from m1_train_eval import DATA, HERE, load, trial_errors  # noqa: E402

PILOT = sys.argv[1] if len(sys.argv) > 1 else "g1"
FAMILY = {"reverb": "reverb", "codec": "codec", "mp3": "codec", "mixsep": "mixsep"}


def failing_of(r: dict) -> list[str]:
    return [v for v, s in r["levels"].items() if s["effect_pts"] >= 10 and s["ci95_pts"][0] > 0]


def judge(model: str, failing: list[str], g1: dict) -> dict:
    manifest = json.loads((DATA / PILOT / "manifest.json").read_text())
    vec = load(PILOT, model)
    sex = {r["singer"]: r["sex"] for r in manifest}
    emb = {(r["singer"], r["song"], n): vec[p[0]] for r in manifest for n, p in r["crops"].items() if p[0] in vec}
    base = trial_errors(emb, sex, "clean")
    per = {v: trial_errors(emb, sex, v) for v in failing}
    singers = sorted(base)
    d = np.array([np.mean([per[v][s] for v in failing]) - base[s] for s in singers])
    rng = np.random.default_rng(0)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    ci = [float(np.percentile(boots, q) * 100) for q in (2.5, 97.5)]
    broken = sorted({FAMILY[k] for k, hits in g1["breakpoint"].items() if hits and k in FAMILY})
    cross = np.mean([g1["levels"][v]["error_pct"] for v in failing])
    both = np.mean([g1["levels"][v]["both_processed_error_pct"] for v in failing])
    res = {"model": model, "n_singers": len(singers), "failing_vocalset": failing,
           "baseline_error_pct": g1["baseline_error_pct"],
           "a_mean_effect_pts": float(d.mean() * 100), "a_ci95_pts": ci,
           "b_families_with_breakpoint": broken,
           "c_cross_error_pct": float(cross), "c_both_processed_error_pct": float(both)}
    res["a"] = bool(res["a_mean_effect_pts"] >= 10 and ci[0] > 0)
    res["b"] = len(broken) >= 2
    res["c"] = bool(both <= cross / 2)
    res["pass"] = bool(res["a"] and res["b"] and res["c"])
    return res


def main() -> None:
    vocalset = {r["model"]: r for r in json.loads((HERE / "f2_results_final.json").read_text())}
    g1 = {r["model"]: r for r in json.loads((HERE / f"{PILOT}_results.json").read_text())}
    union = sorted(set(failing_of(vocalset["cvsm_art"])) | set(failing_of(vocalset["ssl"])))
    results = [judge(m, failing_of(vocalset[m]) if m in vocalset else union, g1[m]) for m in g1]
    (HERE / f"{PILOT}_verdict.json").write_text(json.dumps(results, indent=1))
    for r in results:
        print(f"== {r['model']} ({r['n_singers']} singers) clean baseline {r['baseline_error_pct']:.1f}%")
        print(f"   (a) effect {r['a_mean_effect_pts']:+.1f} pts CI {[round(x, 1) for x in r['a_ci95_pts']]} -> {r['a']}")
        print(f"   (b) families broken {r['b_families_with_breakpoint']} -> {r['b']}")
        print(f"   (c) cross {r['c_cross_error_pct']:.1f}% vs both-processed {r['c_both_processed_error_pct']:.1f}% -> {r['c']}")
        print(f"   PASS {r['pass']}")
    gate = [r["pass"] for r in results if r["model"] in ("cvsm_art", "ssl")]
    print(f"\n{PILOT} REPLICATION PASS (CVSM-ART and SSL):", len(gate) == 2 and all(gate))


if __name__ == "__main__":
    main()
