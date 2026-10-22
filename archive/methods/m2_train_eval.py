"""M2: label-free production-consistency adapter (labels = source clean-crop id, no singer ids).

  ~/ps/.venv-ssl/bin/python -I pilots/m2_train_eval.py [test_pilot]   # default f2 (VocalSet)
Protocol: docs/04_pilot_log.md (M2). Same training crops and F2 test trials as M1.
Compares raw cosine, M1's singer-labelled clean adapter ("clean_singer") and the
label-free crop-id adapter ("cropid"). Failing strengths always come from VocalSet F2a.
Writes pilots/m2_results.json (f2) or pilots/m2_<test_pilot>_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
TEST = sys.argv[1] if len(sys.argv) > 1 else "f2"
from m1_train_eval import DATA, HERE, MODELS, load, train, trial_errors  # noqa: E402


def evaluate(model: str) -> dict:
    train_vec, test_vec = load("m1", model), load(TEST, model)
    manifest = json.loads((DATA / "m1" / "manifest.json").read_text())
    singer_id = {s: k for k, s in enumerate(sorted({r["singer"] for r in manifest}))}
    X, Ys, Yc, clean = [], [], [], []
    for crop_id, r in enumerate(manifest):
        for name, paths in r["crops"].items():
            if paths[0] in train_vec:
                X.append(train_vec[paths[0]])
                Ys.append(singer_id[r["singer"]])
                Yc.append(crop_id)
                clean.append(name == "clean")
    X, Ys, Yc, clean = np.stack(X), np.array(Ys), np.array(Yc), np.array(clean)

    f2 = json.loads((DATA / TEST / "manifest.json").read_text())
    test = [r for r in f2 if r["singer"] not in singer_id]
    sex = {r["singer"]: r["sex"] for r in test}
    keys, vecs = [], []
    for r in test:
        for name, paths in r["crops"].items():
            if paths[0] in test_vec:
                keys.append((r["singer"], r["song"], name))
                vecs.append(test_vec[paths[0]])
    V = np.stack(vecs)

    adapters = {"clean_singer": train(X[clean], Ys[clean]), "cropid": train(X, Yc)}
    spaces = {"raw": V}
    with torch.no_grad():
        for name, net in adapters.items():
            spaces[name] = F.normalize(net(torch.from_numpy(V)), dim=1).numpy()
    # M3: dot product of this concatenation = 0.5 * raw cosine + 0.5 * crop-id cosine.
    spaces["fusion"] = np.concatenate([spaces["raw"], spaces["cropid"]], axis=1) * np.sqrt(0.5)

    f2res = next(r for r in json.loads((HERE / "f2_results_final.json").read_text()) if r["model"] == model)
    failing = [v for v, s in f2res["levels"].items() if s["effect_pts"] >= 10 and s["ci95_pts"][0] > 0]
    per = {n: {v: trial_errors(dict(zip(keys, M)), sex, v) for v in failing + ["clean"]} for n, M in spaces.items()}
    singers = sorted(per["raw"]["clean"])

    def singer_mean(space: str, vs: list[str]) -> np.ndarray:
        return np.array([np.mean([per[space][v][s] for v in vs]) for s in singers])

    rng = np.random.default_rng(0)
    d = singer_mean("cropid", failing) - singer_mean("clean_singer", failing)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    res = {"model": model, "test_singers": singers, "failing_variants": failing,
           "failing_mean_error_pct": {n: float(singer_mean(n, failing).mean() * 100) for n in spaces},
           "clean_error_pct": {n: float(singer_mean(n, ["clean"]).mean() * 100) for n in spaces},
           "cropid_minus_clean_singer_pts": float(d.mean() * 100),
           "cropid_minus_clean_singer_ci95_pts": [float(np.percentile(boots, q) * 100) for q in (2.5, 97.5)]}
    res["clean_trial_change_pts"] = res["clean_error_pct"]["cropid"] - res["clean_error_pct"]["raw"]
    res["pass"] = bool(res["cropid_minus_clean_singer_pts"] <= 5 and res["clean_trial_change_pts"] <= 3)
    e, c = res["failing_mean_error_pct"], res["clean_error_pct"]
    res["m3_pass"] = bool(e["fusion"] <= e["clean_singer"] + 5 and c["fusion"] <= c["raw"] + 3)
    return res


def main() -> None:
    results = [evaluate(m) for m in MODELS if (DATA / TEST / f"emb_{m}.npz").exists()]
    (HERE / ("m2_results.json" if TEST == "f2" else f"m2_{TEST}_results.json")).write_text(json.dumps(results, indent=1))
    for r in results:
        e, c = r["failing_mean_error_pct"], r["clean_error_pct"]
        print(f"\n== {r['model']}  ({len(r['test_singers'])} test singers, {len(r['failing_variants'])} failing strengths)")
        print("   failing-strength error: " + "  ".join(f"{k} {v:.1f}%" for k, v in e.items()))
        print("   clean-trial error:      " + "  ".join(f"{k} {v:.1f}%" for k, v in c.items()))
        print(f"   cropid - clean_singer {r['cropid_minus_clean_singer_pts']:+.1f} pts "
              f"CI {[round(x, 1) for x in r['cropid_minus_clean_singer_ci95_pts']]} | PASS {r['pass']}")
    gate = [r["pass"] for r in results if r["model"] in ("cvsm_art", "ssl")]
    print("\nM2 PASS (CVSM-ART and SSL):", len(gate) == 2 and all(gate))
    gate = [r["m3_pass"] for r in results if r["model"] in ("cvsm_art", "ssl")]
    print("M3 PASS (fusion, CVSM-ART and SSL):", len(gate) == 2 and all(gate))


if __name__ == "__main__":
    main()
