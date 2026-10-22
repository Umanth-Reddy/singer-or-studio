"""F2a step 3: singer error vs production strength (dose-response) per encoder.

  ~/ps/.venv-gate1/bin/python -I pilots/f2_analyze.py [pilot]   # pilot defaults to f2
Trial: query = clean (singer s, song i); positive = same singer, other song j, processed
at one strength; negative = other same-sex singer, song j, clean (shares the query's
production). Error = sim(q,neg) >= sim(q,pos). Baseline: positive clean too.
Breakpoint = mildest strength whose error exceeds baseline by >= 10 points with the
95 % singer-bootstrap CI excluding 0. Writes pilots/f2_results.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = sys.argv[1] if len(sys.argv) > 1 else "f2"
OUT = HERE.parent / "data" / PILOT
MODELS = ("cvsm_art", "cvsm_a", "ssl", "ecapa", "mert")
RNG = np.random.default_rng(0)


def analyze(model: str, manifest: list[dict]) -> dict | None:
    path = OUT / f"emb_{model}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    vec = dict(zip(d["paths"].tolist(), d["vectors"] / np.linalg.norm(d["vectors"], axis=1, keepdims=True)))
    emb, sex = {}, {}
    for r in manifest:
        sex[r["singer"]] = r["sex"]
        for name, paths in r["crops"].items():
            if paths[0] in vec:
                emb[(r["singer"], r["song"], name)] = vec[paths[0]]
    singers = sorted(sex)
    songs = sorted({k[1] for k in emb})
    variants = sorted({k[2] for k in emb})

    def error(variant: str, neg_variant: str = "clean") -> np.ndarray:
        per = defaultdict(list)
        for s in singers:
            for i in songs:
                q = emb.get((s, i, "clean"))
                for j in songs:
                    pos = emb.get((s, j, variant))
                    if q is None or pos is None or j == i:
                        continue
                    sp = float(q @ pos)
                    for s2 in singers:
                        neg = emb.get((s2, j, neg_variant))
                        if s2 != s and sex[s2] == sex[s] and neg is not None:
                            per[s].append(float(q @ neg) >= sp)
        return np.array([np.mean(per[s]) for s in singers if per[s]])

    base = error("clean")
    out = {"model": model, "baseline_error_pct": float(base.mean() * 100), "levels": {}}
    for v in variants:
        if v == "clean":
            continue
        e = error(v)
        diff = e - base
        boots = [diff[RNG.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        both = error(v, neg_variant=v)  # control: both candidates share the processing
        out["levels"][v] = {"error_pct": float(e.mean() * 100), "effect_pts": float(diff.mean() * 100),
                            "both_processed_error_pct": float(both.mean() * 100),
                            "ci95_pts": [float(np.percentile(boots, 2.5) * 100), float(np.percentile(boots, 97.5) * 100)]}
    out["breakpoint"] = {}
    for kind in sorted({v.split("@")[0] for v in out["levels"]}):
        hits = [v for v, s in out["levels"].items() if v.startswith(kind + "@") and s["effect_pts"] >= 10 and s["ci95_pts"][0] > 0]
        out["breakpoint"][kind] = hits
    return out


def main() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())
    results = [r for r in (analyze(m, manifest) for m in MODELS) if r]
    (HERE / f"{PILOT}_results.json").write_text(json.dumps(results, indent=1))
    names = list(results[0]["levels"])
    print(f"{'variant':14s}" + "".join(f"{r['model']:>22s}" for r in results))
    print(f"{'clean (base)':14s}" + "".join(f"{r['baseline_error_pct']:21.1f}%" for r in results))
    for v in names:
        print(f"{v:14s}" + "".join(f"{r['levels'][v]['error_pct']:15.1f}% ({r['levels'][v]['effect_pts']:+5.1f})" for r in results))
    print("\nControl: both candidates processed identically (identity survives if near baseline)")
    for v in names:
        print(f"{v:14s}" + "".join(f"{r['levels'][v]['both_processed_error_pct']:21.1f}%" for r in results))


if __name__ == "__main__":
    main()
