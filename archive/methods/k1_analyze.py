"""K1 step 3: plain vs accompaniment-debiased cross-song positive mining.

  ~/ps/.venv-gate1/bin/python -I pilots/k1_analyze.py
Protocol: docs/04_pilot_log.md (K1). Writes pilots/k1_results.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "k1"
MODELS = ("cvsm_a", "cvsm_art", "ssl")
MIN_VOCAL_RATIO = 0.15  # fixed before data: best-window vocal/accomp RMS
LAMBDA = 1.0
N_BOOT = 2000
RNG = np.random.default_rng(0)


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def zscore_offdiag(s: np.ndarray) -> np.ndarray:
    off = s[~np.eye(len(s), dtype=bool)]
    return (s - off.mean()) / off.std()


def mine(score: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    score = score.copy()
    np.fill_diagonal(score, -np.inf)
    return np.argsort(-score[queries], axis=1)[:, :k]


def analyze(model: str, manifest: list[dict]) -> dict | None:
    path = OUT / f"emb_{model}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    vec = dict(zip(data["paths"].tolist(), unit(data["vectors"])))
    tracks = []
    for r in manifest:
        if max(v / max(a, 1e-9) for v, a in zip(r["vocal_rms"], r["accomp_rms"])) < MIN_VOCAL_RATIO:
            continue
        v = [vec[p] for p in r["crops"]["vocals"] if p in vec]
        a = [vec[p] for p in r["crops"]["accomp"] if p in vec]
        if v and a:
            tracks.append((r, unit(np.mean(v, 0)), unit(np.mean(a, 0))))
    artist = np.array([t[0]["artist"] for t in tracks])
    album = np.array([t[0]["album_title"] for t in tracks])
    V = np.stack([t[1] for t in tracks])
    A = np.stack([t[2] for t in tracks])
    # Queries: tracks whose artist still has >= 2 albums in the pool after filtering.
    albums_of = defaultdict(set)
    for a, b in zip(artist, album):
        albums_of[a].add(b)
    queries = np.array([i for i in range(len(tracks)) if len(albums_of[artist[i]]) >= 2])
    sv, sa = V @ V.T, A @ A.T
    scores = {"plain": sv, "debiased": zscore_offdiag(sv) - LAMBDA * zscore_offdiag(sa),
              "accomp_only": sa}
    q_artist = artist[queries]
    result = {"model": model, "n_pool": len(tracks), "n_queries": len(queries),
              "n_query_artists": int(len(set(q_artist))), "k": {}}
    for k in (1, 5):
        per = {}
        for name, s in scores.items():
            nn = mine(s, queries, k)
            same = artist[nn] == q_artist[:, None]
            cross = same & (album[nn] != album[queries][:, None])
            per[name] = {"same": same.mean(1), "cross": cross.mean(1),
                         "same_album_share_of_hits": float((same & ~cross).sum() / max(same.sum(), 1))}
        uniq = np.unique(q_artist)
        idx = {a: np.where(q_artist == a)[0] for a in uniq}
        boots = []
        for _ in range(N_BOOT):
            rows = np.concatenate([idx[a] for a in RNG.choice(uniq, len(uniq))])
            p, d = per["plain"]["cross"][rows].mean(), per["debiased"]["cross"][rows].mean()
            boots.append(d / p if p > 0 else np.nan)
        boots = np.array(boots)
        summary = {name: {"P_same_artist": float(v["same"].mean()), "CA_cross_album": float(v["cross"].mean()),
                          "same_album_share_of_hits": v["same_album_share_of_hits"]} for name, v in per.items()}
        summary["CA_ratio_debiased_over_plain"] = float(per["debiased"]["cross"].mean() / max(per["plain"]["cross"].mean(), 1e-9))
        summary["CA_ratio_ci95"] = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
        summary["pass"] = bool(summary["CA_ratio_ci95"][0] > 1 and summary["CA_ratio_debiased_over_plain"] >= 1.5
                               and summary["debiased"]["P_same_artist"] >= 0.8 * summary["plain"]["P_same_artist"])
        result["k"][k] = summary
    return result


def main() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())
    results = [r for r in (analyze(m, manifest) for m in MODELS) if r]
    (HERE / "k1_results.json").write_text(json.dumps(results, indent=1))
    for r in results:
        print(f"\n== {r['model']} pool={r['n_pool']} queries={r['n_queries']} artists={r['n_query_artists']}")
        for k, s in r["k"].items():
            print(f" top-{k}:", json.dumps(s))
    top1 = [r["k"][1]["pass"] for r in results]
    print(f"\nK1 PASS (top-1, >= 2 of {len(top1)} encoders):", sum(top1) >= 2, top1)


if __name__ == "__main__":
    main()
