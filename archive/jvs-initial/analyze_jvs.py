"""Score frozen JVS cross-song trials and singer-bootstrap ranking-error CIs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("cvsm_art", "cvsm_a", "ssl")
CONDITIONS = ("clean", "mp3_64", "reverb_20")


def eer(pos: np.ndarray, neg: np.ndarray) -> float:
    thresholds = np.unique(np.r_[pos, neg])
    thresholds = np.r_[-np.inf, (thresholds[:-1] + thresholds[1:]) / 2, np.inf]
    frr = np.array([(pos < t).mean() for t in thresholds])
    far = np.array([(neg >= t).mean() for t in thresholds])
    index = int(np.argmin(np.abs(frr - far)))
    return float((frr[index] + far[index]) / 2)


def cluster_bootstrap(
    errors: np.ndarray, baseline: np.ndarray | None, group_ids: list[str], seed: int
) -> tuple[list[float], list[float] | None]:
    """Resample singer identities for both query and impostor gallery."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(np.asarray(group_ids) == g) for g in sorted(set(group_ids))]
    values, differences = [], []
    for _ in range(5000):
        value = difference = 0.0
        for ids in groups:
            while True:
                counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)))
                denominator = len(ids) - counts
                if not np.any((counts > 0) & (denominator == 0)):
                    break
            weight = len(ids) / len(group_ids)

            def sampled_mean(matrix: np.ndarray) -> float:
                local = matrix[np.ix_(ids, ids)]
                numerator = local @ counts
                per_query = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)
                return float(np.dot(counts, per_query) / len(ids))

            value += weight * sampled_mean(errors)
            if baseline is not None:
                difference += weight * (sampled_mean(errors) - sampled_mean(baseline))
        values.append(value)
        if baseline is not None:
            differences.append(difference)
    ci = [float(x * 100) for x in np.percentile(values, [2.5, 97.5])]
    diff_ci = [float(x * 100) for x in np.percentile(differences, [2.5, 97.5])] if baseline is not None else None
    return ci, diff_ci


def main() -> None:
    manifest = json.loads((ROOT / "experiments/jvs/manifest.json").read_text())
    groups = {r["singer"]: r["group"] for r in manifest}
    singers = sorted(groups)
    results = {}
    for model in MODELS:
        path = ROOT / "experiments/jvs" / f"embeddings_{model}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        V = data["vectors"].astype(np.float32)
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        keys = [tuple(map(str, row)) for row in data["keys"]]
        vec = {key[:3]: V[i] for i, key in enumerate(keys)}
        for c in CONDITIONS:
            for song in ("common", "unique"):
                missing = [s for s in singers if (s, song, c) not in vec]
                if missing:
                    raise ValueError(f"{model}: missing {c}/{song} for {missing[:5]}")
        out = {"model": model, "singers": len(singers), "vocal_range_groups": len(set(groups.values())), "conditions": {}}
        per_case = {}
        singer_groups = [groups[s] for s in singers]

        def score_case(label: str, qc: str, pc: str, nc: str) -> None:
            per_singer, positives, negatives = {}, [], []
            error_matrix = np.zeros((len(singers), len(singers)), dtype=bool)
            for i, s in enumerate(singers):
                qv = vec[(s, "common", qc)]
                pv = vec[(s, "unique", pc)]
                ps = float(qv @ pv)
                ns = []
                for j, t in enumerate(singers):
                    if i == j or groups[t] != groups[s]:
                        continue
                    score = float(qv @ vec[(t, "unique", nc)])
                    error_matrix[i, j] = score >= ps
                    ns.append(score)
                per_singer[s] = float(np.mean(np.asarray(ns) >= ps))
                positives.append(ps)
                negatives.extend(ns)
            err = np.array([per_singer[s] for s in singers], dtype=float)
            ci, _ = cluster_bootstrap(error_matrix, None, singer_groups, 20261112)
            out["conditions"][label] = {
                "query_condition": qc,
                "positive_condition": pc,
                "negative_condition": nc,
                "ranking_error_pct": float(err.mean() * 100),
                "ranking_error_ci95_singer_bootstrap_pct": ci,
                "eer_pct_descriptive": eer(np.asarray(positives), np.asarray(negatives)) * 100,
                "positive_pairs": len(positives),
                "negative_pairs": len(negatives),
                "per_vocal_range_error_pct": {
                    group: float(np.mean([per_singer[s] for s in singers if groups[s] == group]) * 100)
                    for group in sorted(set(groups.values()))
                },
            }
            per_case[label] = error_matrix

        score_case("clean_all", "clean", "clean", "clean")
        for cond in ("mp3_64", "reverb_20"):
            score_case(f"clean_query_{cond}_positive", "clean", cond, "clean")
            score_case(f"{cond}_query_clean_candidates", cond, "clean", "clean")
            score_case(f"matched_{cond}", cond, cond, cond)
        baseline = out["conditions"]["clean_all"]["ranking_error_pct"]
        baseline_matrix = per_case["clean_all"]
        for label, record in out["conditions"].items():
            record["difference_vs_clean_all_pts"] = record["ranking_error_pct"] - baseline
            _, diff_ci = cluster_bootstrap(per_case[label], baseline_matrix, singer_groups, 20261112)
            record["difference_vs_clean_all_ci95_singer_bootstrap_pts"] = diff_ci
        results[model] = out
        print(f"\n{model}: clean {baseline:.1f}%")
        for label, record in out["conditions"].items():
            lo, hi = record["ranking_error_ci95_singer_bootstrap_pct"]
            print(f"  {label:38s} rank-error {record['ranking_error_pct']:5.1f}% [{lo:4.1f}, {hi:4.1f}] | EER {record['eer_pct_descriptive']:5.1f}%")
    out_path = ROOT / "experiments/jvs/results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
