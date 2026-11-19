"""Verify core trial membership, rates and shared-condition intervals.

Requires NumPy. Run beside the CSV, JSON results and crop manifests.
"""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = ("cvsm_art", "cvsm_a", "ssl", "ecapa", "mert")


def read(filename):
    return json.loads((HERE / filename).read_text())


def verify():
    results = read("supplement-results.json")["results"]
    audit = read("trial-audit.json")
    plan = audit["bootstrap"]
    counts = defaultdict(dict)
    with (HERE / "core-singer-counts.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["cohort"], row["model"], row["condition"]
            singer = int(row["singer"])
            if singer in counts[key]:
                raise ValueError(f"Duplicate singer {singer} in {key}")
            counts[key][singer] = tuple(map(int, (
                row["eligible"], row["genuine_only_errors"], row["both_candidate_errors"],
            )))

    checked = 0
    for cohort, label in (("f2", "vocalset"), ("g1", "gtsinger")):
        records = read(f"{cohort}-crop-manifest.json")
        encoded = {(r["singer"], r["song_or_slot"]): r["sex_group"] for r in records}
        candidate_counts = Counter((slot, group) for (_, slot), group in encoded.items())
        expected = Counter()
        for (singer, query_slot), group in encoded.items():
            for (owner, genuine_slot) in encoded:
                if owner == singer and query_slot != genuine_slot:
                    expected[singer] += candidate_counts[genuine_slot, group] - 1
        np.testing.assert_equal(sum(expected.values()), audit[cohort]["triplets"])
        order = sorted(expected)
        rng = np.random.default_rng(plan["seed"])
        draws = rng.integers(0, len(order), size=(plan["draws"], len(order)))

        by_model = {r["model"]:r for r in results[f"{label}_per_condition"]}
        for model in MODELS:
            stored = by_model[model]
            ratios = {}
            for condition in ("clean", *stored["levels"]):
                row = counts.pop((cohort, model, condition))
                np.testing.assert_equal(sorted(row), order)
                trials, genuine, both = np.array([row[singer] for singer in order]).T
                np.testing.assert_equal(trials, [expected[singer] for singer in order])
                if np.any((genuine < 0) | (genuine > trials) | (both < 0) | (both > trials)):
                    raise ValueError("Numerator outside the trial count")
                ratios[condition] = genuine / trials
                if condition == "clean":
                    clean = stored["baseline_error_pct"]
                    reference = [clean, clean]
                else:
                    cell = stored["levels"][condition]
                    reference = [cell["error_pct"], cell["both_processed_error_pct"]]
                np.testing.assert_allclose(
                    [ratios[condition].mean() * 100, (both / trials).mean() * 100],
                    reference, rtol=0, atol=1e-8,
                )
                checked += 1
            difference = np.mean([ratios[k] for k in plan["shared_settings"]], axis=0) - ratios["clean"]
            interval = np.percentile(difference[draws].mean(axis=1), [2.5, 97.5]) * 100
            target = audit[cohort]["encoders"][model]
            np.testing.assert_allclose(
                interval, target["common_ci95_sorted_singer_order"], rtol=0, atol=1e-8,
            )
            comparison = results["common_17_condition_encoder_comparison"][cohort][model]
            np.testing.assert_equal(interval.round(1).tolist(), comparison["ci95"])
            np.testing.assert_allclose(
                difference.mean() * 100, comparison["effect_over_S"], rtol=0, atol=0.051,
            )
    if counts:
        raise ValueError("Unexpected remaining model or condition rows")
    print(f"Verified {checked} core model-condition masks and rates, plus ten shared-set intervals.")


if __name__ == "__main__":
    verify()
