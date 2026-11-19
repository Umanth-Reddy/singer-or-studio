"""Recompute the JVS point estimates and check aggregate result arithmetic."""
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean

HERE = Path(__file__).resolve().parent
STATES = tuple(itertools.product(range(2), repeat=3))


def same(actual, reference, label, tolerance=1e-7):
    if not math.isclose(actual, reference, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"{label}: calculated {actual}, expected {reference}")


def verify_files():
    for entry in (HERE / "SHA256SUMS").read_text().splitlines():
        digest, filename = entry.split("  ", 1)
        if hashlib.sha256((HERE / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"SHA-256 mismatch: {filename}")


def verify_jvs(archive):
    trials = defaultdict(lambda: defaultdict(dict))
    with (HERE / "jvs-trial-scores.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            cohort = row["model"], row["query_song"], row["treatment"]
            query, impostor = row["query_id"], row["impostor_id"]
            if query == impostor or impostor in trials[cohort][query]:
                raise ValueError(f"Invalid trial or duplicate: {cohort}, {query}, {impostor}")
            trials[cohort][query][impostor] = row

    count = 0
    for model, model_data in archive["models"].items():
        for role, role_data in model_data["roles"].items():
            for treatment, result in role_data["conditions"].items():
                cohort = model, role, treatment
                by_singer = trials.pop(cohort)
                same(len(by_singer), archive["singers"], cohort)
                same(sum(map(len, by_singer.values())), role_data["triplets"], cohort)
                derived = {}
                for q, p, n in STATES:
                    cell = f"{q}{p}{n}"
                    scores_by_singer = []
                    for singers in by_singer.values():
                        scores = []
                        for row in singers.values():
                            genuine = float(row[f"g{q}{p}"])
                            impostor = float(row[f"i{q}{n}"])
                            scores.append((100 * (genuine <= impostor), genuine, impostor,
                                           genuine - impostor))
                        scores_by_singer.append(tuple(map(fmean, zip(*scores))))
                    derived[q, p, n] = tuple(map(fmean, zip(*scores_by_singer)))
                    fields = ("ranking_error_pct", "genuine_cosine", "impostor_cosine", "margin")
                    for name, value in zip(fields, derived[q, p, n]):
                        same(value, result["cells"][cell][name]["value"],
                             (cohort, cell, name))
                    count += 1

                def value(q, p, n, metric=0):
                    return derived[q, p, n][metric]

                matches = tuple(itertools.product(range(2), repeat=2))
                checks = {
                    "impostor_condition_match_error_pp":
                        fmean(value(q,p,q) - value(q,p,1-q) for q,p in matches),
                    "impostor_condition_match_cosine":
                        fmean(value(q,p,q,2) - value(q,p,1-q,2) for q,p in matches),
                    "genuine_condition_match_cosine":
                        fmean(value(q,q,n,1) - value(q,1-q,n,1) for q,n in matches),
                    "genuine_only_minus_query_only_error_pp": value(0,1,0) - value(1,0,0),
                    "clean_query_processed_genuine_impostor_match_error_pp":
                        value(0,1,0) - value(0,1,1),
                    "genuine_only_minus_clean_error_pp": value(0,1,0) - value(0,0,0),
                    "genuine_only_minus_clean_genuine_cosine": value(0,1,0,1) - value(0,0,0,1),
                }
                for name, estimate in checks.items():
                    same(estimate, result["contrasts"][name]["value"], (cohort, name))
                if cohort == ("cvsm_art", "common", "reverb_20"):
                    print("Headline JVS result, clean / genuine only / both candidates:")
                    print(f"  {value(0,0,0):.4f}% / {value(0,1,0):.4f}% / {value(0,1,1):.4f}%")
                    print(f"  Impostor intervention reduced error by {value(0,1,0)-value(0,1,1):.4f} points.")
    if trials:
        raise ValueError(f"Unexpected extra JVS trials in {list(trials)}")
    print(f"Verified {count} JVS cells with ranking and score averages, plus 84 point contrasts.")


def verify_aggregates(results):
    selected = {entry["model"]: set(entry["failing_vocalset"])
                for entry in results["vocalset_selected_conditions"]}
    shared = selected["cvsm_art"] | selected["ssl"]
    same(len(shared), 17, "common condition count")
    for short, corpus in (("f2", "vocalset"), ("g1", "gtsinger")):
        for row in results[f"{corpus}_per_condition"]:
            model = row["model"]
            baseline, levels = row["baseline_error_pct"], row["levels"]
            for setting, level in levels.items():
                same(level["error_pct"] - baseline, level["effect_pts"],
                     (corpus, model, setting))
            reported = results["common_17_condition_encoder_comparison"][short][model]
            same(fmean(levels[key]["effect_pts"] for key in shared),
                 reported["effect_over_S"], (corpus, model, "common shift"), tolerance=0.051)
            same(baseline, reported["clean_err_pct"], (corpus, model, "clean"), tolerance=0.051)

    room_key = "impostor_condition_match_error_pp"
    datasets = results["new_measured_response_and_galleries"]["datasets"]
    for dataset, data in datasets.items():
        for model, result in data["models"].items():
            for wet, row in result["levels"].items():
                label = dataset, model, wet
                pairs = [pair[room_key]["value"] for pair in row["per_pair"]]
                same(fmean(pairs), row["pooled"][room_key]["value"], label)
                for i, estimate in enumerate(row["leave_one_pair_out_error_pp"]):
                    same(fmean(pairs[:i] + pairs[i+1:]), estimate["value"], label)
                gallery = row["mixed_gallery"]
                same(gallery["top1_error_pct"] - gallery["all_other_top1_error_pct"],
                     row["posthoc_secondary_gallery_uncertainty"]["mixed_minus_all_other_top1_pp"]["value"],
                     label)
    print("Verified the core condition arithmetic, shared-set means and measured-response contrasts.")


def main():
    verify_files()
    results = json.loads((HERE / "supplement-results.json").read_text())["results"]
    verify_jvs(results["jvs_factorial"]["jvs"])
    verify_aggregates(results)


if __name__ == "__main__":
    main()
