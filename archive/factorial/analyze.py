"""Analyze fixed production contrasts using existing embedding files."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

CELLS = list(itertools.product(range(2), repeat=3))
JVS_MODELS = ("cvsm_art", "cvsm_a", "ssl")
ROOM_MODELS = (*JVS_MODELS, "ecapa", "mert")
WET_LEVELS = ("0.1", "0.2", "0.3", "0.5")
N_BOOT = 5000


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms == 0):
        raise ValueError("Nonfinite or zero embedding")
    return vectors / norms


class Trials:
    def __init__(self, recordings, groups, query_song=None):
        self.recordings = recordings
        singers = sorted(groups)
        singer_index = {s: i for i, s in enumerate(singers)}
        self.n_singers = len(singers)
        self.n_groups = len(set(groups.values()))
        index = {r: i for i, r in enumerate(recordings)}
        songs = sorted({song for _, song in recordings})
        rows = []
        for singer, song in recordings:
            if query_song is not None and song != query_song:
                continue
            for candidate_song in songs:
                if candidate_song == song or (singer, candidate_song) not in index:
                    continue
                for other in singers:
                    if other != singer and groups[singer] == groups[other] and (other, candidate_song) in index:
                        rows.append((index[(singer, song)], index[(singer, candidate_song)],
                                     index[(other, candidate_song)],
                                     singer_index[singer] * self.n_singers + singer_index[other]))
        self.q, self.p, self.n, self.pair = np.asarray(rows).T
        self.counts = np.bincount(self.pair, minlength=self.n_singers ** 2).reshape(self.n_singers, -1)
        self.active = self.counts.ravel() > 0
        self.weights = self._weights(np.ones((1, self.n_singers)))[0]
        rng = np.random.default_rng(20261119)
        group_index = [np.asarray([i for i, s in enumerate(singers) if groups[s] == g])
                       for g in sorted(set(groups.values()))]

        def draw(n):
            counts = np.zeros((n, self.n_singers))
            for ids in group_index:
                counts[:, ids] = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=n)
            return counts

        multiplicities = draw(N_BOOT)
        invalid = np.any((multiplicities > 0) & (multiplicities @ self.counts.T == 0), axis=1)
        while invalid.any():
            multiplicities[invalid] = draw(invalid.sum())
            invalid = np.any((multiplicities > 0) & (multiplicities @ self.counts.T == 0), axis=1)
        self.boot_weights = self._weights(multiplicities)

    def _weights(self, multiplicities):
        denominators = multiplicities @ self.counts.T
        query = np.divide(multiplicities, denominators, out=np.zeros_like(multiplicities),
                          where=denominators > 0)
        weights = query[:, :, None] * multiplicities[:, None, :]
        return weights.reshape(len(multiplicities), -1)[:, self.active] / self.n_singers

    def collapse(self, array):
        return np.bincount(self.pair, weights=array, minlength=self.n_singers ** 2)[self.active]

    def summarize(self, array):
        lo, hi = np.percentile(self.boot_weights @ array, [2.5, 97.5])
        return {"value": float(self.weights @ array), "ci95": [float(lo), float(hi)]}


def matrices(data, trials, left, right):
    a, b = data[left], data[right]
    genuine = np.einsum("ij,ij->i", a[trials.q], b[trials.p])
    impostor = np.einsum("ij,ij->i", a[trials.q], b[trials.n])
    return genuine, impostor


def factorial(data, trials, states):
    similarity = {(q, c): matrices(data, trials, states[q], states[c])
                  for q, c in itertools.product(range(2), repeat=2)}
    cells = {}
    for q, p, n in CELLS:
        genuine = similarity[q, p][0]
        impostor = similarity[q, n][1]
        margin = genuine - impostor
        cells[q, p, n] = np.array([trials.collapse(row)
                                    for row in (100 * (margin <= 0), genuine, impostor, margin)])
    return cells


def contrasts(cells):
    impostor_change = np.mean([cells[q, p, q] - cells[q, p, 1 - q]
                               for q, p in itertools.product(range(2), repeat=2)], axis=0)
    genuine_change = np.mean([cells[q, q, n] - cells[q, 1 - q, n]
                              for q, n in itertools.product(range(2), repeat=2)], axis=0)
    return {
        "impostor_condition_match_error_pp": impostor_change[0],
        "impostor_condition_match_cosine": impostor_change[2],
        "genuine_condition_match_cosine": genuine_change[1],
    }


def describe_cells(trials, cells):
    fields = ("ranking_error_pct", "genuine_cosine", "impostor_cosine", "margin")
    return {"".join(map(str, key)): {field: trials.summarize(row)
                                    for field, row in zip(fields, values)}
            for key, values in cells.items()}


def summarize(trials, values):
    return {k: trials.summarize(v) for k, v in values.items()}


def validate(observed, archived, label):
    difference = abs(observed - archived)
    if difference > 0.01:
        raise ValueError(f"{label} failed replay: {observed} vs {archived}")
    return difference


def jvs_analysis(root):
    manifest = read_json(root / "manifest.json")
    groups = {r["singer"]: r["group"] for r in manifest}
    recordings = [(s, song) for s in sorted(groups) for song in ("common", "unique")]
    trials_by_role = {role: Trials(recordings, groups, role) for role in ("common", "unique")}
    archived = read_json(root / "results.json")
    results = {"singers": len(groups), "models": {}, "max_replay_difference_pp": 0.0}
    for model in JVS_MODELS:
        path = root / f"embeddings_{model}.npz"
        checksum = sha256(path)
        if checksum != read_json(root / f"provenance_{model}.json")["embedding_sha256"]:
            raise ValueError(f"Unexpected JVS checkpoint embeddings: {path}")
        with np.load(path) as packed:
            rows = dict(zip(map(tuple, packed["keys"][:, :3]), normalized(packed["vectors"]), strict=True))
        data = {cond: np.stack([rows[s, r, cond] for s, r in recordings])
                for cond in ("clean", "reverb_20", "mp3_64")}
        output = results["models"][model] = {"embedding_sha256": checksum, "roles": {}}
        for role, trials in trials_by_role.items():
            role_out = output["roles"][role] = {"triplets": len(trials.q), "conditions": {}}
            for cond in ("reverb_20", "mp3_64"):
                cells = factorial(data, trials, ("clean", cond))
                effects = contrasts(cells)
                effects["genuine_only_minus_query_only_error_pp"] = cells[0, 1, 0][0] - cells[1, 0, 0][0]
                effects["clean_query_processed_genuine_impostor_match_error_pp"] = cells[0, 1, 0][0] - cells[0, 1, 1][0]
                effects["genuine_only_minus_clean_error_pp"] = cells[0, 1, 0][0] - cells[0, 0, 0][0]
                effects["genuine_only_minus_clean_genuine_cosine"] = cells[0, 1, 0][1] - cells[0, 0, 0][1]
                role_out["conditions"][cond] = {"cells": describe_cells(trials, cells),
                                                  "contrasts": summarize(trials, effects)}
                if role == "common":
                    mapping = {"000": "clean_all", "010": f"clean_query_{cond}_positive",
                               "100": f"{cond}_query_clean_candidates", "111": f"matched_{cond}"}
                    for cell, label in mapping.items():
                        record = role_out["conditions"][cond]["cells"][cell]["ranking_error_pct"]
                        delta = validate(record["value"], archived[model]["conditions"][label]["ranking_error_pct"],
                                         f"JVS {model} {label}")
                        results["max_replay_difference_pp"] = max(results["max_replay_difference_pp"], delta)
    return results


def room_analysis(root):
    source = read_json(root / "data/selection.json")["rooms"]
    units = sorted(({"id": i, **row} for i, row in enumerate(source)), key=lambda r: r["t30_s"])
    pairs = list(zip(units[::2], units[1::2], strict=True))
    archived = read_json(root / "t1_results.json")
    results = {"room_pairs": pairs, "datasets": {}, "max_replay_difference_pp": 0.0}
    for dataset in ("f2", "g1"):
        directory = root / "data" / dataset
        manifest_path = directory / "manifest.json"
        manifest = read_json(manifest_path)
        groups = {r["singer"]: r["sex"] for r in manifest}
        recordings = [(r["singer"], r["song"]) for r in manifest]
        trials = Trials(recordings, groups)
        dataset_out = results["datasets"][dataset] = {
            "singers": len(groups), "recordings": len(recordings), "triplets_per_cell": len(trials.q),
            "manifest_sha256": sha256(manifest_path), "models": {}}
        for model in ROOM_MODELS:
            path = directory / f"emb_{model}.npz"
            with np.load(path) as packed:
                vectors = dict(zip(packed["paths"].tolist(), normalized(packed["vectors"]), strict=True))
            variants = ["clean"] + [f"reverb@{w}#r{u['id']}" for w in WET_LEVELS for u in units]
            data = {cond: np.stack([vectors[r["crops"][cond][0]] for r in manifest]) for cond in variants}
            clean = factorial(data, trials, ("clean", "clean"))[0, 0, 0][0]
            model_out = dataset_out["models"][model] = {
                "embedding_sha256": sha256(path), "clean_ranking_error_pct": trials.summarize(clean),
                "levels": {}}
            wet_errors = []
            for wet in WET_LEVELS:
                names = [f"reverb@{wet}#r{unit['id']}" for unit in units]
                clean_negative = matrices(data, trials, "clean", "clean")[1]
                ranking_errors = [trials.collapse(100 * (clean_negative >=
                                  matrices(data, trials, "clean", label)[0])) for label in names]
                wet_error = np.mean(ranking_errors, axis=0)
                wet_errors.append(wet_error)
                pair_effects = {}
                for a, b in pairs:
                    label = f"r{a['id']}_r{b['id']}"
                    states = (f"reverb@{wet}#r{a['id']}", f"reverb@{wet}#r{b['id']}")
                    pair_effects[label] = contrasts(factorial(data, trials, states))
                pooled = {k: np.mean([row[k] for row in pair_effects.values()], axis=0)
                          for k in next(iter(pair_effects.values()))}
                # Post-hoc sensitivity for the shortest pair, which contains an outdoor response.
                shortest_pair = next(iter(pair_effects))
                without_shortest = {k: np.mean([row[k] for label, row in pair_effects.items()
                                                if label != shortest_pair], axis=0)
                                    for k in pooled}
                model_out["levels"][wet] = {
                    "posthoc_excluding_shortest_pair": summarize(trials, without_shortest),
                    "pooled": summarize(trials, pooled | {
                        "wet_genuine_only_error_pct": wet_error, "wet_genuine_only_minus_clean_pp": wet_error - clean}),
                    "per_pair": {k: summarize(trials, v) for k, v in pair_effects.items()}}
            prior = archived[dataset][model]["reverb"]
            for observed, reference in ((trials.weights @ clean, prior["clean_err_pct"]),
                                        (trials.weights @ np.mean(wet_errors, axis=0), prior["err_pct"])):
                delta = validate(observed, reference, f"T1 {dataset} {model}")
                results["max_replay_difference_pp"] = max(results["max_replay_difference_pp"], delta)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jvs", required=True, type=Path)
    parser.add_argument("--t1", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    contents = {"bootstrap_replicates": N_BOOT, "numpy_version": np.__version__,
                "sources": {"jvs": str(args.jvs.resolve()), "t1": str(args.t1.resolve())},
                "jvs": jvs_analysis(args.jvs), "measured_responses": room_analysis(args.t1)}
    args.output.write_text(json.dumps(contents, indent=2) + "\n")
    print(f"Saved {args.output}")
    print("Max replay differences (pp):", contents["jvs"]["max_replay_difference_pp"],
          contents["measured_responses"]["max_replay_difference_pp"])


if __name__ == "__main__":
    main()
