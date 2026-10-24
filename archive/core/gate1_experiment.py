from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_ROOT = Path(__file__).resolve().parent / "evidence"
MANIFEST = EVIDENCE_ROOT / "manifest.csv"
CHECKPOINT_ROOT = Path(__file__).resolve().parent / "checkpoints"
CVSM_REPO = PROJECT_ROOT / "existing-work" / "CVSM"
CVSM_CHECKPOINT_DIR = CVSM_REPO / "models" / "htdemucs" / "m4all_cvsmart"
SSL_CHECKPOINT = CHECKPOINT_ROOT / "ssl_contrastive_vc" / "model.ts"
PILOT_ROOT = EVIDENCE_ROOT / "gate1_pilot"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    chosen = [
        row
        for row in rows
        if row["dataset_id"] == "vocalset_v1_0"
        and "/excerpts/" in row["audio_path"]
        and row["expression_label"] in {"straight", "vibrato"}
    ]
    for row in chosen:
        stem = Path(row["audio_path"]).stem.split("_")
        if len(stem) not in {2, 3}:
            raise ValueError(f"unexpected excerpt filename: {row['audio_path']}")
        row["excerpt_key"] = stem[-2]
    chosen.sort(key=lambda row: row["clip_id"])
    return chosen


def center_four_seconds(path: Path, target_sr: int) -> np.ndarray:
    audio, source_sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if source_sr != target_sr:
        divisor = math.gcd(source_sr, target_sr)
        audio = resample_poly(audio, target_sr // divisor, source_sr // divisor).astype(
            np.float32
        )
    target_samples = 4 * target_sr
    if len(audio) >= target_samples:
        start = (len(audio) - target_samples) // 2
        audio = audio[start : start + target_samples]
    else:
        deficit = target_samples - len(audio)
        audio = np.pad(audio, (deficit // 2, deficit - deficit // 2))
    if not np.isfinite(audio).all():
        raise ValueError(f"non-finite audio: {path}")
    return audio


def l2(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("embedding is zero or non-finite")
    return vector / norm


def run_cvsm(rows: list[dict[str, str]]) -> np.ndarray:
    import tensorflow as tf

    sys.path.insert(0, str(CVSM_REPO))
    from cola import constants
    from mscol import network

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    model = network.get_contrastive_network(
        embedding_dim=512,
        temperature=0.2,
        pooling_type="max",
        similarity_type=constants.SimilarityMeasure.BILINEAR,
    )
    checkpoint = tf.train.latest_checkpoint(str(CVSM_CHECKPOINT_DIR))
    if checkpoint is None:
        raise FileNotFoundError(f"no CVSM checkpoint in {CVSM_CHECKPOINT_DIR}")
    restore_status = model.load_weights(checkpoint)
    restore_status.assert_existing_objects_matched()
    restore_status.expect_partial()  # Training optimizer state is unused in inference.
    encoder = model.embedding_model.get_layer("encoder")

    vectors: list[np.ndarray] = []
    for index, row in enumerate(rows, start=1):
        source = (MANIFEST.parent / row["audio_path"]).resolve()
        windows = center_four_seconds(source, 16000).reshape(4, 16000)
        # The authors normalize their preprocessed one-second source windows.
        windows = np.stack([l2(window) for window in windows])
        waveforms = tf.convert_to_tensor(windows)
        spec = tf.abs(tf.signal.stft(waveforms, 400, 160, fft_length=1024))
        mel_weights = tf.signal.linear_to_mel_weight_matrix(
            64, 513, 16000, 60, 7800
        )
        mel = tf.tensordot(spec, mel_weights, 1)
        features = tf.expand_dims(tf.math.log(tf.clip_by_value(mel, 1e-5, 1e8)), -1)
        embedding = encoder(features, training=False).numpy().mean(axis=0)
        vectors.append(l2(embedding))
        if index % 20 == 0 or index == len(rows):
            print(f"CVSM: {index}/{len(rows)} clips", flush=True)
    return np.stack(vectors)


def run_ssl(rows: list[dict[str, str]]) -> np.ndarray:
    import torch

    if not SSL_CHECKPOINT.is_file():
        raise FileNotFoundError(SSL_CHECKPOINT)
    torch.set_num_threads(4)
    model = torch.jit.load(str(SSL_CHECKPOINT), map_location="cpu").eval()
    vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            source = (MANIFEST.parent / row["audio_path"]).resolve()
            waveform = center_four_seconds(source, 44100)
            peak = float(np.max(np.abs(waveform)))
            if peak <= 0:
                raise ValueError(f"silent SSL input: {source}")
            waveform = waveform / peak  # Match the authors' eval.py amplitude normalization.
            embedding = model(torch.from_numpy(waveform[None, :])).detach().cpu().numpy()
            vectors.append(l2(embedding.reshape(-1)))
            if index % 20 == 0 or index == len(rows):
                print(f"Singer Identity SSL: {index}/{len(rows)} clips", flush=True)
    return np.stack(vectors)


def save_embeddings(model_name: str, rows: list[dict[str, str]], vectors: np.ndarray) -> Path:
    PILOT_ROOT.mkdir(parents=True, exist_ok=True)
    repo = CVSM_REPO if model_name == "cvsm" else PROJECT_ROOT / "existing-work" / "ssl-singer-identity"
    if model_name == "cvsm":
        checkpoint_paths = sorted(CVSM_CHECKPOINT_DIR.glob("checkpoint-*.data-*")) + sorted(
            CVSM_CHECKPOINT_DIR.glob("checkpoint-*.index")
        )
    else:
        checkpoint_paths = [SSL_CHECKPOINT]
    provenance = {
        "model": model_name,
        "source_repo_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "manifest_sha256": sha256(MANIFEST),
        "checkpoint_sha256": {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in checkpoint_paths},
        "source_clip_ids": [row["source_clip_id"] for row in rows],
        "clip_ids": [row["clip_id"] for row in rows],
        "preprocessing": "central 4s; mono; CVSM 16kHz/four one-second L2 windows/encoder mean; SSL 44.1kHz/peak-normalized four-second scripted encoder",
    }
    output = PILOT_ROOT / f"{model_name}_embeddings.npz"
    np.savez_compressed(output, vectors=vectors, provenance=json.dumps(provenance, sort_keys=True))
    return output


def score(
    model_name: str,
    rows: list[dict[str, str]],
    embeddings: np.ndarray,
    *,
    expected_independent_singers: int | None = 20,
) -> dict[str, object]:
    by_key = {(row["singer_id"], row["excerpt_key"], row["expression_label"]): index for index, row in enumerate(rows)}
    singers = sorted({row["singer_id"] for row in rows})
    songs = sorted({row["excerpt_key"] for row in rows})
    query_results: list[dict[str, object]] = []
    for singer in singers:
        for song in songs:
            anchor = by_key.get((singer, song, "straight"))
            positive = by_key.get((singer, song, "vibrato"))
            if anchor is None or positive is None:
                continue
            impostors = [
                by_key[(other, song, "straight")]
                for other in singers
                if other != singer and (other, song, "straight") in by_key
            ]
            if not impostors:
                continue
            positive_similarity = float(embeddings[anchor] @ embeddings[positive])
            max_impostor_similarity = float(max(embeddings[anchor] @ embeddings[index] for index in impostors))
            query_results.append(
                {
                    "singer_id": singer,
                    "excerpt_key": song,
                    "anchor_clip_id": rows[anchor]["clip_id"],
                    "positive_clip_id": rows[positive]["clip_id"],
                    "positive_similarity": positive_similarity,
                    "max_impostor_similarity": max_impostor_similarity,
                    "error": max_impostor_similarity >= positive_similarity,
                    "impostor_count": len(impostors),
                }
            )
    by_singer: dict[str, list[bool]] = defaultdict(list)
    for result in query_results:
        by_singer[str(result["singer_id"])].append(bool(result["error"]))
    singer_rates = np.array([np.mean(errors) for singer, errors in sorted(by_singer.items())])
    if expected_independent_singers is not None and len(singer_rates) != expected_independent_singers:
        raise ValueError(
            f"expected {expected_independent_singers} independent singers, got {len(singer_rates)}"
        )
    rng = np.random.default_rng(20261006)
    sample_indices = rng.integers(0, len(singer_rates), size=(10000, len(singer_rates)))
    bootstrap = singer_rates[sample_indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])

    # Added after viewing the primary results. This diagnoses general domain
    # transfer and must not be counted toward the prespecified gate criterion.
    control_errors: list[bool] = []
    for singer in singers:
        for query_song in songs:
            for target_song in songs:
                if query_song == target_song:
                    continue
                anchor = by_key.get((singer, query_song, "straight"))
                positive = by_key.get((singer, target_song, "straight"))
                if anchor is None or positive is None:
                    continue
                impostors = [
                    by_key[(other, target_song, "straight")]
                    for other in singers
                    if other != singer and (other, target_song, "straight") in by_key
                ]
                control_errors.append(
                    max(float(embeddings[anchor] @ embeddings[index]) for index in impostors)
                    >= float(embeddings[anchor] @ embeddings[positive])
                )
    return {
        "model": model_name,
        "queries": len(query_results),
        "independent_singers": len(singer_rates),
        "errors": int(sum(bool(result["error"]) for result in query_results)),
        "mean_singer_error_rate": float(singer_rates.mean()),
        "bootstrap_95_ci": [float(ci_low), float(ci_high)],
        "material_failure_threshold": 0.20,
        "material_failure_supported": bool(ci_low >= 0.20),
        "exploratory_same_technique_different_song_control": {
            "errors": int(sum(control_errors)),
            "queries": len(control_errors),
            "error_rate": float(np.mean(control_errors)),
            "status": "post_hoc_diagnostic_not_gate_evidence",
        },
        "query_results": query_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["cvsm", "ssl", "score"])
    args = parser.parse_args()
    rows = query_rows()
    if args.mode == "score":
        results = []
        for model_name in ("cvsm", "ssl"):
            cache = np.load(PILOT_ROOT / f"{model_name}_embeddings.npz")
            provenance = json.loads(str(cache["provenance"]))
            if provenance["manifest_sha256"] != sha256(MANIFEST):
                raise ValueError(f"stale {model_name} embeddings: manifest changed")
            if provenance["clip_ids"] != [row["clip_id"] for row in rows]:
                raise ValueError(f"stale {model_name} embeddings: query set changed")
            for relative_path, expected_hash in provenance["checkpoint_sha256"].items():
                if sha256(PROJECT_ROOT / relative_path) != expected_hash:
                    raise ValueError(f"stale {model_name} embeddings: checkpoint changed")
            results.append(score(model_name, rows, cache["vectors"]))
        output = PILOT_ROOT / "results.json"
        output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for result in results:
            print(
                result["model"],
                f"errors={result['errors']}/{result['queries']}",
                f"mean_singer_error={result['mean_singer_error_rate']:.3f}",
                f"CI={result['bootstrap_95_ci']}",
                f"material_failure={result['material_failure_supported']}",
            )
        print(f"Results: {output}")
        return

    vectors = run_cvsm(rows) if args.mode == "cvsm" else run_ssl(rows)
    output = save_embeddings(args.mode, rows, vectors)
    print(f"Saved {vectors.shape} embeddings to {output}")


if __name__ == "__main__":
    main()
