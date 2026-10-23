"""C1 steps 2-3: fine-tune CVSM-A per arm, and embed benchmark crops on the GPU.

  .venv-tf/bin/python -I pilots/c1_train.py train <A|B|Bnorev> [steps]
  .venv-tf/bin/python -I pilots/c1_train.py embed <base|art|A|B|Bnorev> <pilot> [step]
Protocol: docs/04_pilot_log.md (C1). Pairs follow existing-work/CVSM/pretraining.py (CVSM-A);
arms differ only in where the positive comes from. Checkpoints: data/c1/ckpt_<arm>/ every
1000 steps. "embed" re-implements tests_before_koding/gate1_experiment.py::run_cvsm (same
centre 4 s crop, per-second L2, EfficientNet "encoder" output averaged over the 4 windows)
and writes data/<pilot>/emb_c1<arm>[_s<step>].npz like embed.py ("base" = original CVSM-A, "art" = original CVSM-ART).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# The default BFC allocator fragmented and hit OOM at ~1.4 GB in use on the 4 GB GPU.
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
import tensorflow as tf  # noqa: E402

for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

HERE = Path(__file__).resolve().parent
PS = HERE.parent.parent
sys.path.insert(0, str(PS / "existing-work" / "CVSM"))
sys.path.insert(0, str(PS / "tests_before_koding"))
from cola import constants  # noqa: E402
from gate1_experiment import center_four_seconds, l2  # noqa: E402
from mscol import network  # noqa: E402

DATA = HERE.parent / "data"
BASE = PS / "existing-work" / "CVSM" / "models" / "open-unmix" / "m4all_cvsma"
PRETRAINED = {"base": BASE, "art": PS / "existing-work" / "CVSM" / "models" / "htdemucs" / "m4all_cvsmart"}
LR, BATCH, STEPS, SEED, P_AUG, WIN = 1e-4, 32, 3000, 0, 0.8, 16000


def build(ckpt_dir: Path) -> tf.keras.Model:
    def new_model():
        return network.get_contrastive_network(embedding_dim=512, temperature=0.2, pooling_type="max",
                                               similarity_type=constants.SimilarityMeasure.BILINEAR)

    # A TensorFlow checkpoint may defer restoring its optimizer until compile.
    # Copy only model weights so fine-tuning starts with a fresh Adam state.
    source = new_model()
    status = source.load_weights(tf.train.latest_checkpoint(str(ckpt_dir)))
    status.assert_existing_objects_matched()
    status.expect_partial()
    weights = source.get_weights()
    del source
    model = new_model()
    model.set_weights(weights)
    # run_eagerly: the authors' train_step uses Python len() on dynamic tensors.
    # jit_compile=False: Adam's per-variable XLA update steps were retraced and leaked GPU memory.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR, jit_compile=False),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], run_eagerly=True)
    return model


def unit(w: np.ndarray) -> np.ndarray:
    return w / (np.linalg.norm(w, axis=-1, keepdims=True) + 1e-9)


def batches(d: dict, arm: str, rng: np.random.Generator):
    """Endless (anchor, positive) batches. All random draws are made for every arm so the
    streams stay aligned; only the positive's source differs."""
    voc, acc, aug, fam = d["voc"], d["acc"], d["aug"], d["fam"]
    n = len(voc)
    while True:
        # Each row is its own class in the in-batch contrastive loss. Reusing a
        # crop within a batch would make its other occurrence a false negative.
        i = rng.choice(n, BATCH, replace=False)
        j = np.where(rng.random(BATCH) < 0.5, rng.integers(0, n, BATCH), i)  # 50 % swapped accompaniment
        s, s2 = rng.integers(0, 3 * WIN + 1, BATCH), rng.integers(0, 3 * WIN + 1, BATCH)
        w = rng.uniform(0.6, 0.8, BATCH)[:, None]
        use_aug, pick = rng.random(BATCH) < P_AUG, rng.random(BATCH)
        cut = lambda x, rows, st: np.stack([x[r, t:t + WIN] for r, t in zip(rows, st)]).astype(np.float32)  # noqa: E731
        anchors = w * cut(acc, j, s) + (1 - w ** 2) * cut(voc, i, s)
        positives = cut(voc, i, s2)
        if not arm.startswith("A"):
            for b in np.flatnonzero(use_aug):
                ok = [k for k in range(aug.shape[1]) if not (arm.startswith("Bnorev") and fam[i[b], k] == "reverb")]
                if ok:
                    positives[b] = aug[i[b], ok[int(pick[b] * len(ok))], s2[b]:s2[b] + WIN]
        yield unit(anchors), unit(positives)


def train(arm: str, steps: int) -> None:
    """arm "<A|B|Bnorev>[_b<batch>]"; the suffix overrides BATCH (screening runs)."""
    global BATCH
    if "_b" in arm:
        BATCH = int(arm.split("_b")[1])
    tf.keras.utils.set_random_seed(SEED)
    d = dict(np.load(DATA / "c1" / "train.npz"))
    model = build(BASE)
    out = DATA / "c1" / f"ckpt_{arm}"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = tf.train.Checkpoint(model)
    gen, t0, log = batches(d, arm, np.random.default_rng(SEED)), time.time(), []
    # One eager train_step per batch (fit() in a loop retraced and leaked GPU memory).
    for step in range(1, steps + 1):
        a, p = next(gen)
        r = {k: float(v) for k, v in model.train_step(tf.constant(np.stack([a, p], axis=-1))).items()}
        if step % 100 == 0:  # metrics are running means since the last reset
            log.append({"step": step, **r})
            print(f"{arm} step {step}/{steps} loss {r['loss']:.3f} acc {r['sparse_categorical_accuracy']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            model.reset_metrics()
        if step % 1000 == 0:
            ckpt.write(str(out / f"step{step}"))
    (out / "train_log.json").write_text(json.dumps(log))


def embed(arm: str, pilot: str, step: int | None) -> None:
    ckpt_dir = PRETRAINED.get(arm, DATA / "c1" / f"ckpt_{arm}")
    model = network.get_contrastive_network(embedding_dim=512, temperature=0.2, pooling_type="max",
                                            similarity_type=constants.SimilarityMeasure.BILINEAR)
    if arm in PRETRAINED:
        status = model.load_weights(tf.train.latest_checkpoint(str(ckpt_dir)))
    else:
        status = tf.train.Checkpoint(model).read(str(ckpt_dir / f"step{step or STEPS}"))
    status.assert_existing_objects_matched()
    status.expect_partial()
    encoder = model.embedding_model.get_layer("encoder")
    mel_w = tf.signal.linear_to_mel_weight_matrix(64, 513, 16000, 60, 7800)

    out = DATA / pilot
    manifest = json.loads((out / "manifest.json").read_text())
    paths = [p for r in manifest for crops in r["crops"].values() for p in crops]
    skipped = [p for p in paths if not np.abs(sf.read(p, dtype="float32")[0]).reshape(4, -1).max(1).min() > 0]
    paths = [p for p in paths if p not in set(skipped)]
    vectors = []
    for k in range(0, len(paths), 64):
        wins = np.stack([np.stack([l2(w) for w in center_four_seconds(Path(p), 16000).reshape(4, WIN)])
                         for p in paths[k:k + 64]])  # (b, 4, 16000)
        spec = tf.abs(tf.signal.stft(wins.reshape(-1, WIN), 400, 160, fft_length=1024))
        feats = tf.expand_dims(tf.math.log(tf.clip_by_value(tf.tensordot(spec, mel_w, 1), 1e-5, 1e8)), -1)
        e = encoder(feats, training=False).numpy().reshape(len(wins), 4, -1).mean(1)
        vectors.extend(l2(v) for v in e)
    name = f"emb_c1{arm}" + (f"_s{step}" if step else "")
    np.savez_compressed(out / f"{name}.npz", vectors=np.stack(vectors), paths=np.array(paths), skipped=np.array(skipped))
    print(f"wrote {len(paths)} {name} embeddings; skipped {len(skipped)}")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else STEPS)
    else:
        embed(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else None)
