"""C1 step 1: training data for the CVSM-A fine-tuning arms.

  .venv/bin/python -I pilots/c1_render.py
Protocol: docs/04_pilot_log.md (C1). K1 FMA crops with vocal_rms >= 0.01: clean vocal,
accompaniment and N_AUG production variants of the vocal (make_augment from M1, MUSDB train
accompaniments, rng seed 1). All at 16 kHz, float16. Writes data/c1/train.npz.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
from f1_render import SR, load_accomp, make_codec, make_mixsep  # noqa: E402
from m1_render import FAMILIES, MUSDB_TRAIN, make_augment  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "c1"
N_AUG, MIN_RMS = 4, 0.01


def to16k(x: np.ndarray) -> np.ndarray:
    return resample_poly(x, 160, 441).astype(np.float32)[: 4 * 16000]


def main() -> None:
    rng = random.Random(1)
    codec, mixsep = make_codec(), make_mixsep()
    songs = sorted(MUSDB_TRAIN.glob("*.stem.mp4"))
    augment = make_augment(rng, [load_accomp(p) for p in rng.sample(songs, 8)], codec, mixsep)
    manifest = json.loads((ROOT / "data" / "k1" / "manifest.json").read_text())
    items = [(r["id"], r["artist"], v, a) for r in manifest
             for v, a, rms in zip(r["crops"]["vocals"], r["crops"]["accomp"], r["vocal_rms"]) if rms >= MIN_RMS]
    voc, acc, aug, fam = [], [], [], []
    for k, (_, _, vpath, apath) in enumerate(items):
        x = sf.read(vpath, dtype="float32")[0]
        assert len(x) == 4 * SR
        variants, families = [], []
        for _ in range(N_AUG):
            f = rng.choice(FAMILIES)
            variants.append(to16k(np.asarray(augment(x, f)[0], dtype=np.float32)[: len(x)]))
            families.append(f)
        voc.append(to16k(x))
        acc.append(to16k(sf.read(apath, dtype="float32")[0]))
        aug.append(np.stack(variants))
        fam.append(families)
        if k % 100 == 0:
            print(f"{k}/{len(items)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "train.npz", voc=np.stack(voc).astype(np.float16), acc=np.stack(acc).astype(np.float16),
             aug=np.stack(aug).astype(np.float16), fam=np.array(fam), track=np.array([i[0] for i in items]),
             artist=np.array([i[1] for i in items]))
    print(f"done: {len(items)} crops x (clean + {N_AUG} variants)")


if __name__ == "__main__":
    main()
