"""M1 step 1: clean + production-augmented training crops for the TRAIN singers only.

  .venv/bin/python -I pilots/m1_render.py
Protocol: docs/04_pilot_log.md (M1). Sources: VocalSet arpeggios + scales, straight
technique (no excerpts: those are the F2 test material). Accompaniments come from
MUSDB *train*; reverb seeds differ from the F1/F2 test seed (0).
Writes data/m1/crops/... and data/m1/manifest.json (one record per clean crop, with
"clean" and "aug0".."aug5" crops and the family used for each augmentation).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
from f1_render import SR, VOCALSET, WIN, comp, load_accomp, make_codec, make_mixsep, reverb, shelf  # noqa: E402
from f2_render import mp3  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "m1"
TRAIN = ["female1", "female2", "female3", "female4", "male1", "male2", "male3", "male4", "male5"]
N_WIN, N_AUG = 3, 6
FAMILIES = ("reverb", "codec", "eq", "mixsep", "comp")
MUSDB_TRAIN = ROOT.parent / "datasets" / "PS_Acoustics" / "MUSDB18" / "train"


def make_augment(rng: random.Random, accomps: list[np.ndarray], codec, mixsep):
    """Random production change from FAMILIES; returns (audio, params). Shared by M1 and C1."""

    def augment(x: np.ndarray, family: str) -> tuple[np.ndarray, dict]:
        if family == "reverb":
            p = {"wet": rng.uniform(0.05, 0.5), "rt60": rng.uniform(0.3, 1.5), "seed": rng.randint(1, 10**6)}
            return reverb(x, **p), p
        if family == "codec":
            if rng.random() < 0.5:
                p = {"encodec_kbps": rng.choice([1.5, 3.0, 6.0, 12.0, 24.0])}
                return codec(x, kbps=p["encodec_kbps"]), p
            p = {"mp3_kbps": rng.choice([32, 64, 96, 128])}
            return mp3(x, p["mp3_kbps"]), p
        if family == "eq":
            p = {"low_hz": rng.uniform(100, 400), "low_db": rng.uniform(-12, 12),
                 "high_hz": rng.uniform(2000, 8000), "high_db": rng.uniform(-12, 12)}
            return shelf(shelf(x, p["low_hz"], p["low_db"], high=False), p["high_hz"], p["high_db"], high=True), p
        if family == "mixsep":
            p = {"var_db": rng.uniform(-6, 12), "accomp": rng.randrange(len(accomps))}
            return mixsep(x, var_db=p["var_db"], accomp=accomps[p["accomp"]]), p
        return comp(x), {}

    return augment


def main() -> None:
    rng = random.Random(0)
    codec, mixsep = make_codec(), make_mixsep()
    songs = sorted(MUSDB_TRAIN.glob("*.stem.mp4"))
    accomps = [load_accomp(p) for p in rng.sample(songs, 8)]

    augment = make_augment(rng, accomps, codec, mixsep)

    def save(y: np.ndarray, name: str) -> str:
        y = np.asarray(y, dtype=np.float32)[:WIN]
        y = y / (np.max(np.abs(y)) + 1e-9) * 0.5
        path = OUT / "crops" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, y, SR, subtype="FLOAT")
        return str(path)

    records = []
    for singer in TRAIN:
        files = sorted((VOCALSET / singer).glob("*/straight/*.wav"))
        files = [f for f in files if f.parent.parent.name in ("arpeggios", "scales")]
        for f in files:
            x, sr = sf.read(f, dtype="float32", always_2d=True)
            x = x.mean(1)
            if sr != SR:
                x = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, SR)[0].numpy()
            starts = list(range(0, len(x) - WIN + 1, WIN))
            best = sorted(sorted(starts, key=lambda s: -float(np.mean(x[s:s + WIN] ** 2)))[:N_WIN])
            for k, s in enumerate(best):
                clip = x[s:s + WIN]
                if not np.abs(clip).reshape(4, -1).max(1).min() > 0:
                    continue
                stem = f"{singer}/{f.stem}_w{k}"
                rec = {"singer": singer, "sex": singer.rstrip("0123456789"), "source": str(f), "start": s,
                       "crops": {"clean": [save(clip, f"{stem}_clean.wav")]}, "families": {}}
                for a in range(N_AUG):
                    family = rng.choice(FAMILIES)
                    y, params = augment(clip, family)
                    rec["crops"][f"aug{a}"] = [save(y, f"{stem}_aug{a}.wav")]
                    rec["families"][f"aug{a}"] = {"family": family, **params}
                records.append(rec)
        print(f"{singer}: {sum(r['singer'] == singer for r in records)} clean crops", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(records, indent=1))
    print(f"done: {len(records)} clean crops x {1 + N_AUG}")


if __name__ == "__main__":
    main()
