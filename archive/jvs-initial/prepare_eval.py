"""Build fixed four-second JVS-MuSiC crops for a frozen external check.

Conditions are clean, MP3 64 kb/s, and deterministic synthetic reverb (20% wet,
RT60=1 s). Processed audio is local and ignored by Git because the dataset terms
prohibit audio redistribution.
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "jvs-music"
INVENTORY = ROOT / "audit" / "jvs_music_inventory.json"
OUT = DATA / "processed"
SR = 24_000
WINDOW = 4 * SR


def crop_center(x: np.ndarray) -> np.ndarray:
    if len(x) < WINDOW:
        pad = WINDOW - len(x)
        x = np.pad(x, (pad // 2, pad - pad // 2))
    start = (len(x) - WINDOW) // 2
    return x[start : start + WINDOW]


def normalize(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x)))
    if peak <= 1e-9 or not np.isfinite(peak):
        raise ValueError("silent or non-finite input")
    return (x / peak * 0.5).astype(np.float32)


def mp3_64(x: np.ndarray) -> np.ndarray:
    enc = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
         "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3", "pipe:1"],
        input=x.astype(np.float32).tobytes(), capture_output=True, check=True,
    ).stdout
    dec = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1"],
        input=enc, capture_output=True, check=True,
    ).stdout
    y = np.frombuffer(dec, dtype=np.float32)
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y[: len(x)]


def reverb_20(x: np.ndarray, seed: int) -> np.ndarray:
    rt60, wet = 1.0, 0.2
    t = np.arange(int(1.2 * rt60 * SR), dtype=np.float64) / SR
    rng = np.random.default_rng(seed)
    rir = rng.standard_normal(len(t)) * np.exp(-6.91 * t / rt60)
    rir /= np.sqrt(np.sum(rir**2))
    tail = fftconvolve(x, rir, mode="full")[: len(x)]
    return ((1 - wet) * x + wet * tail * (np.std(x) / (np.std(tail) + 1e-9))).astype(np.float32)


def voice_groups() -> dict[str, str]:
    groups: dict[str, str] = {}
    with __import__("zipfile").ZipFile(DATA / "jvs_music_ver1.zip") as z:
        for name in z.namelist():
            if name.startswith("jvs_music_ver1/similarity/") and name.endswith(".csv"):
                group = Path(name).stem
                lines = z.read(name).decode("utf-8-sig").splitlines()
                for line in lines:
                    if line.strip():
                        groups[line.split(",", 1)[0]] = group
    return groups


def main() -> None:
    inventory = json.loads(INVENTORY.read_text())
    groups = voice_groups()
    if len(groups) != 100 or set(groups) != set(inventory["singers"]):
        raise ValueError("official similarity groups do not map exactly to all 100 singers")
    records = []
    for record in inventory["records"]:
        source = ROOT / record["path"]
        x, sr = sf.read(source, dtype="float32", always_2d=True)
        if sr != SR or x.shape[1] != 1:
            raise ValueError(f"unexpected format for {source}: sr={sr}, channels={x.shape[1]}")
        x = x[:, 0]
        variants = {
            "clean": x,
            "mp3_64": mp3_64(x),
            "reverb_20": reverb_20(x, seed=1000 + int(record["singer"][3:]) * 2 + (record["song"] == "unique")),
        }
        row = {"singer": record["singer"], "song": record["song"], "group": groups[record["singer"]], "crops": {}}
        for condition, audio in variants.items():
            dest = OUT / condition / record["singer"] / f"{record['song']}.wav"
            dest.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dest, normalize(crop_center(audio)), SR, subtype="FLOAT")
            row["crops"][condition] = str(dest.relative_to(ROOT))
        records.append(row)
        if len(records) % 20 == 0:
            print(f"prepared {len(records)}/{len(inventory['records'])} recordings", flush=True)
    output = Path(__file__).resolve().parent / "manifest.json"
    output.write_text(json.dumps(records, indent=2) + "\n")
    print(f"wrote {len(records)} recordings × 3 conditions to {output}")
    print("matched vocal-range groups:", {g: sum(r["group"] == g for r in records) // 2 for g in sorted(set(groups.values()))})


if __name__ == "__main__":
    main()
