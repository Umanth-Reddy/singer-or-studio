"""P1 step 1: separate vocals with htdemucs and cut vocal-active 4 s crops.

Run with claude-research/.venv (CUDA torch + demucs):
  .venv/bin/python -I pilots/p1_separate.py
Writes data/p1/crops/<set>/<song>_w<i>.wav (44.1 kHz mono) and data/p1/manifest.json.
For MUSDB songs the same windows are also cut from the ground-truth vocal stem
(set "musdb_gt") as the separation control.
"""

from __future__ import annotations

import csv
import json
import random
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
OUT = DATA / "p1"
MUSDB = ROOT.parent / "datasets" / "PS_Acoustics" / "MUSDB18"
SONICS = DATA / "sonics"
SR = 44100
EXCERPT_S = 60
WIN_S = 4
N_WIN = 3
N_PER_GENERATOR = 150
GENERATORS = {"suno": "chirp-v3.5", "udio": "udio-120s"}


def decode(path: Path, start: float, stream: int = 0) -> np.ndarray:
    """Decode EXCERPT_S seconds from `start` as float32 stereo (2, n)."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-t", str(EXCERPT_S), "-i", str(path),
           "-map", f"0:a:{stream}", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def duration(path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    return float(subprocess.run(cmd, capture_output=True, check=True, text=True).stdout.strip())


def top_windows(vocals_mono: np.ndarray) -> tuple[list[int], list[float]]:
    """Start samples of the N_WIN loudest non-overlapping 4 s windows."""
    hop = WIN_S * SR
    starts = list(range(0, len(vocals_mono) - hop + 1, hop))
    rms = [float(np.sqrt(np.mean(vocals_mono[s:s + hop] ** 2))) for s in starts]
    order = np.argsort(rms)[::-1][:N_WIN]
    return [starts[i] for i in sorted(order)], [rms[i] for i in sorted(order)]


def sources() -> list[dict]:
    items = []
    for split in ("train", "test"):
        for path in sorted((MUSDB / split).glob("*.stem.mp4")):
            name = path.name.removesuffix(".stem.mp4")
            items.append({"set": "musdb", "song": name, "artist": name.split(" - ")[0], "path": str(path)})
    if not zipfile.is_zipfile(SONICS / "part_01.zip"):
        print("SONICS part_01.zip incomplete: separating MUSDB only")
        return items
    rows = list(csv.DictReader(open(SONICS / "fake_songs.csv", encoding="utf-8")))
    with zipfile.ZipFile(SONICS / "part_01.zip") as archive:
        members = {Path(n).stem: n for n in archive.namelist() if n.endswith(".mp3")}
    rng = random.Random(0)
    for key, algorithm in GENERATORS.items():
        pool = [r for r in rows if r["algorithm"] == algorithm and r["filename"] in members]
        for r in rng.sample(pool, min(N_PER_GENERATOR, len(pool))):
            items.append({"set": key, "song": r["filename"], "artist": None,
                          "member": members[r["filename"]], "style": r["style"]})
    return items


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model("htdemucs").to(device).eval()
    vocal_index = model.sources.index("vocals")
    items = sources()
    manifest_path = OUT / "manifest.json"
    done = {m["song"]: m for m in json.loads(manifest_path.read_text())} if manifest_path.exists() else {}
    archive = None
    tmp = OUT / "tmp.mp3"
    OUT.mkdir(parents=True, exist_ok=True)
    for n, item in enumerate(items, 1):
        if item["song"] in done:
            continue
        if item["set"] == "musdb":
            path = Path(item["path"])
        else:
            archive = archive or zipfile.ZipFile(SONICS / "part_01.zip")
            tmp.write_bytes(archive.read(item["member"]))
            path = tmp
        start = max(0.0, duration(path) / 2 - EXCERPT_S / 2)
        mix = decode(path, start)
        with torch.no_grad():
            ref = mix.mean(0)
            wav = torch.from_numpy((mix - ref.mean()) / (ref.std() + 1e-8))[None].to(device)
            est = apply_model(model, wav, device=device, split=True, overlap=0.25)[0, vocal_index]
            vocals = (est.cpu().numpy() * (ref.std() + 1e-8) + ref.mean()).mean(0)
        starts, rms = top_windows(vocals)
        record = dict(item, start_s=start, windows=starts, window_rms=rms, crops={})
        variants = {item["set"]: vocals}
        if item["set"] == "musdb":
            variants["musdb_gt"] = decode(path, start, stream=4).mean(0)
        for set_name, signal in variants.items():
            folder = OUT / "crops" / set_name
            folder.mkdir(parents=True, exist_ok=True)
            record["crops"][set_name] = []
            for i, s in enumerate(starts):
                crop = folder / f"{item['song']}_w{i}.wav"
                sf.write(crop, signal[s:s + WIN_S * SR], SR, subtype="FLOAT")
                record["crops"][set_name].append(str(crop))
        done[item["song"]] = record
        if n % 10 == 0:
            manifest_path.write_text(json.dumps(list(done.values()), indent=1))
            print(f"{n}/{len(items)} separated", flush=True)
    manifest_path.write_text(json.dumps(list(done.values()), indent=1))
    print(f"done: {len(done)} songs")


if __name__ == "__main__":
    main()
