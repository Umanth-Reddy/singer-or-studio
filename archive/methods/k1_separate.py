"""K1 step 1: separate FMA-medium tracks into vocal + accompaniment crops.

  .venv/bin/python -I pilots/k1_separate.py
Uses data/fmam/s00-s02.parquet. Selects every track of artists with >= 2 albums
(max 4 per album) plus 600 random distractor tracks (seed 0). Per track: htdemucs
on the 30 s clip, the 2 loudest non-overlapping 4 s vocal windows, and the SAME
windows cut from the accompaniment (drums + bass + other).
Writes data/k1/crops/{vocals,accomp}/<id>_w<i>.wav and data/k1/manifest.json.
"""

from __future__ import annotations

import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

ROOT = Path(__file__).resolve().parent.parent
SHARDS = [ROOT / "data" / "fmam" / f"s0{i}.parquet" for i in range(3)]
OUT = ROOT / "data" / "k1"
SR = 44100
WIN = 4 * SR
N_WIN = 2
MAX_PER_ALBUM = 4
N_DISTRACTORS = 600


def decode(mp3: bytes) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, input=mp3, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def select(meta: list[dict]) -> list[int]:
    albums = defaultdict(set)
    for r in meta:
        albums[r["artist"]].add(r["album_title"])
    per_album = defaultdict(int)
    chosen = []
    for i, r in enumerate(meta):
        if len(albums[r["artist"]]) >= 2 and per_album[(r["artist"], r["album_title"])] < MAX_PER_ALBUM:
            per_album[(r["artist"], r["album_title"])] += 1
            chosen.append(i)
    others = [i for i, r in enumerate(meta) if len(albums[r["artist"]]) < 2]
    return chosen + random.Random(0).sample(others, min(N_DISTRACTORS, len(others)))


def main() -> None:
    tables = [pq.read_table(s, columns=["artist", "album_title", "title", "license", "audio"]) for s in SHARDS]
    meta = [r for t in tables for r in t.drop(["audio"]).to_pylist()]
    audio = [a for t in tables for a in t.column("audio").to_pylist()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model("htdemucs").to(device).eval()
    vocal_index = model.sources.index("vocals")
    for stem in ("vocals", "accomp"):
        (OUT / "crops" / stem).mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    done = {m["id"]: m for m in json.loads(manifest_path.read_text())} if manifest_path.exists() else {}
    selected = select(meta)
    for n, i in enumerate(selected, 1):
        if i in done:
            continue
        try:
            mix = decode(audio[i]["bytes"])
        except subprocess.CalledProcessError:
            print(f"undecodable track {i}", flush=True)
            continue
        if mix.shape[1] < 2 * WIN:
            continue
        with torch.no_grad():
            ref = mix.mean(0)
            wav = torch.from_numpy((mix - ref.mean()) / (ref.std() + 1e-8))[None].to(device)
            est = apply_model(model, wav, device=device, split=True, overlap=0.25)[0].cpu().numpy()
            est = est * (ref.std() + 1e-8) + ref.mean()
        vocals = est[vocal_index].mean(0)
        accomp = np.delete(est, vocal_index, axis=0).sum(0).mean(0)
        starts = list(range(0, len(vocals) - WIN + 1, WIN))
        rms = [float(np.sqrt(np.mean(vocals[s:s + WIN] ** 2))) for s in starts]
        best = sorted(np.argsort(rms)[::-1][:N_WIN])
        record = dict(meta[i], id=i, distractor=n > len(selected) - N_DISTRACTORS,
                      vocal_rms=[rms[b] for b in best],
                      accomp_rms=[float(np.sqrt(np.mean(accomp[starts[b]:starts[b] + WIN] ** 2))) for b in best],
                      crops={"vocals": [], "accomp": []})
        for k, b in enumerate(best):
            for stem, signal in (("vocals", vocals), ("accomp", accomp)):
                path = OUT / "crops" / stem / f"{i}_w{k}.wav"
                sf.write(path, signal[starts[b]:starts[b] + WIN], SR, subtype="FLOAT")
                record["crops"][stem].append(str(path))
        done[i] = record
        if n % 20 == 0:
            manifest_path.write_text(json.dumps(list(done.values()), indent=1))
            print(f"{n}/{len(selected)} separated", flush=True)
    manifest_path.write_text(json.dumps(list(done.values()), indent=1))
    print(f"done: {len(done)} tracks")


if __name__ == "__main__":
    main()
