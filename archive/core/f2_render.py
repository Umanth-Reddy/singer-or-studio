"""F2a step 1: intensity sweep of each production change on the F1 clips.

  .venv/bin/python -I pilots/f2_render.py [sources.json] [pilot]   # defaults: F1 manifest, f2
Reuses the clean 4 s windows chosen in F1 (data/f1/manifest.json) and the F1
processing functions with a strength argument. Writes data/f2/crops/<cond>/... and
data/f2/manifest.json (crops keyed "<kind>@<level>", plus "clean").
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
from f1_render import SR, WIN, eq, make_codec, make_mixsep, reverb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
F1 = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "f1" / "manifest.json"
OUT = ROOT / "data" / (sys.argv[2] if len(sys.argv) > 2 else "f2")


def mp3(x: np.ndarray, kbps: int) -> np.ndarray:
    pcm = x.astype(np.float32).tobytes()
    enc = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                          "-c:a", "libmp3lame", "-b:a", f"{kbps}k", "-f", "mp3", "-"],
                         input=pcm, capture_output=True, check=True).stdout
    dec = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         input=enc, capture_output=True, check=True).stdout
    y = np.frombuffer(dec, dtype=np.float32)
    return np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]


def main() -> None:
    codec, mixsep = make_codec(), make_mixsep()
    sweep = {"reverb": (reverb, [0.1, 0.2, 0.3, 0.5]),
             "codec": (lambda x, k: codec(x, kbps=k), [24.0, 12.0, 6.0, 3.0, 1.5]),
             "mp3": (mp3, [128, 64, 32]),
             "eq": (lambda x, g: eq(x, gain_db=g), [3.0, 6.0, 12.0]),
             "mixsep": (lambda x, v: mixsep(x, var_db=v), [12.0, 6.0, 0.0, -6.0])}
    records = []
    for r in json.loads(F1.read_text()):
        x, sr = sf.read(r["source"], dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != SR:
            x = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, SR)[0].numpy()
        rec = {k: r[k] for k in ("singer", "sex", "song", "source", "start")}
        rec["crops"] = {}
        variants = {"clean": x} | {f"{kind}@{lvl:g}": fn(x, lvl) for kind, (fn, levels) in sweep.items() for lvl in levels}
        for name, y in variants.items():
            y = np.asarray(y, dtype=np.float32)[r["start"]:r["start"] + WIN]
            y = y / (np.max(np.abs(y)) + 1e-9) * 0.5
            path = OUT / "crops" / name / f"{r['singer']}_{r['song']}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, y, SR, subtype="FLOAT")
            rec["crops"][name] = [str(path)]
        records.append(rec)
        print(f"{r['singer']} {r['song']} done", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(records, indent=1))
    print(f"done: {len(records)} clips x {len(records[0]['crops'])} variants")


if __name__ == "__main__":
    main()
