"""F3: low-pass the clean F2 clips to test the SSL bandwidth explanation.

  .venv/bin/python -I pilots/f3_lowpass.py
Writes data/f3/manifest.json with crops: clean and mp3@128 (reused F2 files) plus
lp@16000 and lp@11000. Then: embed.py ssl f3 ; f2_analyze.py f3
"""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "f3"


def main() -> None:
    records = []
    for r in json.loads((ROOT / "data" / "f2" / "manifest.json").read_text()):
        crops = {k: r["crops"][k] for k in ("clean", "mp3@128")}
        x, sr = sf.read(crops["clean"][0], dtype="float32")
        for cutoff in (16000, 11000):
            y = sosfiltfilt(butter(8, cutoff, fs=sr, output="sos"), x)
            path = OUT / "crops" / f"lp{cutoff}" / f"{r['singer']}_{r['song']}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, y, sr, subtype="FLOAT")
            crops[f"lp@{cutoff}"] = [str(path)]
        records.append({k: r[k] for k in ("singer", "sex", "song")} | {"crops": crops})
    (OUT / "manifest.json").write_text(json.dumps(records, indent=1))
    print(f"done: {len(records)} clips")


if __name__ == "__main__":
    main()
