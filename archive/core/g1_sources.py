"""G1 step 1: pick the source window for each GTSinger clip (F1-style source manifest).

  .venv/bin/python -I pilots/g1_sources.py
Protocol: docs/04_pilot_log.md (G1). Writes data/g1/sources.json; render with
  .venv/bin/python -I pilots/f2_render.py data/g1/sources.json g1
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
GTSINGER = ROOT.parent / "Q1-candidate-3" / ".external_data"
SR_WIN, HOP_S = 4.0, 0.5
SEX = {"Soprano": "female", "Alto": "female", "Tenor": "male", "Bass": "male"}


def main() -> None:
    records = []
    for singer_dir in sorted(GTSINGER.glob("*/*-*-*")):
        files = sorted(singer_dir.glob("*/*/Control_Group/*.wav"))
        for slot, f in enumerate(files):
            x, sr = sf.read(f, dtype="float32", always_2d=True)
            x = x.mean(1)
            win, hop = int(SR_WIN * sr), int(HOP_S * sr)
            start = max(range(0, len(x) - win + 1, hop), key=lambda s: float(np.mean(x[s:s + win] ** 2)))
            # f2_render crops at 44.1 kHz after resampling; store the start in those samples.
            records.append({"singer": singer_dir.name, "sex": SEX[singer_dir.name.split("-")[1]],
                            "song": f"s{slot}", "source": str(f), "start": int(round(start * 44100 / sr))})
    out = ROOT / "data" / "g1" / "sources.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=1))
    print(f"{len(records)} clips from {len({r['singer'] for r in records})} singers")


if __name__ == "__main__":
    main()
