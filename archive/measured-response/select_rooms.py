"""T1 step 0: pick 8 MIT IR Survey rooms by T30 quantile and 5 held-out MUSDB test accompaniments.

  .venv/bin/python -I TEMP/select_rooms.py
Protocol: TEMP/PROTOCOL.md (T1). Writes TEMP/data/selection.json. Uses only the RIRs' decay,
never model results.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

TEMP = Path(__file__).resolve().parent
RIRS = TEMP / "data" / "rirs_mit" / "Audio"
MUSDB_TEST = TEMP.parent.parent / "datasets" / "PS_Acoustics" / "MUSDB18" / "test"
USED_ACCOMP = "Al James - Schoolboy Facination.stem.mp4"
N_ROOMS, N_ACCOMP = 8, 5


def t30(path: Path) -> float | None:
    """Broadband T30 from Schroeder backward integration (-5 to -35 dB fit, scaled to 60 dB)."""
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(1)
    x = x[int(np.argmax(np.abs(x))):]  # start at the direct path
    edc = np.cumsum(x[::-1] ** 2)[::-1]
    db = 10 * np.log10(edc / edc[0] + 1e-30)
    i5, i35 = np.argmax(db <= -5), np.argmax(db <= -35)
    if db[-1] > -35 or i35 <= i5:
        return None  # decay too shallow for a T30 fit
    t = np.arange(i5, i35) / sr
    slope = np.polyfit(t, db[i5:i35], 1)[0]
    return float(-60 / slope)


def main() -> None:
    rooms = sorted((p.name, v) for p in sorted(RIRS.glob("*.wav")) if (v := t30(p)) is not None)
    rooms.sort(key=lambda r: r[1])
    t = np.array([r[1] for r in rooms])
    picked = []
    for q in (np.arange(N_ROOMS) * 2 + 1) / (2 * N_ROOMS):
        target = np.quantile(t, q)
        order = np.argsort(np.abs(t - target))
        k = next(int(i) for i in order if rooms[int(i)][0] not in {p["file"] for p in picked})
        picked.append({"file": rooms[k][0], "t30_s": round(rooms[k][1], 3), "quantile": float(q)})
    songs = sorted(p.name for p in MUSDB_TEST.glob("*.stem.mp4") if p.name != USED_ACCOMP)
    accomps = random.Random(0).sample(songs, N_ACCOMP)
    out = {"rooms": picked, "n_rooms_with_t30": len(rooms), "accompaniments": accomps}
    (TEMP / "data" / "selection.json").write_text(json.dumps(out, indent=1))
    for p in picked:
        print(f"{p['t30_s']:6.2f} s  {p['file']}")
    print("accompaniments:", *accomps, sep="\n  ")


if __name__ == "__main__":
    main()
