"""T1 step 1: render the F2a/G1 clean windows under 8 real rooms and 5 held-out accompaniments.

  .venv/bin/python -I TEMP/render.py f2   # VocalSet (sources: data/f1/manifest.json)
  .venv/bin/python -I TEMP/render.py g1   # GTSinger (sources: data/g1/sources.json)
Protocol: TEMP/PROTOCOL.md (T1). Reuses the pilot processing code unchanged; only the room and
the accompaniment differ. Writes TEMP/data/<pilot>/crops/<variant>/... and manifest.json, with
variants "clean", "reverb@<wet>#r<k>" and "mixsep@<var>#a<k>".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import fftconvolve

TEMP = Path(__file__).resolve().parent
ROOT = TEMP.parent
sys.path.insert(0, str(ROOT / "pilots"))  # -I drops the script dir
from f1_render import SR, WIN, load_accomp, make_mixsep  # noqa: E402

SOURCES = {"f2": ROOT / "data" / "f1" / "manifest.json", "g1": ROOT / "data" / "g1" / "sources.json"}
MUSDB_TEST = ROOT.parent / "datasets" / "PS_Acoustics" / "MUSDB18" / "test"
WETS, VARS = (0.1, 0.2, 0.3, 0.5), (12.0, 6.0, 0.0, -6.0)


def load_rir(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = torchaudio.functional.resample(torch.from_numpy(x.mean(1))[None], sr, SR)[0].numpy()
    x = x[int(np.argmax(np.abs(x))):]  # start at the direct path, like the synthetic RIR
    return x / np.sqrt(np.sum(x ** 2))


def reverb(x: np.ndarray, rir: np.ndarray, wet: float) -> np.ndarray:
    """Same dry/wet mix as f1_render.reverb, with a measured RIR instead of decaying noise."""
    tail = fftconvolve(x, rir)[: len(x)]
    return (1 - wet) * x + wet * tail * (np.std(x) / (np.std(tail) + 1e-9))


def main(pilot: str) -> None:
    out = TEMP / "data" / pilot
    sel = json.loads((TEMP / "data" / "selection.json").read_text())
    rirs = [load_rir(TEMP / "data" / "rirs_mit" / "Audio" / r["file"]) for r in sel["rooms"]]
    accomps = [load_accomp(MUSDB_TEST / a) for a in sel["accompaniments"]]
    mixsep = make_mixsep()
    records = []
    for r in json.loads(SOURCES[pilot].read_text()):
        x, sr = sf.read(r["source"], dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != SR:
            x = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, SR)[0].numpy()
        variants = {"clean": x}
        variants |= {f"reverb@{w:g}#r{k}": reverb(x, rir, w) for k, rir in enumerate(rirs) for w in WETS}
        variants |= {f"mixsep@{v:g}#a{k}": mixsep(x, var_db=v, accomp=a) for k, a in enumerate(accomps) for v in VARS}
        rec = {k: r[k] for k in ("singer", "sex", "song", "source", "start")}
        rec["crops"] = {}
        for name, y in variants.items():
            y = np.asarray(y, dtype=np.float32)[r["start"]:r["start"] + WIN]
            y = y / (np.max(np.abs(y)) + 1e-9) * 0.5  # same peak level as f2_render
            path = out / "crops" / name / f"{r['singer']}_{r['song']}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, y, SR, subtype="FLOAT")
            rec["crops"][name] = [str(path)]
        records.append(rec)
        print(f"{r['singer']} {r['song']} done", flush=True)
    (out / "manifest.json").write_text(json.dumps(records, indent=1))
    print(f"done: {len(records)} clips x {len(records[0]['crops'])} variants")


if __name__ == "__main__":
    main(sys.argv[1])
