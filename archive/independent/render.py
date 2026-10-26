"""Build the audio sets for the Oct-26 checks. Original pilot code is imported, never edited.

  .venv/bin/python -I claude-check-oct-26/render.py rerun    # check 1: re-render F2a key variants from source
  .venv/bin/python -I claude-check-oct-26/render.py loud     # check 2: loudness-matched + pure-gain copies
  .venv/bin/python -I claude-check-oct-26/render.py accomp   # check 3a: 10 new accompaniments, VocalSet + GTSinger
  .venv/bin/python -I claude-check-oct-26/render.py realsong # check 3b: manifest over real MUSDB songs (P1 crops)
Each writes claude-check-oct-26/data/<set>/manifest.json (F2a manifest schema) and crops.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
OUT = HERE / "data"
sys.path.insert(0, str(ROOT / "pilots"))
from f1_render import SR, WIN, eq, load_accomp, make_codec, make_mixsep, reverb  # noqa: E402
from f2_render import mp3  # noqa: E402

KEY = ("clean", "reverb@0.3", "codec@6", "mp3@64", "mixsep@0", "eq@12")
MUSDB_TEST = ROOT.parent / "datasets" / "PS_Acoustics" / "MUSDB18" / "test"


def load_source(r: dict) -> np.ndarray:
    """Same decoding as f2_render: mono, resampled to 44.1 kHz."""
    x, sr = sf.read(r["source"], dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        x = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, SR)[0].numpy()
    return x


def write(name: str, records: list[dict]) -> None:
    (OUT / name).mkdir(parents=True, exist_ok=True)
    (OUT / name / "manifest.json").write_text(json.dumps(records, indent=1))
    print(f"{name}: {len(records)} clips x {len(records[0]['crops'])} variants", flush=True)


def save(name: str, variant: str, r: dict, y: np.ndarray, peak: bool = True) -> str:
    y = np.asarray(y, dtype=np.float32)
    if peak:
        y = y / (np.max(np.abs(y)) + 1e-9) * 0.5  # f2_render's peak normalisation
    path = OUT / name / "crops" / variant / f"{r['singer']}_{r['song']}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, SR, subtype="FLOAT")
    return str(path)


def rerun() -> None:
    codec, mixsep = make_codec(), make_mixsep()
    fns = {"clean": lambda x: x, "reverb@0.3": lambda x: reverb(x, wet=0.3), "codec@6": lambda x: codec(x, kbps=6.0),
           "mp3@64": lambda x: mp3(x, 64), "mixsep@0": lambda x: mixsep(x, var_db=0.0), "eq@12": lambda x: eq(x, gain_db=12.0)}
    stored = {(r["singer"], r["song"]): r for r in json.loads((DATA / "f2" / "manifest.json").read_text())}
    records, diffs = [], {v: [] for v in fns}
    for r in json.loads((DATA / "f1" / "manifest.json").read_text()):
        x = load_source(r)
        rec = {k: r[k] for k in ("singer", "sex", "song", "source", "start")} | {"crops": {}}
        for v, fn in fns.items():
            path = save("rerun", v, r, np.asarray(fn(x))[r["start"]:r["start"] + WIN])
            rec["crops"][v] = [path]
            old = sf.read(stored[(r["singer"], r["song"])]["crops"][v][0], dtype="float32")[0]
            diffs[v].append(float(np.max(np.abs(sf.read(path, dtype="float32")[0] - old))))
        records.append(rec)
    write("rerun", records)
    (OUT / "rerun" / "waveform_diff.json").write_text(json.dumps(
        {v: {"max_abs": max(d), "median_abs": float(np.median(d))} for v, d in diffs.items()}, indent=1))


def loud() -> None:
    """RMS-match every key variant (removes level differences); add pure gain changes of clean."""
    records = []
    for r in json.loads((DATA / "f2" / "manifest.json").read_text()):
        rec = {k: r[k] for k in ("singer", "sex", "song", "source", "start")} | {"crops": {}}
        for v in KEY:
            y = sf.read(r["crops"][v][0], dtype="float32")[0]
            rec["crops"][v + "_rms"] = [save("loud", v + "_rms", r, y * 0.05 / (np.sqrt(np.mean(y ** 2)) + 1e-9), peak=False)]
        clean = sf.read(r["crops"]["clean"][0], dtype="float32")[0]
        for g in (-12, -6, 6):
            rec["crops"][f"gain@{g}"] = [save("loud", f"gain@{g}", r, clean * 10 ** (g / 20), peak=False)]
        rec["crops"]["clean"] = r["crops"]["clean"]
        records.append(rec)
    write("loud", records)


def accomp() -> None:
    used = {"Al James - Schoolboy Facination"} | {Path(a).name.removesuffix(".stem.mp4") for a in
                                                  json.loads((ROOT / "TEMP" / "data" / "selection.json").read_text())["accompaniments"]}
    pool = [p for p in sorted(MUSDB_TEST.glob("*.stem.mp4")) if p.name.removesuffix(".stem.mp4") not in used]
    chosen = random.Random(26).sample(pool, 10)
    accs = [load_accomp(p) for p in chosen]
    mixsep = make_mixsep()
    for name, src in (("accomp_f2", DATA / "f1" / "manifest.json"), ("accomp_g1", DATA / "g1" / "sources.json")):
        records = []
        for r in json.loads(src.read_text()):
            x = load_source(r)
            rec = {k: r[k] for k in ("singer", "sex", "song", "source", "start")} | {"crops": {}}
            rec["crops"]["clean"] = [save(name, "clean", r, x[r["start"]:r["start"] + WIN])]
            for k, a in enumerate(accs):
                y = mixsep(x, var_db=0.0, accomp=a)[r["start"]:r["start"] + WIN]
                rec["crops"][f"mixsep@0#b{k}"] = [save(name, f"mixsep@0#b{k}", r, y)]
            records.append(rec)
            print(f"{name} {r['singer']} {r['song']}", flush=True)
        write(name, records)
    (OUT / "accomp_choice.json").write_text(json.dumps([p.name for p in chosen], indent=1))


def realsong() -> None:
    """Real songs: htdemucs vocals separated from the real MUSDB mix vs the ground-truth vocal stem."""
    records = []
    for r in json.loads((DATA / "p1" / "manifest.json").read_text()):
        if r["set"] != "musdb" or "musdb_gt" not in r["crops"]:
            continue
        records.append({"song": r["song"], "artist": r["artist"],
                        "crops": {"sep": r["crops"]["musdb"], "gt": r["crops"]["musdb_gt"]}})
    write("realsong", records)


if __name__ == "__main__":
    {"rerun": rerun, "loud": loud, "accomp": accomp, "realsong": realsong}[sys.argv[1]]()
