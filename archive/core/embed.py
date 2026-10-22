"""Embed every crop listed in data/<pilot>/manifest.json with a frozen encoder.

  ~/ps/.venv-gate1/bin/python -I pilots/embed.py cvsm_a <pilot>    # label-agnostic CVSM-A
  ~/ps/.venv-gate1/bin/python -I pilots/embed.py cvsm_art <pilot>  # label-informed CVSM-ART
  ~/ps/.venv-ssl/bin/python   -I pilots/embed.py ssl <pilot>
<pilot> is k1 or f1. Writes data/<pilot>/emb_<model>.npz (vectors, paths, skipped).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "tests_before_koding"))
import gate1_experiment as g  # noqa: E402

CVSM_DIRS = {"cvsm_a": g.CVSM_REPO / "models" / "open-unmix" / "m4all_cvsma",
             "cvsm_art": g.CVSM_REPO / "models" / "htdemucs" / "m4all_cvsmart"}


def main(model: str, pilot: str) -> None:
    OUT = HERE.parent / "data" / pilot
    manifest = json.loads((OUT / "manifest.json").read_text())
    paths = [p for r in manifest for crops in r["crops"].values() for p in crops]
    # CVSM L2-normalises each 1 s quarter, so a crop with an all-zero quarter is unusable (all models skip it).
    skipped = [p for p in paths if not np.abs(sf.read(p, dtype="float32")[0]).reshape(4, -1).max(1).min() > 0]
    paths = [p for p in paths if p not in set(skipped)]
    if model in CVSM_DIRS:
        g.CVSM_CHECKPOINT_DIR = CVSM_DIRS[model]
        vectors = g.run_cvsm([{"audio_path": p} for p in paths])
    else:
        vectors = g.run_ssl([{"audio_path": p} for p in paths])
    np.savez_compressed(OUT / f"emb_{model}.npz", vectors=vectors, paths=np.array(paths), skipped=np.array(skipped))
    print(f"wrote {len(paths)} {model} embeddings; skipped {len(skipped)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
