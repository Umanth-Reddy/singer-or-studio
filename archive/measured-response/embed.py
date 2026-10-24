"""Embed every crop of a manifest with one frozen encoder; output goes to TEMP only.

  ~/ps/.venv-gate1/bin/python -I TEMP/embed.py cvsm_a|cvsm_art <dataset>
  ~/ps/.venv-ssl/bin/python   -I TEMP/embed.py ssl <dataset>
  .venv/bin/python            -I TEMP/embed.py ecapa|mert|wavlm <dataset>
<dataset>: t1_f2 / t1_g1 (TEMP renders) or f2 / g1 (original pilot crops, read-only).
Same encoders and silent-quarter skip as pilots/embed.py and pilots/embed_extra.py.
WavLM (T3): microsoft/wavlm-base-plus-sv x-vector on the whole 16 kHz crop.
Writes emb_<model>.npz next to the TEMP renders (TEMP/data/{f2,g1}) for t1_*, and to
TEMP/data/orig_<dataset>/ for the original pilot crops.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

TEMP = Path(__file__).resolve().parent
ROOT = TEMP.parent
MANIFESTS = {"t1_f2": TEMP / "data" / "f2", "t1_g1": TEMP / "data" / "g1",
             "f2": ROOT / "data" / "f2", "g1": ROOT / "data" / "g1"}


def wavlm():
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()

    def embed(w: "torch.Tensor") -> "torch.Tensor":
        inputs = fe(w.cpu().numpy(), sampling_rate=16000, return_tensors="pt").to(device)
        return model(**inputs).embeddings[0]

    return 16000, embed


def torch_encoder(model: str, paths: list[str]) -> np.ndarray:
    import torch
    import torchaudio

    sys.path.insert(0, str(ROOT / "pilots"))
    import embed_extra

    sr, embed = {"ecapa": embed_extra.ecapa, "mert": embed_extra.mert, "wavlm": wavlm}[model]()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vectors = []
    with torch.no_grad():
        for k, p in enumerate(paths):
            x, file_sr = sf.read(p, dtype="float32")
            w = torchaudio.functional.resample(torch.from_numpy(x), file_sr, sr).to(device)
            vectors.append(embed(w).float().cpu().numpy())
            if k % 500 == 0:
                print(f"{model}: {k}/{len(paths)}", flush=True)
    return np.stack(vectors)


def main(model: str, dataset: str) -> None:
    src = MANIFESTS[dataset]
    manifest = json.loads((src / "manifest.json").read_text())
    paths = [p for r in manifest for crops in r["crops"].values() for p in crops]
    skipped = [p for p in paths if not np.abs(sf.read(p, dtype="float32")[0]).reshape(4, -1).max(1).min() > 0]
    paths = [p for p in paths if p not in set(skipped)]
    if model in ("cvsm_a", "cvsm_art", "ssl"):
        sys.path.insert(0, str(ROOT / "pilots"))
        import embed as pe

        g = pe.g
        if model == "ssl":
            vectors = g.run_ssl([{"audio_path": p} for p in paths])
        else:
            g.CVSM_CHECKPOINT_DIR = pe.CVSM_DIRS[model]
            vectors = g.run_cvsm([{"audio_path": p} for p in paths])
    else:
        vectors = torch_encoder(model, paths)
    out = src if src.is_relative_to(TEMP) else TEMP / "data" / f"orig_{dataset}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"emb_{model}.npz", vectors=vectors, paths=np.array(paths), skipped=np.array(skipped))
    print(f"wrote {len(paths)} {model} embeddings to {out}; skipped {len(skipped)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
