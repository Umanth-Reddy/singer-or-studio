"""E1: embed every crop of data/<pilot>/manifest.json with an extra frozen encoder.

  .venv/bin/python -I pilots/embed_extra.py <ecapa|mert> <pilot>
Protocol: docs/04_pilot_log.md (E1). ECAPA: speechbrain/spkrec-ecapa-voxceleb on the whole
16 kHz crop. MERT: m-a-p/MERT-v1-95M at 24 kHz, time-mean of the all-layer average.
Skips the same silent-quarter crops as embed.py. Writes data/<pilot>/emb_<model>.npz.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

HERE = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ecapa():
    from speechbrain.inference.speaker import EncoderClassifier

    model = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": DEVICE},
                                           savedir=str(HERE.parent / "data" / "models" / "ecapa"))
    return 16000, lambda w: model.encode_batch(w[None])[0, 0]


def mert(source="m-a-p/MERT-v1-95M", local_files_only=False):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        source, trust_remote_code=True, local_files_only=local_files_only
    ).to(DEVICE).eval()
    # Recent transformers no longer return hidden_states from MERT's remote code, so hook the
    # 13 states directly: encoder input after layer_norm (state 0) and each of the 12 layers.
    hs: list[torch.Tensor] = []
    grab = lambda _m, _i, out: hs.append(out[0] if isinstance(out, tuple) else out)  # noqa: E731
    model.encoder.layer_norm.register_forward_hook(grab)
    for layer in model.encoder.layers:
        layer.register_forward_hook(grab)

    def embed(w: torch.Tensor) -> torch.Tensor:
        w = (w - w.mean()) / (w.std() + 1e-7)  # MERT feature extractor normalisation
        hs.clear()
        model(w[None])
        assert len(hs) == 13
        return torch.stack(hs).mean(0)[0].mean(0)

    return 24000, embed


def main(model: str, pilot: str) -> None:
    out = HERE.parent / "data" / pilot
    manifest = json.loads((out / "manifest.json").read_text())
    paths = [p for r in manifest for crops in r["crops"].values() for p in crops]
    skipped = [p for p in paths if not np.abs(sf.read(p, dtype="float32")[0]).reshape(4, -1).max(1).min() > 0]
    paths = [p for p in paths if p not in set(skipped)]
    sr, embed = {"ecapa": ecapa, "mert": mert}[model]()
    vectors = []
    with torch.no_grad():
        for k, p in enumerate(paths):
            x, file_sr = sf.read(p, dtype="float32")
            w = torchaudio.functional.resample(torch.from_numpy(x), file_sr, sr).to(DEVICE)
            vectors.append(embed(w).float().cpu().numpy())
            if k % 200 == 0:
                print(f"{model} {pilot}: {k}/{len(paths)}", flush=True)
    np.savez_compressed(out / f"emb_{model}.npz", vectors=np.stack(vectors), paths=np.array(paths),
                        skipped=np.array(skipped))
    print(f"wrote {len(paths)} {model} embeddings; skipped {len(skipped)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
