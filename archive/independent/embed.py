"""Embed every crop of claude-check-oct-26/data/<set>/manifest.json with one frozen encoder.

  ~/ps/.venv-gate1/bin/python -I claude-check-oct-26/embed.py <cvsm_art|cvsm_a> <set>
  ~/ps/.venv-ssl/bin/python   -I claude-check-oct-26/embed.py ssl <set>
  .venv/bin/python            -I claude-check-oct-26/embed.py <ecapa|mert|xvect> <set>
<set> may also be an absolute manifest path (embeddings then go next to it as emb_<model>.npz;
only crops not already embedded there are computed). Reuses the original pilot embedding code
unchanged (imported, never edited). Same silent-quarter skip rule as pilots/embed.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parent / "tests_before_koding"))
sys.path.insert(0, str(ROOT / "pilots"))


def encoder(model: str):
    if model in ("cvsm_art", "cvsm_a"):
        import gate1_experiment as g
        g.CVSM_CHECKPOINT_DIR = {"cvsm_a": g.CVSM_REPO / "models" / "open-unmix" / "m4all_cvsma",
                                 "cvsm_art": g.CVSM_REPO / "models" / "htdemucs" / "m4all_cvsmart"}[model]
        return lambda paths: g.run_cvsm([{"audio_path": p} for p in paths])
    if model == "ssl":
        import gate1_experiment as g
        return lambda paths: g.run_ssl([{"audio_path": p} for p in paths])
    import torch
    import torchaudio
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model == "xvect":
        from speechbrain.inference.speaker import EncoderClassifier
        m = EncoderClassifier.from_hparams("speechbrain/spkrec-xvect-voxceleb", run_opts={"device": device},
                                           savedir=str(HERE / "models" / "xvect"))
        sr, fn = 16000, lambda w: m.encode_batch(w[None])[0, 0]
    else:
        import embed_extra
        sr, fn = {"ecapa": embed_extra.ecapa, "mert": embed_extra.mert}[model]()

    def run(paths):
        out = []
        with torch.no_grad():
            for k, p in enumerate(paths):
                x, file_sr = sf.read(p, dtype="float32")
                w = torchaudio.functional.resample(torch.from_numpy(x), file_sr, sr).to(device)
                out.append(fn(w).float().cpu().numpy())
                if k % 200 == 0:
                    print(f"{model}: {k}/{len(paths)}", flush=True)
        return np.stack(out)
    return run


def main(model: str, target: str) -> None:
    manifest_path = Path(target) if target.endswith(".json") else HERE / "data" / target / "manifest.json"
    out = manifest_path.parent / f"emb_{model}.npz"
    manifest = json.loads(manifest_path.read_text())
    paths = sorted({p for r in manifest for crops in r["crops"].values() for p in crops})
    cache = {}
    if out.exists():
        old = np.load(out)
        cache = dict(zip(old["paths"].tolist(), old["vectors"]))
    skipped = [p for p in paths if not np.abs(sf.read(p, dtype="float32")[0]).reshape(4, -1).max(1).min() > 0]
    paths = [p for p in paths if p not in set(skipped)]
    todo = [p for p in paths if p not in cache]
    if todo:
        cache.update(zip(todo, encoder(model)(todo)))
    np.savez_compressed(out, vectors=np.stack([cache[p] for p in paths]), paths=np.array(paths),
                        skipped=np.array(skipped))
    print(f"wrote {len(paths)} {model} embeddings ({len(todo)} new) to {out}; skipped {len(skipped)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
