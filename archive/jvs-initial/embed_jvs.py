"""Create JVS-MuSiC embeddings with the study's frozen CVSM or SSL encoders.

Run CVSM variants in ps/.venv-gate1 and SSL in ps/.venv-ssl. This script reads
the existing model repositories/checkpoints without copying or modifying them.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, "/legacy/ps/glass-breaker/q2-study/external")
import gate1_experiment as g  # noqa: E402

MODELS = {
    "cvsm_art": ("/legacy/ps/glass-breaker/models/original-cvsm/htdemucs/m4all_cvsmart", "tf"),
    "cvsm_a": ("/legacy/ps/glass-breaker/models/original-cvsm/open-unmix/m4all_cvsma", "tf"),
    "ssl": ("/legacy/ps/glass-breaker/models/original-ssl/ssl_contrastive_vc/model.ts", "torch"),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(model_name: str) -> None:
    model_path, framework = MODELS[model_name]
    manifest = json.loads((ROOT / "experiments/jvs/manifest.json").read_text())
    rows = []
    keys = []
    for r in manifest:
        for condition, rel in r["crops"].items():
            keys.append((r["singer"], r["song"], condition, r["group"]))
            rows.append({"audio_path": str(ROOT / rel)})
    g.MANIFEST = ROOT / "audit/jvs_music_inventory.json"
    vectors = []
    if framework == "tf":
        g.CVSM_REPO = Path("/legacy/ps/existing-work/CVSM")
        g.CVSM_CHECKPOINT_DIR = Path(model_path)
        vectors = g.run_cvsm(rows)
    else:
        g.SSL_CHECKPOINT = Path(model_path)
        vectors = g.run_ssl(rows)
    output = ROOT / "experiments/jvs" / f"embeddings_{model_name}.npz"
    np.savez_compressed(output, vectors=np.asarray(vectors, dtype=np.float32), keys=np.asarray(keys))
    checkpoint_files = [Path(model_path)] if Path(model_path).is_file() else sorted(Path(model_path).glob("checkpoint-*.index")) + sorted(Path(model_path).glob("checkpoint-*.data-*"))
    provenance = {
        "model": model_name,
        "checkpoint_files": [{"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size} for p in checkpoint_files],
        "embedding_sha256": sha(output),
        "rows": len(rows),
        "preprocessing": "prepared centered 4 s mono JVS crops; CVSM uses its original 16 kHz / four normalized 1 s window path; SSL uses its original peak-normalized 44.1 kHz path",
    }
    (ROOT / "experiments/jvs" / f"provenance_{model_name}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"saved {len(vectors)} embeddings ({np.asarray(vectors).shape[1]} dimensions) to {output}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in MODELS:
        raise SystemExit(f"usage: {sys.argv[0]} {'|'.join(MODELS)}")
    main(sys.argv[1])
