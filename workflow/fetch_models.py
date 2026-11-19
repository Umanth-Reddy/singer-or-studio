"""Fetch the frozen public encoders into the ignored models directory."""
import argparse
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from common import CORE_MODELS, ROOT
from embed import model_files

CVSM_REVISION = "ffbb3ae0799b42d83e202db7f9ac04b8c6bdbdcf"
HUB_MODELS = {
    "ssl": ("BernardoTorres/singer-identity", "e60d96b",
            "singer_identity_ssl/checkpoints/ssl_contrastive_vc",
            ["contrastive-vc/model.ts"]),
    "ecapa": ("speechbrain/spkrec-ecapa-voxceleb", "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
              "ecapa/weights", ["*.yaml", "*.ckpt", "label_encoder.txt"]),
    "mert": ("m-a-p/MERT-v1-95M", "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5",
             "mert/weights", ["*.json", "*.py", "pytorch_model.bin"]),
}


def fetch_cvsm(models):
    repo = models / "cvsm/repo"
    if repo.exists():
        return
    repo.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="cvsm-checkout-", dir=repo.parent) as staging:
        checkout = Path(staging) / "repo"
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        "https://github.com/cgaroufis/CVSM.git", str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "sparse-checkout", "set", "cola", "mscol",
                        "models/htdemucs/m4all_cvsmart", "models/open-unmix/m4all_cvsma"], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", CVSM_REVISION], check=True)
        checkout.rename(repo)


def fetch_hub(models, model):
    from huggingface_hub import snapshot_download

    upstream, revision, dest, patterns = HUB_MODELS[model]
    snapshot = Path(snapshot_download(upstream, revision=revision, allow_patterns=patterns))
    target = models / dest
    target.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for source in snapshot.glob(pattern):
            if source.is_file() and not (target / source.name).exists():
                shutil.copy2(source, target / source.name)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model", nargs="+", choices=CORE_MODELS)
    cli.add_argument("--models", type=Path, default=ROOT / "models")
    args = cli.parse_args()
    models = args.models.resolve()
    for model in dict.fromkeys(args.model):
        if model.startswith("cvsm"):
            fetch_cvsm(models)
        else:
            fetch_hub(models, model)
        model_files(models, model)
        print(f"Verified {model} checkpoint files")


if __name__ == "__main__":
    main()
