"""Apply the archived frozen-encoder preprocessing to a prepared run."""
import argparse
from pathlib import Path

from common import MODELS_BY_TASK, ROOT, TASKS, archived, digest, read, write


def model_files(models, model):
    identities = {
        "cvsm_art": ("cvsm/repo/models/htdemucs/m4all_cvsmart/",),
        "cvsm_a": ("cvsm/repo/models/open-unmix/m4all_cvsma/",),
        "ssl": ("singer_identity_ssl/checkpoints/ssl_contrastive_vc/model.ts",),
        "ecapa": ("ecapa/weights/",),
        "mert": ("mert/weights/",),
    }
    checked = {}
    for line in (ROOT / "docs/checkpoints.sha256").read_text().splitlines():
        checksum, historic = line.split("  ", 1)
        rel = historic.removeprefix("models/")
        if not rel.startswith(identities[model]):
            continue
        path = models / rel
        if model == "ecapa" and path.name == "label_encoder.ckpt" and not path.exists():
            path = path.with_name("label_encoder.txt")
        if digest(path) != checksum:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
        checked[path.relative_to(models).as_posix()] = checksum
    if not checked:
        raise ValueError(f"No checksum specification for {model}")
    return checked


def infer(model, files, models):
    import numpy as np
    import soundfile as sf

    if model in ("cvsm_art", "cvsm_a", "ssl"):
        gate = archived("core", "gate1_experiment")
        gate.CVSM_REPO = models / "cvsm/repo"
        gate.CVSM_CHECKPOINT_DIR = gate.CVSM_REPO / {
            "cvsm_art": "models/htdemucs/m4all_cvsmart",
            "cvsm_a": "models/open-unmix/m4all_cvsma",
        }.get(model, "")
        if model in ("cvsm_art", "cvsm_a"):
            checkpoint = "checkpoint-40" if model == "cvsm_art" else "checkpoint-26"
            gate.CVSM_CHECKPOINT_PREFIX = str(gate.CVSM_CHECKPOINT_DIR / checkpoint)
        gate.SSL_CHECKPOINT = models / "singer_identity_ssl/checkpoints/ssl_contrastive_vc/model.ts"
        run = gate.run_ssl if model == "ssl" else gate.run_cvsm
        return run([{"audio_path": str(p)} for p in files])

    import torch
    import torchaudio

    helper = archived("core", "embed_extra")
    if model == "ecapa":
        from speechbrain.inference.speaker import EncoderClassifier
        obj = EncoderClassifier.from_hparams(
            source=str(models / "ecapa/weights"),
            savedir=str(models / "ecapa/weights"), run_opts={"device": helper.DEVICE})
        rate = 16000

        def encoder(wave):
            return obj.encode_batch(wave[None])[0, 0]
    else:
        rate, encoder = helper.mert(str(models / "mert/weights"), local_files_only=True)

    vectors = []
    with torch.no_grad():
        for path in files:
            x, original = sf.read(path, dtype="float32")
            wave = torchaudio.functional.resample(torch.from_numpy(x), original, rate)
            vectors.append(encoder(wave.to(helper.DEVICE)).float().cpu().numpy())
    return np.stack(vectors)


def main():
    import numpy as np
    import soundfile as sf

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("task", choices=TASKS)
    cli.add_argument("model", choices=sorted({m for v in MODELS_BY_TASK.values() for m in v}))
    cli.add_argument("--models", type=Path, default=ROOT / "models")
    cli.add_argument("--runs", type=Path, default=ROOT / "runs")
    args = cli.parse_args()
    task, model = args.task, args.model
    if model not in MODELS_BY_TASK[task]:
        cli.error(f"{model} was outside the {task} experiment")
    out = args.runs.resolve() / task
    models = args.models.resolve()
    checkpoints = model_files(models, model)
    manifest = out / "manifest.json"
    rows = read(manifest)
    keys, paths, skipped = [], [], []
    for rect in rows:
        for condition, value in rect["crops"].items():
            if task == "jvs":
                keys.append((rect["singer"], rect["song"], condition, rect["group"]))
                paths.append(out / value)
            elif task == "room":
                keys.append((rect["dataset"], rect["singer"], rect["song"], condition))
                paths.append(out / value)
            else:
                for crop in value:
                    given = Path(crop)
                    samples, _ = sf.read(given, dtype="float32")
                    if not np.abs(samples).reshape(4, -1).max(1).min() > 0:
                        skipped.append(str(given))
                    else:
                        paths.append(given)
    vectors = infer(model, paths, models)
    if task == "jvs":
        target = out / f"embeddings_{model}.npz"
    elif task == "room":
        target = out / f"{model}.npz"
    else:
        target = out / f"emb_{model}.npz"
    extra = {"keys": np.asarray(keys)} if keys else {
        "paths": np.asarray([str(p) for p in paths]), "skipped": np.asarray(skipped),
    }
    np.savez_compressed(target, vectors=vectors, **extra)
    suffix = f"{model}_provenance.json" if task == "room" else f"provenance_{model}.json"
    write(out / suffix, {"model": model, "count": len(vectors), "checkpoints": checkpoints,
                         "manifest_sha256": digest(manifest), "embedding_sha256": digest(target),
                         "embeddings_sha256": digest(target)})
    print(f"Wrote {target} with {len(vectors)} embeddings")


if __name__ == "__main__":
    main()
