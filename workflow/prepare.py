"""Render precisely the archived crop selections into an isolated run folder."""
import argparse
import itertools
from pathlib import Path
import sys
import zipfile

from common import ARCHIVE, EVIDENCE, ROOT, TASKS, archived, digest, read, write


def prepare_jvs(data, out):
    import numpy as np
    import soundfile as sf

    recipe = archived("jvs-initial", "prepare_eval")
    archive = data / "jvs/jvs_music_ver1.zip"
    inventory = read(ARCHIVE / "jvs-initial/jvs_music_inventory.json")
    if archive.stat().st_size != inventory["archive_bytes"]:
        raise ValueError("JVS-MuSiC archive size differs from the recorded release")
    recipe.DATA = archive.parent
    groups = recipe.voice_groups()
    originals = read(ARCHIVE / "jvs-initial/manifest.json")
    if groups != {row["singer"]: row["group"] for row in originals}:
        raise ValueError("JVS official grouping differs from the archived manifest")
    items = []
    with zipfile.ZipFile(archive) as source:
        for row in inventory["records"]:
            member = "jvs_music_ver1/" + row["path"].split("/raw/", 1)[1]
            with source.open(member) as stream:
                x, rate = sf.read(stream, dtype="float32", always_2d=True)
            if rate != recipe.SR or x.shape != (row["frames"], 1):
                raise ValueError(f"Unexpected layout in {member}")
            singer, song = row["singer"], row["song"]
            seed = 1000 + int(singer[3:]) * 2 + (song == "unique")
            waveform = x[:, 0]
            crops = {}
            for variant, y in (
                ("clean", waveform), ("mp3_64", recipe.mp3_64(waveform)),
                ("reverb_20", recipe.reverb_20(waveform, seed)),
            ):
                rel = Path("audio") / variant / singer / f"{song}.wav"
                path = out / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, recipe.normalize(recipe.crop_center(y)), rate, subtype="FLOAT")
                crops[variant] = rel.as_posix()
            items.append({"singer": singer, "song": song, "group": groups[singer], "crops": crops})
    write(out / "manifest.json", items)


def prepare_room(data, out):
    import numpy as np
    import soundfile as sf
    from scipy.signal import fftconvolve

    recipe = archived("room-followup", "confirm")
    rows = read(EVIDENCE / "response-crop-manifest.json")
    pairs = read(ARCHIVE / "room-followup/room_pairs.json")
    filters = []
    for record in itertools.chain.from_iterable(pairs):
        path = data / "mit-ir" / record["file"]
        if digest(path) != record["sha256"]:
            raise ValueError(f"Response checksum mismatch: {path}")
        audio = recipe.read_mono(path)
        aligned = audio[np.argmax(np.abs(audio)):]
        filters.append(aligned / np.linalg.norm(aligned))
    (out / "audio").mkdir(parents=True, exist_ok=True)
    for row in rows:
        relative = (Path("covox") / row["source"] if row["dataset"] == "covox"
                    else Path("nus") / row["singer"] / "sing" / row["source"])
        source = data / relative
        if digest(source) != row["sha256"]:
            raise ValueError(f"Recording checksum mismatch: {source}")
        x = recipe.read_mono(source)
        row["source"] = relative.as_posix()
        sf.write(out / row["crops"]["clean"], recipe.crop4(x, row["start"]), recipe.RATE, subtype="FLOAT")
        for i, rir in enumerate(filters):
            tail = fftconvolve(x, rir)[:len(x)]
            tail *= np.std(x) / (np.std(tail) + 1e-9)
            for wet in recipe.WET:
                y = recipe.crop4((1 - wet) * x + wet * tail, row["start"])
                sf.write(out / row["crops"][f"{wet}_r{i}"], y, recipe.RATE, subtype="FLOAT")
    write(out / "room_pairs.json", pairs)
    write(out / "manifest.json", rows)


def prepare_core(task, data, out):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    # These scripts import the original renderer under its historical module name.
    original = ARCHIVE / "core"
    sys.path.insert(0, str(original))
    import f1_render as core
    from f2_render import mp3

    ds = task.removeprefix("t1-")
    corpus = "vocalset" if ds == "f2" else "gtsinger"
    sources = read(EVIDENCE / f"{ds}-rerun-inputs.json")
    accomp = data / "musdb18/test/Al James - Schoolboy Facination.stem.mp4"
    mixsep = core.make_mixsep(accomp)
    if task.startswith("t1-"):
        measured = archived("measured-response", "render")
        choice = read(ARCHIVE / "measured-response/selection.json")
        rirs = [measured.load_rir(data / "mit-ir" / r["file"]) for r in choice["rooms"]]
        backs = [core.load_accomp(data / "musdb18/test" / name) for name in choice["accompaniments"]]

        def transforms(x):
            yield "clean", x
            for i, rir in enumerate(rirs):
                for wet in measured.WETS:
                    yield f"reverb@{wet:g}#r{i}", measured.reverb(x, rir, wet)
            for i, backing in enumerate(backs):
                for ratio in measured.VARS:
                    yield f"mixsep@{ratio:g}#a{i}", mixsep(x, var_db=ratio, accomp=backing)
    else:
        codec = core.make_codec()
        sweep = {
            "reverb": (core.reverb, (0.1, 0.2, 0.3, 0.5)),
            "codec": (codec, (24, 12, 6, 3, 1.5)),
            "mp3": (mp3, (128, 64, 32)),
            "eq": (core.eq, (3, 6, 12)),
            "mixsep": (mixsep, (12, 6, 0, -6)),
        }

        def transforms(x):
            yield "clean", x
            for family, (fn, levels) in sweep.items():
                for level in levels:
                    yield f"{family}@{level:g}", fn(x, level)

    manifest = []
    for row in sources:
        source = data / corpus / row["source"]
        if digest(source) != row["sha256"]:
            raise ValueError(f"Recording checksum mismatch: {source}")
        x, rate = sf.read(source, dtype="float32", always_2d=True)
        x = x.mean(1)
        if rate != core.SR:
            x = torchaudio.functional.resample(torch.from_numpy(x)[None], rate, core.SR)[0].numpy()
        start = row["start"]
        crops = {}
        for name, y in transforms(x):
            crop = np.asarray(y, dtype=np.float32)[start:start + core.WIN]
            if len(crop) != core.WIN:
                raise ValueError(f"Unexpected cropped length: {source} / {name}")
            crop = crop / (np.max(np.abs(crop)) + 1e-9) * 0.5
            path = out / "audio" / name / f"{row['singer']}_{row['song']}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, crop, core.SR, subtype="FLOAT")
            crops[name] = [str(path)]
        manifest.append({**row, "source": str(source), "crops": crops})
        print(f"{task}: {row['singer']} {row['song']}", flush=True)
    write(out / "manifest.json", manifest)


def prepare_musdb(data, out):
    import soundfile as sf
    import torch

    recipe = archived("independent", "p1_separate")
    rows = read(EVIDENCE / "musdb18-crop-manifest.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = recipe.get_model("htdemucs").to(device).eval()
    vocal_index = model.sources.index("vocals")
    manifest = []
    for i, row in enumerate(rows):
        source = data / "musdb18" / row["corpus_split"] / f"{row['track']}.stem.mp4"
        start = row["excerpt_start_seconds"]
        mix = recipe.decode(source, start)
        reference = mix.mean(0)
        with torch.no_grad():
            wave = torch.from_numpy((mix - reference.mean()) / (reference.std() + 1e-8))[None].to(device)
            output = recipe.apply_model(model, wave, device=device, split=True, overlap=0.25)[0, vocal_index]
        estimate = (output.cpu().numpy() * (reference.std() + 1e-8) + reference.mean()).mean(0)
        target = recipe.decode(source, start, stream=4).mean(0)
        crops = {}
        for kind, samples in (("gt", target), ("sep", estimate)):
            paths = []
            for j, offset in enumerate(row["crop_starts_samples_44100"]):
                path = out / "audio" / kind / f"{i:03}_{j}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                crop = samples[offset:offset + recipe.SR * recipe.WIN_S]
                if len(crop) != recipe.SR * recipe.WIN_S:
                    raise ValueError(f"Unexpected cropped length: {source} / {kind}")
                sf.write(path, crop, recipe.SR, subtype="FLOAT")
                paths.append(str(path))
            crops[kind] = paths
        manifest.append({"artist": row["artist_label"], "song": row["track"], "crops": crops})
        print(f"MUSDB18: {i + 1}/{len(rows)}", flush=True)
    write(out / "manifest.json", manifest)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("task", choices=TASKS)
    cli.add_argument("--datasets", type=Path, default=ROOT / "datasets")
    cli.add_argument("--runs", type=Path, default=ROOT / "runs")
    args = cli.parse_args()
    out = args.runs.resolve() / args.task
    if (out / "manifest.json").exists():
        cli.error(f"{out} already has a completed render. Choose another --runs directory.")
    data = args.datasets.resolve()
    if args.task in ("jvs", "room", "musdb"):
        handlers = {"jvs": prepare_jvs, "room": prepare_room, "musdb": prepare_musdb}
        handlers[args.task](data, out)
    else:
        prepare_core(args.task, data, out)
    print(f"Wrote {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
