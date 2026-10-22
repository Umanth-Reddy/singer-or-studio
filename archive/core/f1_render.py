"""F1 step 1: render VocalSet excerpts under six production conditions.

  .venv/bin/python -I pilots/f1_render.py
Protocol: docs/04_pilot_log.md (F1). Writes data/f1/crops/<cond>/<singer>_<song>.wav
and data/f1/manifest.json (one record per singer x song, crops keyed by condition).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from encodec import EncodecModel
from scipy.signal import fftconvolve, lfilter

ROOT = Path(__file__).resolve().parent.parent
VOCALSET = ROOT.parent / "datasets" / "PS_Acoustics" / "VocalSet" / "FULL"
ACCOMP_SONG = ROOT.parent / "datasets" / "PS_Acoustics" / "MUSDB18" / "test" / "Al James - Schoolboy Facination.stem.mp4"
OUT = ROOT / "data" / "f1"
SR = 44100
WIN = 4 * SR
SONGS = ("caro", "dona", "row")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def shelf(x: np.ndarray, f0: float, gain_db: float, high: bool) -> np.ndarray:
    """RBJ cookbook shelving biquad, slope 1."""
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / SR
    alpha = np.sin(w) / 2 * np.sqrt(2)
    c, k = np.cos(w), 2 * np.sqrt(a) * alpha
    s = -1 if high else 1
    b = [a * ((a + 1) - s * (a - 1) * c + k), s * 2 * a * ((a - 1) - s * (a + 1) * c), a * ((a + 1) - s * (a - 1) * c - k)]
    d = [(a + 1) + s * (a - 1) * c + k, -s * 2 * ((a - 1) + s * (a + 1) * c), (a + 1) + s * (a - 1) * c - k]
    return lfilter(np.array(b) / d[0], np.array(d) / d[0], x)


def eq(x: np.ndarray, gain_db: float = 6.0) -> np.ndarray:
    return shelf(shelf(x, 200, gain_db, high=False), 4000, -gain_db, high=True)


def reverb(x: np.ndarray, wet: float = 0.3, rt60: float = 1.0, seed: int = 0) -> np.ndarray:
    t = np.arange(int(1.2 * rt60 * SR)) / SR
    rir = np.random.default_rng(seed).standard_normal(len(t)) * np.exp(-6.91 * t / rt60)
    tail = fftconvolve(x, rir / np.sqrt(np.sum(rir ** 2)))[: len(x)]
    return (1 - wet) * x + wet * tail * (np.std(x) / (np.std(tail) + 1e-9))


def comp(x: np.ndarray) -> np.ndarray:
    level = np.abs(x)
    env = np.empty_like(x)
    att, rel = np.exp(-1 / (0.005 * SR)), np.exp(-1 / (0.08 * SR))
    e = 0.0
    for i, v in enumerate(level):
        e = att * e + (1 - att) * v if v > e else rel * e + (1 - rel) * v
        env[i] = e
    db = 20 * np.log10(env + 1e-9)
    gain_db = np.where(db > -30, (db + 30) * (1 / 4 - 1), 0.0)
    y = x * 10 ** (gain_db / 20)
    return y * (np.std(x) / (np.std(y) + 1e-9))


def make_codec():
    model = EncodecModel.encodec_model_24khz().to(DEVICE).eval()

    def codec(x: np.ndarray, kbps: float = 6.0) -> np.ndarray:
        model.set_target_bandwidth(kbps)
        t = torchaudio.functional.resample(torch.from_numpy(x.astype(np.float32))[None], SR, model.sample_rate)
        with torch.no_grad():
            y = model.decode(model.encode(t[None].to(DEVICE)))[0, 0, : t.shape[-1]].cpu()
        return torchaudio.functional.resample(y, model.sample_rate, SR).numpy()[: len(x)]

    return codec


def load_accomp(song: Path = ACCOMP_SONG) -> np.ndarray:
    """60 s of a MUSDB song's drums + bass + other, mono."""
    cmd = ["ffmpeg", "-v", "error", "-ss", "30", "-t", "60", "-i", str(song), "-filter_complex",
           "[0:a:1][0:a:2][0:a:3]amix=inputs=3:normalize=0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.float32)


def make_mixsep():
    model = get_model("htdemucs").to(DEVICE).eval()
    default = load_accomp()

    def mixsep(x: np.ndarray, var_db: float = 0.0, accomp: np.ndarray | None = None) -> np.ndarray:
        a = np.resize(default if accomp is None else accomp, len(x))
        mix = x + a * (np.std(x) / (np.std(a) + 1e-9)) * 10 ** (-var_db / 20)  # vocal-to-accompaniment ratio
        stereo = np.stack([mix, mix])
        ref = mix
        with torch.no_grad():
            wav = torch.from_numpy(((stereo - ref.mean()) / (ref.std() + 1e-8)).astype(np.float32))[None].to(DEVICE)
            est = apply_model(model, wav, device=DEVICE, split=True, overlap=0.25)[0, model.sources.index("vocals")]
        return (est.cpu().numpy() * (ref.std() + 1e-8) + ref.mean()).mean(0)

    return mixsep


def main() -> None:
    conditions = {"clean": lambda x: x, "eq": eq, "reverb": reverb, "comp": comp,
                  "codec": make_codec(), "mixsep": make_mixsep()}
    records = []
    for singer_dir in sorted(VOCALSET.iterdir()):
        if not singer_dir.is_dir():
            continue
        for song in SONGS:
            src = next((singer_dir / "excerpts" / "straight").glob(f"*_{song}_straight.wav"), None)
            if src is None:
                print(f"missing {singer_dir.name} {song}")
                continue
            x, sr = sf.read(src, dtype="float32", always_2d=True)
            x = x.mean(1)
            if sr != SR:
                x = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, SR)[0].numpy()
            hop = SR // 2
            starts = range(0, len(x) - WIN + 1, hop)
            start = max(starts, key=lambda s: float(np.mean(x[s:s + WIN] ** 2)))
            record = {"singer": singer_dir.name, "sex": singer_dir.name.rstrip("0123456789"),
                      "song": song, "source": str(src), "start": start, "crops": {}}
            for name, fn in conditions.items():
                y = np.asarray(fn(x), dtype=np.float32)[start:start + WIN]
                y = y / (np.max(np.abs(y)) + 1e-9) * 0.5  # same peak level everywhere
                path = OUT / "crops" / name / f"{singer_dir.name}_{song}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, y, SR, subtype="FLOAT")
                record["crops"][name] = [str(path)]
            records.append(record)
            print(f"{singer_dir.name} {song} done", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(records, indent=1))
    print(f"done: {len(records)} clips x {len(conditions)} conditions")


if __name__ == "__main__":
    main()
