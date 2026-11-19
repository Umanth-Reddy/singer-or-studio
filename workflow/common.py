"""Shared paths and readers for the portable research commands."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
ARCHIVE = ROOT / "archive"
TASKS = ("f2", "g1", "t1-f2", "t1-g1", "jvs", "room", "musdb")
CORE_MODELS = ("cvsm_art", "cvsm_a", "ssl", "ecapa", "mert")
MODELS_BY_TASK = {task: CORE_MODELS for task in TASKS} | {
    "jvs": CORE_MODELS[:3], "room": ("cvsm_art", "ssl"),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def archived(folder, script, name=None):
    path = ARCHIVE / folder / f"{script}.py"
    spec = importlib.util.spec_from_file_location(name or f"archive_{folder}_{script}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
