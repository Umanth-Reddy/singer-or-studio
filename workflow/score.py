"""Score the rerun with the recorded analysis, leaving the released evidence intact."""
import argparse
from pathlib import Path

from common import MODELS_BY_TASK, ROOT, TASKS, archived, read, write


def score_jvs(out, models):
    # The initial script saves the common-song orientation including EER.
    # Give its archival path layout a local home within this run.
    import shutil
    work = out / "initial"
    replay = work / "experiments/jvs"
    replay.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out / "manifest.json", replay / "manifest.json")
    for model in models:
        name = f"embeddings_{model}.npz"
        link = replay / name
        if not link.exists():
            link.symlink_to(out / name)
    initial = archived("jvs-initial", "analyze_jvs")
    initial.ROOT = work
    initial.MODELS = models
    initial.main()
    shutil.copyfile(replay / "results.json", out / "results.json")
    analysis = archived("factorial", "analyze")
    analysis.JVS_MODELS = models
    write(out / "factorial-results.json", analysis.jvs_analysis(out))


def score_core(task, out, models):
    core = archived("core", "f2_analyze")
    core.OUT = out
    manifest = read(out / "manifest.json")
    results = [core.analyze(model, manifest) for model in models]
    write(out / "scored-results.json", results)


def score_t1(task, out, models, runs):
    import numpy as np

    analysis = archived("measured-response", "analyze")
    ds = task.removeprefix("t1-")
    before = runs / ds
    results = {}
    for model in models:
        new = analysis.Bench(out, {model: out}, (model,))
        old = analysis.Bench(before, {model: before}, (model,))
        metrics = {}
        for family in ("reverb", "mixsep"):
            conditions = [v for v in new.variants if v.startswith(family + "@")]
            effect = new.effect(conditions)
            effect["both_processed_err_pct"] = float(np.nanmean(new.error(conditions, both=True)) * 100)
            effect["original_effect_pts"] = old.effect([
                v for v in analysis.S if v.startswith(family + "@")
            ])["effect_pts"]
            effect["per_unit"] = {
                unit: new.effect([v for v in conditions if v.endswith(f"#{unit}")])
                for unit in sorted({v.split("#")[1] for v in conditions})
            }
            metrics[family] = effect
        results[model] = metrics
    write(out / "scored-results.json", results)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("task", choices=TASKS)
    cli.add_argument("--runs", type=Path, default=ROOT / "runs")
    cli.add_argument("--model", action="append", help="Limit scoring to named model(s).")
    args = cli.parse_args()
    task = args.task
    models = tuple(args.model or MODELS_BY_TASK[task])
    if not set(models) <= set(MODELS_BY_TASK[task]):
        cli.error(f"Unsupported model for {task}")
    runs = args.runs.resolve()
    out = runs / task
    names = {m: f"embeddings_{m}.npz" if task == "jvs" else f"{m}.npz" if task == "room"
             else f"emb_{m}.npz" for m in models}
    missing = [str(out / n) for n in names.values() if not (out / n).is_file()]
    if missing:
        cli.error(f"Missing embeddings: {', '.join(missing)}")
    if task == "jvs":
        score_jvs(out, models)
    elif task == "room":
        analysis = archived("factorial", "analyze")
        archived("room-followup", "confirm").score(out, models, analysis)
    elif task == "musdb":
        analysis = archived("independent", "analyze")
        # The archive uses HERE/data/realsong; accept a direct input folder instead.
        write(out / "scored-results.json", analysis.realsong(out, models))
    elif task.startswith("t1-"):
        score_t1(task, out, models, runs)
    else:
        score_core(task, out, models)
    print(f"Scoring complete for {task} in {out}")


if __name__ == "__main__":
    main()
