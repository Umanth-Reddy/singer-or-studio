"""Score the Oct-26 checks. Trial scoring is re-implemented here (not imported) as an independent check.

  .venv/bin/python -I claude-check-oct-26/analyze.py <rerun|loud|accomp|realsong|speaker>
Trial (as F2a): clean query (singer s, song i); positive = s, song j != i, processed; negative = other
same-sex singer, song j, clean. Error = sim(q, neg) >= sim(q, pos). Singer-macro, 2000x singer bootstrap.
Writes claude-check-oct-26/results_<check>.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
MODELS = ("cvsm_art", "cvsm_a", "ssl", "ecapa", "mert")
S = sorted({v for r in json.loads((ROOT / "pilots" / "g1_verdict.json").read_text())
            if r["model"] in ("cvsm_art", "ssl") for v in r["failing_vocalset"]})


def vectors(folder: Path, model: str) -> dict[str, np.ndarray] | None:
    f = folder / f"emb_{model}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return dict(zip(d["paths"].tolist(), d["vectors"] / np.linalg.norm(d["vectors"], axis=1, keepdims=True)))


class Trials:
    def __init__(self, manifest: list[dict], vec: dict[str, np.ndarray]):
        self.sex = {r["singer"]: r["sex"] for r in manifest}
        self.emb = {(r["singer"], r["song"], v): vec[p[0]] for r in manifest for v, p in r["crops"].items() if p[0] in vec}
        self.songs = sorted({r["song"] for r in manifest})

    def per_singer(self, pos_v: str, q_v: str = "clean", neg_v: str = "clean") -> dict[str, float]:
        per = defaultdict(list)
        for s in self.sex:
            for i in self.songs:
                q = self.emb.get((s, i, q_v))
                for j in self.songs:
                    p = self.emb.get((s, j, pos_v))
                    if j == i or q is None or p is None:
                        continue
                    sp = q @ p
                    for s2, x in self.sex.items():
                        n = self.emb.get((s2, j, neg_v))
                        if s2 != s and x == self.sex[s] and n is not None:
                            per[s].append(q @ n >= sp)
        return {s: float(np.mean(v)) for s, v in per.items()}

    def mean_over(self, variants: list[str], **kw) -> dict[str, float]:
        tabs = [self.per_singer(v, **kw) for v in variants]
        return {s: float(np.mean([t[s] for t in tabs if s in t])) for s in self.sex if any(s in t for t in tabs)}


def ci(d: np.ndarray, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return [round(float(np.percentile(b, 2.5)) * 100, 1), round(float(np.percentile(b, 97.5)) * 100, 1)]


def effect(err: dict, base: dict) -> dict:
    ks = sorted(set(err) & set(base))
    d = np.array([err[k] - base[k] for k in ks])
    return {"err_pct": round(float(np.mean([err[k] for k in ks])) * 100, 1),
            "effect_pts": round(float(d.mean()) * 100, 1), "ci95": ci(d),
            "passes": bool(d.mean() >= 0.10 and ci(d)[0] > 0)}


def load_set(folder: Path, model: str, extra: Path | None = None) -> Trials | None:
    vec = vectors(folder, model)
    if vec is None:
        return None
    if extra is not None:
        vec |= vectors(extra, model) or {}
    return Trials(json.loads((folder / "manifest.json").read_text()), vec)


def rerun() -> dict:
    """Check 1: re-rendered crops vs stored F2a results."""
    stored = {r["model"]: r for r in json.loads((ROOT / "pilots" / "f2_results.json").read_text())}
    out = {"waveform_diff": json.loads((HERE / "data" / "rerun" / "waveform_diff.json").read_text())}
    for m in MODELS:
        t = load_set(HERE / "data" / "rerun", m)
        if t is None:
            continue
        base = t.per_singer("clean")
        row = {"clean": {"new": round(np.mean(list(base.values())) * 100, 1), "stored": round(stored[m]["baseline_error_pct"], 1)}}
        for v in ("reverb@0.3", "codec@6", "mp3@64", "mixsep@0", "eq@12"):
            row[v] = {"new": effect(t.per_singer(v), base)["err_pct"], "stored": round(stored[m]["levels"][v]["error_pct"], 1)}
        out[m] = row
    return out


def loud() -> dict:
    """Check 2: does level matter? RMS-matched variants and pure gain changes of clean."""
    out = {}
    for m in MODELS + ("xvect",):
        t = load_set(HERE / "data" / "loud", m, extra=DATA / "f2")
        orig = load_set(DATA / "f2", m)
        if t is None or orig is None:
            continue
        b_rms = t.per_singer("clean_rms", q_v="clean_rms", neg_v="clean_rms")
        b = orig.per_singer("clean")
        row = {}
        for v in ("reverb@0.3", "codec@6", "mp3@64", "mixsep@0", "eq@12"):
            row[v] = {"peak_norm_effect": effect(orig.per_singer(v), b)["effect_pts"],
                      "rms_matched_effect": effect(t.per_singer(v + "_rms", q_v="clean_rms", neg_v="clean_rms"), b_rms)["effect_pts"]}
        for g in (-12, -6, 6):
            row[f"gain@{g}dB_only"] = effect(t.per_singer(f"gain@{g}"), t.per_singer("clean"))
        out[m] = row
    return out


def accomp() -> dict:
    """Check 3a: mix-and-separate with 10 accompaniments never used before."""
    out = {"choice": json.loads((HERE / "data" / "accomp_choice.json").read_text())}
    for name, orig_name in (("accomp_f2", "f2"), ("accomp_g1", "g1")):
        stored = {r["model"]: r for r in json.loads((ROOT / "pilots" / f"{orig_name}_results.json").read_text())}
        out[name] = {}
        for m in MODELS:
            t = load_set(HERE / "data" / name, m)
            if t is None:
                continue
            base = t.per_singer("clean")
            vs = sorted(v for v in next(iter(json.loads((HERE / "data" / name / "manifest.json").read_text())))["crops"] if v != "clean")
            per = {v.split("#")[1]: effect(t.per_singer(v), base) for v in vs}
            out[name][m] = {"all_10": effect(t.mean_over(vs), base),
                            "both_processed_err_pct": round(float(np.mean(
                                [np.mean(list(t.per_singer(v, neg_v=v).values())) for v in vs])) * 100, 1),
                            "per_accomp_effects": [p["effect_pts"] for p in per.values()],
                            "n_passing": f"{sum(p['passes'] for p in per.values())}/{len(per)}",
                            "original_al_james_effect": round(stored[m]["levels"]["mixsep@0"]["effect_pts"], 1)}
    return out


def realsong() -> dict:
    """Check 3b: real MUSDB songs. Query = ground-truth vocal stem, window a. Positive = same song, other
    window b, either ground-truth stem (baseline) or htdemucs-separated from the real mix. Negatives =
    ground-truth stem, window b, of every song by another artist. Song-macro, song bootstrap."""
    manifest = json.loads((HERE / "data" / "realsong" / "manifest.json").read_text())
    out = {}
    for m in MODELS:
        vec = vectors(HERE / "data" / "realsong", m)
        if vec is None:
            continue
        songs = [(r["artist"], [vec.get(p) for p in r["crops"]["gt"]], [vec.get(p) for p in r["crops"]["sep"]]) for r in manifest]
        res = {"gt": [], "sep": [], "sep_neg_sep": []}
        for x, (artist, gt, sep) in enumerate(songs):
            errs = {k: [] for k in res}
            for a in range(3):
                for b in range(3):
                    if a == b or gt[a] is None or gt[b] is None or sep[b] is None:
                        continue
                    q = gt[a]
                    ng = [o[1][b] for y, o in enumerate(songs) if y != x and o[0] != artist and o[1][b] is not None]
                    ns = [o[2][b] for y, o in enumerate(songs) if y != x and o[0] != artist and o[2][b] is not None]
                    errs["gt"].append(np.mean([q @ n >= q @ gt[b] for n in ng]))
                    errs["sep"].append(np.mean([q @ n >= q @ sep[b] for n in ng]))
                    errs["sep_neg_sep"].append(np.mean([q @ n >= q @ sep[b] for n in ns]))
            if errs["gt"]:
                for k in res:
                    res[k].append(np.mean(errs[k]))
        g, s, ss = (np.array(res[k]) for k in ("gt", "sep", "sep_neg_sep"))
        out[m] = {"n_songs": len(g), "gt_err_pct": round(g.mean() * 100, 1), "sep_err_pct": round(s.mean() * 100, 1),
                  "effect_pts": round((s - g).mean() * 100, 1), "ci95": ci(s - g),
                  "both_separated_err_pct": round(ss.mean() * 100, 1)}
    return out


def speaker() -> dict:
    """Check 4: is ECAPA's robustness real? Paired singer-bootstrap of effect differences over S, plus x-vector."""
    out = {}
    for ds in ("f2", "g1"):
        eff, base_err = {}, {}
        for m in MODELS + ("xvect",):
            folder = DATA / ds if m != "xvect" else HERE / "data" / f"xv_{ds}"
            t = load_set(folder, m)
            if t is None:
                continue
            base = t.per_singer("clean")
            err = t.mean_over(S)
            eff[m] = {s: err[s] - base[s] for s in base}
            base_err[m] = base
        res = {}
        for m in eff:
            d = np.array(list(eff[m].values()))
            res[m] = {"clean_err_pct": round(np.mean(list(base_err[m].values())) * 100, 1),
                      "effect_over_S": round(d.mean() * 100, 1), "ci95": ci(d)}
        for m in ("cvsm_art", "cvsm_a", "ssl", "mert", "xvect"):
            if m in eff:
                ks = sorted(set(eff[m]) & set(eff["ecapa"]))
                d = np.array([eff[m][k] - eff["ecapa"][k] for k in ks])
                c = np.array([base_err[m][k] - base_err["ecapa"][k] for k in ks])
                res[f"{m}_minus_ecapa"] = {"effect_diff_pts": round(d.mean() * 100, 1), "ci95": ci(d),
                                           "clean_err_diff_pts": round(c.mean() * 100, 1), "clean_ci95": ci(c)}
        out[ds] = res
    return out


if __name__ == "__main__":
    check = sys.argv[1]
    res = {"rerun": rerun, "loud": loud, "accomp": accomp, "realsong": realsong, "speaker": speaker}[check]()
    (HERE / f"results_{check}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
