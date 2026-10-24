"""T1 / T3 analysis: production-mismatch effect per family, judged by TEMP/PROTOCOL.md.

  ~/ps/.venv-ssl/bin/python -I TEMP/analyze.py t1   # real rooms + held-out accompaniments
  ~/ps/.venv-ssl/bin/python -I TEMP/analyze.py t1 f2   # interim look, VocalSet only (no verdict)
  ~/ps/.venv-ssl/bin/python -I TEMP/analyze.py t3   # WavLM-SV + CVSM-ART/ECAPA fusion
Trial identical to pilots/f2_analyze.py: clean query (singer s, song i); positive = s, song j,
processed; negative = other same-sex singer, song j, clean (or processed alike for the
both-processed control). Error = sim(q, neg) >= sim(q, pos). Effects are singer-macro with a
2000x singer bootstrap. Writes TEMP/<test>_results.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TEMP = Path(__file__).resolve().parent
ROOT = TEMP.parent
SPECIALISED = ("cvsm_art", "cvsm_a", "ssl")
MODELS = SPECIALISED + ("ecapa", "mert")
DATASETS = {"f2": "VocalSet", "g1": "GTSinger"}
# Fixed set S of E1 (docs/04): VocalSet F2a failing strengths of CVSM-ART and SSL.
S = sorted({v for r in json.loads((ROOT / "pilots" / "g1_verdict.json").read_text())
            if r["model"] in ("cvsm_art", "ssl") for v in r["failing_vocalset"]})


class Bench:
    """Embeddings of one dataset keyed (singer, song, variant), plus a pairwise similarity."""

    def __init__(self, manifest_dir: Path, emb_dirs: dict[str, Path], models: tuple[str, ...]):
        manifest = json.loads((manifest_dir / "manifest.json").read_text())
        self.sex = {r["singer"]: r["sex"] for r in manifest}
        self.singers = sorted(self.sex)
        self.songs = sorted({r["song"] for r in manifest})
        self.variants = sorted({v for r in manifest for v in r["crops"]})
        self.vecs = []
        for m in models:
            d = np.load(emb_dirs[m] / f"emb_{m}.npz")
            vec = dict(zip(d["paths"].tolist(), d["vectors"] / np.linalg.norm(d["vectors"], axis=1, keepdims=True)))
            self.vecs.append({(r["singer"], r["song"], n): vec[p[0]] for r in manifest
                              for n, p in r["crops"].items() if p[0] in vec})
        self.keys = set.intersection(*(set(v) for v in self.vecs))

    def sim(self, a: tuple, b: tuple) -> float:
        return float(np.mean([float(v[a] @ v[b]) for v in self.vecs]))  # fusion = mean cosine

    def outcomes(self, variants: list[str], both: bool = False) -> dict[str, list[bool]]:
        per = defaultdict(list)
        for variant in variants:
            for s in self.singers:
                for i in self.songs:
                    q = (s, i, "clean")
                    for j in self.songs:
                        pos = (s, j, variant)
                        if j == i or q not in self.keys or pos not in self.keys:
                            continue
                        sp = self.sim(q, pos)
                        for s2 in self.singers:
                            neg = (s2, j, variant if both else "clean")
                            if s2 != s and self.sex[s2] == self.sex[s] and neg in self.keys:
                                per[s].append(self.sim(q, neg) >= sp)
        return per

    def error(self, variants: list[str], both: bool = False) -> np.ndarray:
        per = self.outcomes(variants, both)
        return np.array([np.mean(per[s]) if per[s] else np.nan for s in self.singers])

    def effect(self, variants: list[str], seed: int = 0) -> dict:
        base, err = self.error(["clean"]), self.error(variants)
        diff = (err - base)[~np.isnan(err - base)]
        rng = np.random.default_rng(seed)
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        ci = [float(np.percentile(boots, 2.5) * 100), float(np.percentile(boots, 97.5) * 100)]
        return {"clean_err_pct": float(np.nanmean(base) * 100), "err_pct": float(np.nanmean(err) * 100),
                "effect_pts": float(diff.mean() * 100), "ci95_pts": ci,
                "passes": bool(diff.mean() * 100 >= 10 and ci[0] > 0)}


def original_dirs(ds: str) -> dict[str, Path]:
    dirs = {m: ROOT / "data" / ds for m in MODELS}
    return dirs | {"wavlm": TEMP / "data" / f"orig_{ds}"}


def t1(only: str | None = None) -> dict:
    out = {"S": S}
    for ds, name in DATASETS.items():
        if only and ds != only:
            continue
        new_dir = TEMP / "data" / ds
        out[ds] = {}
        for m in MODELS:
            new = Bench(new_dir, {m: new_dir}, (m,))
            old = Bench(ROOT / "data" / ds, original_dirs(ds), (m,))
            res = {}
            for fam, tag in (("reverb", "r"), ("mixsep", "a")):
                vs = [v for v in new.variants if v.startswith(fam + "@")]
                e = new.effect(vs)
                e["both_processed_err_pct"] = float(np.nanmean(new.error(vs, both=True)) * 100)
                orig = old.effect([v for v in S if v.startswith(fam + "@")])
                e["original_effect_pts"] = orig["effect_pts"]
                e["ratio_to_original"] = e["effect_pts"] / orig["effect_pts"] if orig["effect_pts"] else None
                units = sorted({v.split("#")[1] for v in vs})
                e["per_unit"] = {u: new.effect([v for v in vs if v.endswith("#" + u)]) for u in units}
                e["units_passing"] = f"{sum(x['passes'] for x in e['per_unit'].values())}/{len(units)}"
                res[fam] = e
                print(f"{name:9s} {m:9s} {fam:7s} clean {e['clean_err_pct']:5.1f}  effect {e['effect_pts']:+6.1f} "
                      f"[{e['ci95_pts'][0]:+.1f}, {e['ci95_pts'][1]:+.1f}]  orig {orig['effect_pts']:+6.1f}  "
                      f"both {e['both_processed_err_pct']:5.1f}/{e['err_pct']:5.1f}  units {e['units_passing']}", flush=True)
            out[ds][m] = res
    if only:
        return out  # interim look: the verdict needs both datasets
    out["verdict"] = {}
    for fam in ("reverb", "mixsep"):
        n = {ds: sum(out[ds][m][fam]["passes"] for m in SPECIALISED) for ds in DATASETS}
        spec = {ds: np.mean([out[ds][m][fam]["effect_pts"] for m in SPECIALISED]) for ds in DATASETS}
        out["verdict"][fam] = {"specialised_passing": n, "holds": all(v >= 2 for v in n.values()),
                               "specialised_mean_gt_ecapa_mert": {ds: bool(spec[ds] > max(
                                   out[ds]["ecapa"][fam]["effect_pts"], out[ds]["mert"][fam]["effect_pts"])) for ds in DATASETS}}
        print(f"VERDICT {fam}: {out['verdict'][fam]}")
    return out


def t3() -> dict:
    out = {"S": S}
    for ds, name in DATASETS.items():
        dirs = original_dirs(ds)
        out[ds] = {}
        for label, models in (("wavlm", ("wavlm",)), ("cvsm_art", ("cvsm_art",)), ("ecapa", ("ecapa",)),
                              ("fusion_cvsm_art_ecapa", ("cvsm_art", "ecapa"))):
            e = Bench(ROOT / "data" / ds, dirs, models).effect(S)
            out[ds][label] = e
            print(f"{name:9s} {label:22s} clean {e['clean_err_pct']:5.1f}  effect over S {e['effect_pts']:+6.1f} "
                  f"[{e['ci95_pts'][0]:+.1f}, {e['ci95_pts'][1]:+.1f}]", flush=True)
        t1_dir = TEMP / "data" / ds
        if (t1_dir / "emb_wavlm.npz").exists():
            b = Bench(t1_dir, {"wavlm": t1_dir}, ("wavlm",))
            out[ds]["wavlm_t1"] = {fam: b.effect([v for v in b.variants if v.startswith(fam + "@")])
                                   for fam in ("reverb", "mixsep")}
            print(f"{name:9s} wavlm on T1: " + ", ".join(f"{f} {e['effect_pts']:+.1f}" for f, e in out[ds]["wavlm_t1"].items()))
    fus, art = (lambda ds: out[ds]["fusion_cvsm_art_ecapa"]), (lambda ds: out[ds]["cvsm_art"])
    out["fusion_useful_baseline"] = all(fus(ds)["effect_pts"] <= 0.5 * art(ds)["effect_pts"]
                                        and fus(ds)["clean_err_pct"] <= art(ds)["clean_err_pct"] + 2 for ds in DATASETS)
    print("fusion useful baseline (protocol rule):", out["fusion_useful_baseline"])
    return out


if __name__ == "__main__":
    test, only = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    res = t1(only) if test == "t1" else t3()
    (TEMP / f"{test}{'_' + only if only else ''}_results.json").write_text(json.dumps(res, indent=1))
