"""M1 step 2: train adapters on TRAIN-singer embeddings, evaluate on F2 trials of TEST singers.

  ~/ps/.venv-ssl/bin/python -I pilots/m1_train_eval.py
Protocol: docs/04_pilot_log.md (M1). Adapters: "aug" (clean + all augmentations),
"clean" (clean crops only), and leave-one-family-out "no_<family>". Raw cosine is the
reference. Writes pilots/m1_results.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
MODELS = ("cvsm_art", "cvsm_a", "ssl")
FAMILIES = ("reverb", "codec", "eq", "mixsep", "comp")
TEST_FAMILY = {"reverb": "reverb", "codec": "codec", "mp3": "codec", "mixsep": "mixsep", "eq": "eq"}
EPOCHS, LR, TEMP, SEED = 300, 1e-3, 0.1, 0


def load(pilot: str, model: str) -> dict[str, np.ndarray]:
    d = np.load(DATA / pilot / f"emb_{model}.npz")
    v = d["vectors"] / np.linalg.norm(d["vectors"], axis=1, keepdims=True)
    return dict(zip(d["paths"].tolist(), v.astype(np.float32)))


def supcon(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = F.normalize(z, dim=1)
    logits = z @ z.T / TEMP
    eye = torch.eye(len(z), dtype=torch.bool)
    logits = logits.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob * pos).sum(1).div(pos.sum(1).clamp(min=1)).mean()


def train(x: np.ndarray, y: np.ndarray) -> torch.nn.Module:
    torch.manual_seed(SEED)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 512), torch.nn.ReLU(), torch.nn.Linear(512, 256))
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = supcon(net(xt), yt)
        loss.backward()
        opt.step()
    return net.eval()


def trial_errors(emb: dict, sex: dict, variant: str) -> dict[str, float]:
    """F2 trial: clean query, processed same-singer positive, clean same-sex negatives."""
    singers = sorted(sex)
    songs = sorted({k[1] for k in emb})
    out = {}
    for s in singers:
        errs = []
        for i in songs:
            q = emb.get((s, i, "clean"))
            for j in songs:
                pos = emb.get((s, j, variant))
                if q is None or pos is None or i == j:
                    continue
                sp = float(q @ pos)
                for s2 in singers:
                    neg = emb.get((s2, j, "clean"))
                    if s2 != s and sex[s2] == sex[s] and neg is not None:
                        errs.append(float(q @ neg) >= sp)
        if errs:
            out[s] = float(np.mean(errs))
    return out


def evaluate(model: str) -> dict:
    train_vec, test_vec = load("m1", model), load("f2", model)
    manifest = json.loads((DATA / "m1" / "manifest.json").read_text())
    singer_id = {s: k for k, s in enumerate(sorted({r["singer"] for r in manifest}))}
    rows = []  # (vector, singer id, family or "clean")
    for r in manifest:
        for name, paths in r["crops"].items():
            if paths[0] in train_vec:
                fam = "clean" if name == "clean" else r["families"][name]["family"]
                rows.append((train_vec[paths[0]], singer_id[r["singer"]], fam))
    X = np.stack([r[0] for r in rows])
    Y = np.array([r[1] for r in rows])
    fam = np.array([r[2] for r in rows])
    train_singers = set(singer_id)

    f2 = json.loads((DATA / "f2" / "manifest.json").read_text())
    test = [r for r in f2 if r["singer"] not in train_singers]
    sex = {r["singer"]: r["sex"] for r in test}
    keys, vecs = [], []
    for r in test:
        for name, paths in r["crops"].items():
            if paths[0] in test_vec:
                keys.append((r["singer"], r["song"], name))
                vecs.append(test_vec[paths[0]])
    V = np.stack(vecs)

    adapters = {"aug": train(X, Y), "clean": train(X[fam == "clean"], Y[fam == "clean"])}
    for f in FAMILIES:
        keep = fam != f
        adapters[f"no_{f}"] = train(X[keep], Y[keep])

    spaces = {"raw": V}
    with torch.no_grad():
        for name, net in adapters.items():
            spaces[name] = F.normalize(net(torch.from_numpy(V)), dim=1).numpy()

    f2res = next(r for r in json.loads((HERE / "f2_results_final.json").read_text()) if r["model"] == model)
    failing = [v for v, s in f2res["levels"].items() if s["effect_pts"] >= 10 and s["ci95_pts"][0] > 0]
    variants = sorted({k[2] for k in keys} - {"clean"})
    per = {}  # space -> variant -> {singer: error}
    for name, M in spaces.items():
        emb = dict(zip(keys, M))
        per[name] = {v: trial_errors(emb, sex, v) for v in variants + ["clean"]}
    test_singers = sorted(per["raw"]["clean"])

    def singer_mean(space: str, vs: list[str]) -> np.ndarray:
        return np.array([np.mean([per[space][v][s] for v in vs]) for s in test_singers])

    rng = np.random.default_rng(0)

    def ci(d: np.ndarray) -> list[float]:
        b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        return [float(np.percentile(b, 2.5) * 100), float(np.percentile(b, 97.5) * 100)]

    res = {"model": model, "n_train_vectors": len(X), "test_singers": test_singers, "failing_variants": failing,
           "failing_mean_error_pct": {}, "clean_error_pct": {}, "per_variant_error_pct": {}}
    for name in spaces:
        res["failing_mean_error_pct"][name] = float(singer_mean(name, failing).mean() * 100)
        res["clean_error_pct"][name] = float(singer_mean(name, ["clean"]).mean() * 100)
        res["per_variant_error_pct"][name] = {v: float(np.mean(list(per[name][v].values())) * 100) for v in variants}
    raw, aug, clean = (singer_mean(n, failing) for n in ("raw", "aug", "clean"))
    res["aug_relative_reduction_vs_raw"] = float(1 - aug.mean() / raw.mean())
    res["aug_minus_clean_pts"] = float((aug - clean).mean() * 100)
    res["aug_minus_clean_ci95_pts"] = ci(aug - clean)
    res["clean_trial_change_pts"] = res["clean_error_pct"]["aug"] - res["clean_error_pct"]["raw"]
    res["pass"] = bool(res["aug_relative_reduction_vs_raw"] >= 0.5 and res["aug_minus_clean_pts"] <= -10
                       and res["aug_minus_clean_ci95_pts"][1] < 0 and res["clean_trial_change_pts"] <= 3)
    # Leave-one-family-out: error on held-out family's failing variants, adapter never saw that family.
    res["lofo"] = {}
    for f in ("reverb", "codec", "mixsep"):
        vs = [v for v in failing if TEST_FAMILY[v.split("@")[0]] == f]
        if vs:
            res["lofo"][f] = {n: float(singer_mean(n, vs).mean() * 100) for n in ("raw", "clean", "aug", f"no_{f}")}
    return res


def main() -> None:
    results = [evaluate(m) for m in MODELS if (DATA / "m1" / f"emb_{m}.npz").exists()]
    (HERE / "m1_results.json").write_text(json.dumps(results, indent=1))
    for r in results:
        e, c = r["failing_mean_error_pct"], r["clean_error_pct"]
        print(f"\n== {r['model']}  ({len(r['test_singers'])} test singers, {len(r['failing_variants'])} failing strengths)")
        print("   mean error on failing strengths: " + "  ".join(f"{k} {e[k]:.1f}%" for k in ("raw", "clean", "aug")))
        print("   clean-trial error:              " + "  ".join(f"{k} {c[k]:.1f}%" for k in ("raw", "clean", "aug")))
        print(f"   aug vs raw: -{r['aug_relative_reduction_vs_raw'] * 100:.0f}% relative | aug - clean-adapter "
              f"{r['aug_minus_clean_pts']:+.1f} pts CI {[round(x, 1) for x in r['aug_minus_clean_ci95_pts']]} | PASS {r['pass']}")
        for f, v in r["lofo"].items():
            print(f"   held-out {f:7s}: " + "  ".join(f"{k} {x:.1f}%" for k, x in v.items()))
    gate = [r["pass"] for r in results if r["model"] in ("cvsm_art", "ssl")]
    print("\nM1 PASS (CVSM-ART and SSL):", len(gate) == 2 and all(gate))


if __name__ == "__main__":
    main()
