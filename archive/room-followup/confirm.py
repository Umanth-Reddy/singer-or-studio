"""Render held-out response pairs, run frozen encoders, and score mixed galleries."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, resample_poly
import soundfile as sf

PS = Path("/legacy/ps")
GB = PS / "glass-breaker"
WS = Path("/legacy/workspace")
T1 = GB / "q2-study/TEMP"
RIRS = PS / "claude-research/TEMP/data/rirs_mit/Audio"
RATE = 44100
WIN = RATE * 4
WET = (0.2, 0.5)
MODELS = {
    "ssl": GB / "models/original-ssl/ssl_contrastive_vc/model.ts",
    "cvsm_art": GB / "models/original-cvsm/htdemucs/m4all_cvsmart",
}


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def decay(path):
    samples, rate = sf.read(path, dtype="float64", always_2d=True)
    x = samples.mean(axis=1)
    x = x[int(np.argmax(np.abs(x))):]
    power = np.cumsum(x[::-1] ** 2)[::-1]
    if not len(power) or power[0] == 0:
        return None
    db = 10 * np.log10(power / power[0] + 1e-30)
    i5, i35 = np.argmax(db <= -5), np.argmax(db <= -35)
    if db[-1] > -35 or i35 <= i5:
        return None
    slope = np.polyfit(np.arange(i5, i35) / rate, db[i5:i35], 1)[0]
    return float(-60 / slope) if slope < 0 else None


def room_pairs(sha256):
    categories = ("bedroom", "livingroom", "office", "bathroom", "kitchen", "diningroom",
                  "gym", "classroom", "hospital", "library", "hallway", "theater", "church")
    excluded = {r["file"] for r in read(T1 / "data/selection.json")["rooms"]}
    candidates = []
    for path in sorted(RIRS.glob("*.wav")):
        match = re.fullmatch(r"h\d+_(.+)_\d+txts\.wav", path.name)
        if path.name in excluded or not match:
            continue
        label = re.sub(r"[^a-z]", "", match[1].lower())
        category = next((c for c in categories if label.startswith(c)), None)
        if not category:
            continue
        t30 = decay(path)
        if t30 is not None and 0.2 <= t30 <= 1.2:
            candidates.append({"file": path.name, "t30": t30, "category": category, "sha256": sha256(path)})
    pairs, used = [], set()
    for target in (0.30, 0.45, 0.60, 0.80):
        eligible = [(a, b) for a, b in itertools.combinations(candidates, 2)
                    if a["category"] != b["category"] and max(a["t30"], b["t30"]) - min(a["t30"], b["t30"]) <= .08
                    and a["sha256"] != b["sha256"] and a["sha256"] not in used and b["sha256"] not in used]
        a,b = min(eligible, key=lambda p: (sum(abs(r["t30"] - target) for r in p),
                                          abs(p[0]["t30"] - p[1]["t30"]), p[0]["file"], p[1]["file"]))
        pairs.append([a, b])
        used.update((a["sha256"], b["sha256"]))
    return pairs


def read_mono(path):
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != RATE:
        d = math.gcd(RATE, sr)
        a = resample_poly(a, RATE // d, sr // d)
    if not np.isfinite(a).all() or not len(a):
        raise ValueError(f"Invalid input {path}")
    return a


def crop4(x, start):
    if start is None:
        start = (len(x) - WIN) // 2
    a = x[max(0,start):max(0,start)+WIN]
    left = max(0, -start)
    a = np.pad(a[:WIN-left], (left, max(0, WIN - len(a) - left)))
    peak = np.max(np.abs(a))
    if a.shape != (WIN,) or peak <= 0:
        raise ValueError("Invalid crop")
    return a * (0.5 / peak)


def render(root, sha256):
    pairs = room_pairs(sha256)
    save(root / "room_pairs.json", pairs)
    responses = {}
    print("Response pairs:", [[(r["file"],round(r["t30"],3)) for r in p] for p in pairs], flush=True)
    for i, room in enumerate(itertools.chain.from_iterable(pairs)):
        y = read_mono(RIRS / room["file"])
        y = y[np.argmax(np.abs(y)):]
        responses[i] = y / np.linalg.norm(y)
    covox = PS / "Q1-candidate-6/.external_data/covox"
    rows = [{"dataset":"covox", "singer":f"S{s:02}", "song":song, "group":"female",
             "source":str(covox / f"S{s:02}_{song}_pop_L.wav"), "start":None}
            for s in range(1,23) for song in ("AL","CH")]
    nus = read(GB / "shareable-research/experiments/nus_paired/manifest.json")
    # The local publisher inventory does not attach authoritative groups to these IDs.
    rows += [{"dataset":"nus", "singer":r["singer"], "song":r["song"], "group":"all",
              "source":r["source"], "start":r["start"]} for r in nus]
    audio = root / "audio"
    audio.mkdir(exist_ok=True)
    for index, row in enumerate(rows):
        src = Path(row["source"])
        x = read_mono(src)
        row["sha256"] = sha256(src)
        row["crops"] = {}

        def write(label, y):
            p = audio / f"{index:03}_{label}.wav"
            sf.write(p, crop4(y, row["start"]), RATE, subtype="FLOAT")
            row["crops"][label] = str(p.relative_to(root))

        write("clean", x)
        x_std = np.std(x)
        for i, rir in responses.items():
            tail = fftconvolve(x, rir)[:len(x)]
            tail *= x_std / (np.std(tail) + 1e-9)
            for wet in WET:
                write(f"{wet}_r{i}", (1-wet)*x + wet*tail)
    save(root / "manifest.json", rows)
    print(f"Rendered {len(rows)} recordings and {sum(len(r['crops']) for r in rows)} crops.", flush=True)


def inference(root, model, sha256):
    sys.path.insert(0, str(GB / "q2-study/external"))
    import gate1_experiment as gate
    gate.MANIFEST = root / "manifest.json"
    rows = read(gate.MANIFEST)
    indexed = [(r["dataset"], r["singer"], r["song"], c, f) for r in rows for c, f in r["crops"].items()]
    inputs = [{"audio_path": f} for *_, f in indexed]
    checkpoint = MODELS[model]
    if model == "ssl":
        gate.SSL_CHECKPOINT = checkpoint
        vectors = gate.run_ssl(inputs)
        files = [checkpoint]
    else:
        gate.CVSM_REPO = PS / "existing-work/CVSM"
        gate.CVSM_CHECKPOINT_DIR = checkpoint
        vectors = gate.run_cvsm(inputs)
        files = sorted(checkpoint.glob("checkpoint-40.*"))
    out = root / f"{model}.npz"
    np.savez_compressed(out, vectors=vectors, keys=np.asarray([k[:-1] for k in indexed]))
    result = {"count": len(vectors), "manifest_sha256": sha256(gate.MANIFEST),
              "embeddings_sha256": sha256(out), "checkpoints": {str(p):sha256(p) for p in files}}
    save(root / f"{model}_provenance.json", result)
    print(result, flush=True)


def mixed_gallery(data, trials, pair, seed):
    """Balanced impostor mixes, with condition assignment independent of label."""
    rng = np.random.default_rng(seed)
    # Each group corresponds to one query recording and one genuine recording.
    cuts = np.r_[0, np.flatnonzero(np.diff(trials.q) | np.diff(trials.p))+1, len(trials.q)]
    owners = {r[0]: i for i, r in enumerate(sorted({(s,) for s, _ in trials.recordings}))}
    rates = np.zeros((trials.n_singers, 6))
    totals = np.zeros(trials.n_singers)
    similarities = {}
    for q, c in itertools.product(range(2), repeat=2):
        a,b = data[pair[q]], data[pair[c]]
        similarities[q,c] = (np.einsum("ij,ij->i",a[trials.q],b[trials.p]),
                             np.einsum("ij,ij->i",a[trials.q],b[trials.n]))
    for start, end in zip(cuts[:-1], cuts[1:]):
        size = end - start
        indices = np.argsort(rng.random((16, size)), axis=1)
        mask = indices < size//2
        mask = np.vstack([mask, ~mask])
        sid = owners[trials.recordings[trials.q[start]][0]]
        for q,p in itertools.product(range(2), repeat=2):
            genuine = similarities[q,p][0][start]
            scores = np.where(mask, similarities[q,q][1][start:end],similarities[q,1-q][1][start:end])
            winner = np.argmax(scores,axis=1)
            is_match = mask[np.arange(len(mask)),winner]
            failed = scores[np.arange(len(mask)),winner] >= genuine
            all_other_fails = np.max(similarities[q,1-q][1][start:end]) >= genuine
            all_same_fails = np.max(similarities[q,q][1][start:end]) >= genuine
            rates[sid] += [np.sum(failed), np.sum(failed & is_match),
                           np.sum(failed & ~is_match), np.sum(is_match),
                           len(mask) * all_other_fails, len(mask) * all_same_fails]
            totals[sid] += len(mask)
    return 100 * rates / totals[:, None]


def score(root, selected_models, base):
    manifest = read(root / "manifest.json")
    outputs = {"analysis": "held-out responses and new singer cohorts for this reverb question",
               "bootstrap_replicates":base.N_BOOT, "room_pairs":read(root/"room_pairs.json"),
               "datasets":{}}
    for dataset in ("covox", "nus"):
        rows = [r for r in manifest if r["dataset"] == dataset]
        source_recordings = [(r["singer"], r["song"]) for r in rows]
        if dataset == "nus":
            # NUS singers do not all sing the same titles. Use one recording per
            # singer per gallery, deterministically ordered by title within singer.
            ordered_songs = {s: sorted(title for singer,title in source_recordings if singer == s)
                             for s in {s for s,_ in source_recordings}}
            if any(len(titles) != 4 or len(set(titles)) != 4 for titles in ordered_songs.values()):
                raise ValueError("NUS requires four distinct recordings per singer")
            trial_recordings = [(s, str(ordered_songs[s].index(song))) for s,song in source_recordings]
        else:
            trial_recordings = source_recordings
        trials = base.Trials(trial_recordings,{r["singer"]: r["group"] for r in rows})
        box = outputs["datasets"][dataset] = {"singers":trials.n_singers, "recordings":len(rows),
                                             "triplets_per_cell":len(trials.q), "models":{}}
        for name in selected_models:
            p = root / f"{name}.npz"
            prov = read(root/f"{name}_provenance.json")
            if prov["embeddings_sha256"] != base.sha256(p) or prov["manifest_sha256"] != base.sha256(root/"manifest.json"):
                raise ValueError("Manifest or embedding hash mismatch")
            with np.load(p) as d:
                indexed = dict(zip(map(tuple, d["keys"]), base.normalized(d["vectors"]), strict=True))
            data = {c:np.stack([indexed[dataset,s,song,c] for s,song in source_recordings]) for c in rows[0]["crops"]}
            clean = base.factorial(data,trials,("clean","clean"))[0,0,0][0]
            null_gallery = mixed_gallery(data,trials,("clean","clean"),20261119)
            if not (np.array_equal(null_gallery[:,1],null_gallery[:,2]) and
                    np.array_equal(null_gallery[:,0],null_gallery[:,4]) and
                    np.array_equal(null_gallery[:,0],null_gallery[:,5])):
                raise ValueError("Complementary assignments failed identical-embedding sanity check")
            model = box["models"][name] = {"provenance":prov, "clean":trials.summarize(clean), "levels":{}}
            for wet in WET:
                pair_effects, mixed=[],[]
                for i in range(4):
                    pair=(f"{wet}_r{2*i}",f"{wet}_r{2*i+1}")
                    result=base.contrasts(base.factorial(data,trials,pair))
                    pair_effects.append(result)
                    mixed.append(mixed_gallery(data,trials,pair,20261119+i))
                pooled={k:np.mean([p[k] for p in pair_effects],axis=0) for k in pair_effects[0]}
                mixed_per_singer=np.mean(mixed,axis=0)
                gallery=mixed_per_singer.mean(axis=0)
                # Secondary sensitivity, conditional on the fixed galleries and rooms.
                rng=np.random.default_rng(20261119)
                query_ids=rng.integers(0,trials.n_singers,size=(base.N_BOOT,trials.n_singers))
                excess=mixed_per_singer[:,1]-mixed_per_singer[:,2]
                gallery_ci=np.percentile(excess[query_ids].mean(axis=1),[2.5,97.5]).tolist()
                mix_vs_other = mixed_per_singer[:,0] - mixed_per_singer[:,4]
                mix_vs_other_ci = np.percentile(mix_vs_other[query_ids].mean(axis=1),[2.5,97.5]).tolist()
                err_key="impostor_condition_match_error_pp"
                model["levels"][str(wet)]={"pooled":base.summarize(trials,pooled),
                    "per_pair":[base.summarize(trials,r) for r in pair_effects],
                    "leave_one_pair_out_error_pp":[trials.summarize(np.mean([r[err_key] for j,r in enumerate(pair_effects)
                                                   if i != j],axis=0)) for i in range(len(pair_effects))],
                    "mixed_gallery":dict(zip(("top1_error_pct","matching_impostor_wins_pct",
                                             "other_impostor_wins_pct","matching_impostor_all_wins_pct",
                                             "all_other_top1_error_pct","all_same_top1_error_pct"),gallery)),
                    "mixed_gallery_excess_matched_error_pp":float(excess.mean()),
                    "posthoc_secondary_gallery_uncertainty": {
                        "fixed_gallery_query_bootstrap_ci95":gallery_ci,
                        "mixed_minus_all_other_top1_pp": {"value":float(mix_vs_other.mean()),"ci95":mix_vs_other_ci},
                        "per_pair_excess_matched_error_pp":[float((m[:,1]-m[:,2]).mean()) for m in mixed]}}
    if "ssl" in selected_models:
        passed = []
        for box in outputs["datasets"].values():
            level=box["models"]["ssl"]["levels"]["0.2"]
            key="impostor_condition_match_error_pp"
            primary=level["pooled"][key]
            passed.append(primary["value"]>5 and primary["ci95"][0]>2 and
                          sum(p[key]["value"]>0 for p in level["per_pair"])>=3 and
                          level["mixed_gallery_excess_matched_error_pp"]>0)
        outputs["advance_q1"]=all(passed)
    save(root / "mixed_gallery_results.json", outputs)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=("render","ssl","cvsm_art","score"))
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--base",type=Path,required=True)
    ap.add_argument("--models",nargs="+",default=list(MODELS),choices=MODELS)
    a=ap.parse_args()
    sys.path.insert(0,str(a.base))
    import analyze as base
    if a.stage=="render":
        render(a.out,base.sha256)
    elif a.stage=="score":
        score(a.out,a.models,base)
    else:
        inference(a.out,a.stage,base.sha256)


if __name__=="__main__":
    main()
