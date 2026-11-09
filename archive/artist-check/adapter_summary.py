#!/usr/bin/env python3
"""Paired artist-level raw-versus-adapter comparison on held-out artists."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;R=HERE/"results"
def per_artist(d):
 out={m:{} for m in ("native","mismatch_32","both_32")}
 for artist in sorted(set(d["artists"].tolist())):
  ids=np.flatnonzero(d["artists"]==artist);qmeans={m:[] for m in out}
  for q in ids:
   pos=ids[ids!=q];neg=np.flatnonzero(d["artists"]!=artist)
   for mode,qv,pv,nv in (
    ("native",d["native"][q],d["native"][pos],d["native"][neg]),
    ("mismatch_32",d["native"][q],d["mp3_32"][pos],d["native"][neg]),
    ("both_32",d["mp3_32"][q],d["mp3_32"][pos],d["mp3_32"][neg])):
    qmeans[mode].append(float(((nv@qv)[:,None]>=(pv@qv)[None,:]).mean())*100)
  for mode in out:out[mode][artist]=float(np.mean(qmeans[mode]))
 return out
def main():
 result={}
 for model in ("cvsm_art","cvsm_a","ssl"):
  raw=np.load(R/f"heldout_raw_{model}.npz");rawm=per_artist(raw);result[model]={}
  for seed in (0,1,2):
   adapt=np.load(R/f"adapter_{model}_s{seed}.npz")
   np.testing.assert_array_equal(raw["ids"],adapt["ids"]);np.testing.assert_array_equal(raw["artists"],adapt["artists"])
   am=per_artist(adapt);record={}
   for mode in rawm:
    names=sorted(rawm[mode]);delta=np.array([am[mode][n]-rawm[mode][n] for n in names]);rng=np.random.default_rng(20261109+seed)
    boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(10000)])
    record[mode]={"raw_error_percent":float(np.mean([rawm[mode][n] for n in names])),
      "adapter_error_percent":float(np.mean([am[mode][n] for n in names])),"adapter_minus_raw_pp":float(delta.mean()),
      "artist_bootstrap_95pct":[float(v) for v in np.quantile(boot,[.025,.975])],
      "per_artist_adapter_minus_raw_pp":{n:float(am[mode][n]-rawm[mode][n]) for n in names}}
   result[model][str(seed)]=record
   print(model,"seed",seed,"mismatch change",round(record["mismatch_32"]["adapter_minus_raw_pp"],2),
     "pp; clean change",round(record["native"]["adapter_minus_raw_pp"],2),"pp",flush=True)
 (R/"adapter_comparison.json").write_text(json.dumps(result,indent=2))
if __name__=="__main__":main()
