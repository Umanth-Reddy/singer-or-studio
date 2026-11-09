#!/usr/bin/env python3
"""Artist-macro pairwise ranking error with singer/artist bootstrap uncertainty."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;R=HERE/"results"
def metrics(native,low,artists):
 out={m:{} for m in ("native","mismatch_32","both_32")}
 for artist in sorted(set(artists.tolist())):
  ids=np.flatnonzero(artists==artist);per={m:[] for m in out}
  for q in ids:
   pos=ids[ids!=q];neg=np.flatnonzero(artists!=artist)
   for m,qv,pv,nv in (("native",native[q],native[pos],native[neg]),("mismatch_32",native[q],low[pos],native[neg]),("both_32",low[q],low[pos],low[neg])):
    per[m].append(float(((nv@qv)[:,None]>=(pv@qv)[None,:]).mean()))
  for m in out:out[m][artist]=100*float(np.mean(per[m]))
 return out
def main():
 result={}
 for path in sorted(R.glob("*.npz")):
  if path.name=="inputs.npz":continue
  d=np.load(path);a=metrics(d["native"],d["mp3_32"],d["artists"]);summary={}
  for mode,vals in a.items():summary[mode]={"macro_error_percent":float(np.mean(list(vals.values()))),"per_artist_error_percent":vals}
  names=sorted(a["native"]);delta=np.array([a["mismatch_32"][n]-a["native"][n] for n in names]);rng=np.random.default_rng(20261109)
  b=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(10000)])
  summary["mismatch_minus_native_pp"]={"mean":float(delta.mean()),"artist_bootstrap_95pct":[float(x) for x in np.quantile(b,[.025,.975])],"resampling_unit":"artist"}
  summary["n_artists"]=len(names);summary["n_recordings"]=int(len(d["ids"]))
  summary["definition"]="For each query, compare positives from different tracks by same artist against all other-artist negatives; error if negative cosine >= positive cosine. Average within query, then equally across artists."
  result[path.stem]=summary
  print(path.stem,{m:round(v["macro_error_percent"],2) for m,v in summary.items() if isinstance(v,dict) and "macro_error_percent" in v},flush=True)
 (R/"metrics.json").write_text(json.dumps(result,indent=2));print(R/"metrics.json")
if __name__=="__main__":main()
