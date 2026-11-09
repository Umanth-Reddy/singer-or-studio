import json
from pathlib import Path
from collections import defaultdict
import numpy as np
SOURCE=Path('/legacy/ps/claude-research/data')
FRESH=Path(__file__).resolve().parents[1]
MODELS=['cvsm_art','cvsm_a','ssl','ecapa','mert']
CONDS=['reverb@0.3','codec@6','mp3@64','mixsep@0','eq@12']
def read_vectors(path):
 d=np.load(path); v=d['vectors']; v=v/np.linalg.norm(v,axis=1,keepdims=True); return dict(zip(d['paths'].tolist(),v.astype(np.float32)))
def score(man,vec,pos='clean',query='clean',negative='clean'):
 sex={r['singer']:r['sex'] for r in man}; emb={(r['singer'],r['song'],v):vec[p[0]] for r in man for v,p in r['crops'].items() if p[0] in vec}
 songs=sorted({r['song'] for r in man}); per=defaultdict(list)
 for s in sorted(sex):
  for i in songs:
   q=emb.get((s,i,query))
   if q is None: continue
   for j in songs:
    if j==i: continue
    for_pos=emb.get((s,j,pos))
    if for_pos is None: continue
    sp=float(q@for_pos)
    for s2 in sorted(sex):
     if s2==s or sex[s2]!=sex[s]: continue
     n=emb.get((s2,j,negative))
     if n is not None: per[s].append(float(q@n)>=sp)
 return {s:float(np.mean(v)) for s,v in per.items()}
def effect(a,b):
 k=sorted(set(a)&set(b)); d=np.array([a[s]-b[s] for s in k]); rng=np.random.default_rng(0); boots=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)])
 return {'err_pct':float(np.mean([a[s] for s in k])*100),'effect_pts':float(d.mean()*100),'ci95_pts':[float(np.percentile(boots,2.5)*100),float(np.percentile(boots,97.5)*100)]}
out={'note':'fresh inference on source dataset crops; trial metric independently reimplemented; existing embeddings compared on same six conditions','datasets':{}}
for ds in ('f2','g1'):
 man=json.loads((FRESH/'manifests'/f'{ds}.json').read_text()); out['datasets'][ds]={'singers':len({r['singer'] for r in man}),'models':{}}
 for m in MODELS:
  if not (FRESH/'fresh_embeddings'/ds/f'emb_{m}.npz').exists(): continue
  fresh=read_vectors(FRESH/'fresh_embeddings'/ds/f'emb_{m}.npz'); original=read_vectors(SOURCE/ds/f'emb_{m}.npz')
  row={}
  bf=score(man,fresh); bo=score(man,original)
  row['fresh_clean_error_pct']=float(np.mean(list(bf.values()))*100); row['original_clean_error_pct']=float(np.mean(list(bo.values()))*100)
  row['conditions']={}
  for c in CONDS:
   ef=score(man,fresh)
   p_f=score(man,fresh,query='clean',negative='clean')
   # score function's pos is negative arg; both control uses same candidate transform via query/negative args
   cur=score(man,fresh,pos=c,query='clean',negative='clean')
   orig=score(man,original,pos=c,query='clean',negative='clean')
   both=score(man,fresh,pos=c,query=c,negative=c)
   row['conditions'][c]={'fresh':effect(cur,p_f),'original':effect(orig,bo),'both_processed_error_pct':float(np.mean(list(both.values()))*100)}
  out['datasets'][ds]['models'][m]=row
(FRESH/'fresh_scores.json').write_text(json.dumps(out,indent=2))
for ds,dsr in out['datasets'].items():
 print('\n',ds,'singers',dsr['singers'])
 for m,x in dsr['models'].items():
  print(m,'clean new/orig',round(x['fresh_clean_error_pct'],1),round(x['original_clean_error_pct'],1),' | '+' '.join(f"{c} {x['conditions'][c]['fresh']['err_pct']:.1f} ({x['conditions'][c]['fresh']['effect_pts']:+.1f}; old {x['conditions'][c]['original']['err_pct']:.1f}) both {x['conditions'][c]['both_processed_error_pct']:.1f}" for c in CONDS))
