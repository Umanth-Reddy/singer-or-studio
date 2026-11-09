import json
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path('/legacy/ps/claude-research')
PIL=ROOT/'pilots'; DATA=ROOT/'data'
MODELS=['cvsm_art','cvsm_a','ssl','ecapa','mert']

def load(pilot, model):
    d=np.load(DATA/pilot/f'emb_{model}.npz')
    v=d['vectors']; v=v/np.linalg.norm(v,axis=1,keepdims=True)
    return dict(zip(d['paths'].tolist(),v.astype(np.float32)))

def data(pilot, model):
    manifest=json.loads((DATA/pilot/'manifest.json').read_text())
    vec=load(pilot,model)
    sex={r['singer']:r['sex'] for r in manifest}
    emb={(r['singer'],r['song'],v):vec[p[0]] for r in manifest for v,p in r['crops'].items() if p[0] in vec}
    return manifest,sex,emb

def per_singer(sex,emb,variant,qv='clean',nv='clean'):
    songs=sorted({k[1] for k in emb}); per=defaultdict(list)
    for s in sorted(sex):
      for i in songs:
       q=emb.get((s,i,qv))
       if q is None: continue
       for j in songs:
        p=emb.get((s,j,variant))
        if p is None or i==j: continue
        sp=float(q@p)
        for s2 in sorted(sex):
         n=emb.get((s2,j,nv))
         if s2!=s and sex[s2]==sex[s] and n is not None: per[s].append(float(q@n)>=sp)
    return {s:float(np.mean(a)) for s,a in per.items()}

def ci(d, seed=0):
    rng=np.random.default_rng(seed); b=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)])
    return [float(np.percentile(b,2.5)*100),float(np.percentile(b,97.5)*100)]

def official(pilot):
 p=PIL/f'{pilot}_results_final.json'
 if not p.exists(): p=PIL/f'{pilot}_results.json'
 return {r['model']:r for r in json.loads(p.read_text())}

out={'source':'read-only /legacy/ps/claude-research/data; recomputed by independent script; inputs are stored encoder embeddings and manifests','f2':{},'g1':{},'c1':{},'input_inventory':{}}
for model in MODELS:
 m,sex,emb=data('f2',model); f2res={}
 base=per_singer(sex,emb,'clean'); f2res['clean_error_pct']=float(np.mean(list(base.values()))*100)
 for v in sorted({k[2] for k in emb if k[2]!='clean'}):
  e=per_singer(sex,emb,v); keys=sorted(set(base)&set(e)); d=np.array([e[s]-base[s] for s in keys]); both=per_singer(sex,emb,v,nv=v)
  f2res[v]={'error_pct':float(np.mean([e[s] for s in keys])*100),'effect_pts':float(d.mean()*100),'ci95_pts':ci(d),'both_processed_error_pct':float(np.mean(list(both.values()))*100)}
 failing=[v for v,x in f2res.items() if v not in ('clean','clean_error_pct') and x['effect_pts']>=10 and x['ci95_pts'][0]>0]
 f2res['failing_variants']=failing
 out['f2'][model]=f2res
 # compare common variants with stored original JSON
 stored=official('f2').get(model,{})
 comp={}
 if stored:
  comp['clean_abs_diff_pts']=abs(f2res['clean_error_pct']-stored['baseline_error_pct'])
  for v in ('reverb@0.3','codec@6','mp3@64','mixsep@0','eq@12'):
   if v in stored.get('levels',{}) and v in f2res: comp[v+'_abs_diff_pts']=abs(f2res[v]['error_pct']-stored['levels'][v]['error_pct'])
 out['f2'][model]['difference_vs_archived']=comp
# G1 effect at VocalSet-failing strengths per model, plus union for extra encoders.
union=sorted(set(out['f2']['cvsm_art']['failing_variants'])|set(out['f2']['ssl']['failing_variants']))
for model in MODELS:
 m,sex,emb=data('g1',model); base=per_singer(sex,emb,'clean')
 variants=out['f2'][model]['failing_variants'] if model in ('cvsm_art','cvsm_a','ssl') else union
 per={v:per_singer(sex,emb,v) for v in variants}
 keys=sorted(set(base).intersection(*(set(x) for x in per.values())))
 d=np.array([np.mean([per[v][s] for v in variants])-base[s] for s in keys])
 both=np.array([np.mean([per_singer(sex,emb,v,nv=v)[s] for v in variants])-base[s] for s in keys])
 out['g1'][model]={'n_singers':len(keys),'baseline_error_pct':float(np.mean([base[s] for s in keys])*100),'failing_variants_from_f2':variants,'cross_error_pct':float(np.mean([np.mean([per[v][s] for v in variants]) for s in keys])*100),'effect_pts':float(d.mean()*100),'ci95_pts':ci(d),'both_processed_error_pct':float(np.mean([per_singer(sex,emb,v,nv=v)[s] for v in variants for s in keys])*100)}
# Recompute C1 arms using same fixed S: ART+SSL F2 fail variants.
S=union
for ds in ('f2','g1'):
 arms={}
 for arm in ('c1base','c1A','c1B','c1Bnorev'):
  man,sex,emb=data(ds,arm); base=per_singer(sex,emb,'clean'); variants={v:per_singer(sex,emb,v) for v in S}; keys=sorted(set(base).intersection(*(set(x) for x in variants.values())))
  e=np.array([np.mean([variants[v][s] for v in S])-base[s] for s in keys]); c=np.array([base[s] for s in keys])
  arms[arm]={'n_singers':len(keys),'clean_error_pct':float(c.mean()*100),'effect_pts':float(e.mean()*100)}
 out['c1'][ds]=arms
# dataset manifests, unique singers/tracks, embedding coverage
for ds in ('f2','g1'):
 man=json.loads((DATA/ds/'manifest.json').read_text()); out['input_inventory'][ds]={'manifest_rows':len(man),'singers':len({r['singer'] for r in man}),'songs':len({(r['singer'],r['song']) for r in man}),'conditions':sorted({v for r in man for v in r['crops']})}
(Path(__file__).resolve().parents[1]/'full_saved_embedding_rescore.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'inventory':out['input_inventory'],'f2':{m:{'clean':round(x['clean_error_pct'],2),'key':{v:round(x[v]['error_pct'],2) for v in ['reverb@0.3','codec@6','mp3@64','mixsep@0','eq@12']},'max_diff_archived':max(x['difference_vs_archived'].values(),default=0)} for m,x in out['f2'].items()},'g1':out['g1'],'c1':out['c1']},indent=2))
