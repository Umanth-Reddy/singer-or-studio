"""Exploratory M2 seed and held-out production checks; does not touch original results."""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pilots'))
import m1_train_eval as m1
m1.DATA=Path('/legacy/ps/claude-research/data')
torch.set_num_threads(2)
manifest=json.loads((m1.DATA/'m1/manifest.json').read_text())
train_singers={r['singer'] for r in manifest}
results=[]
for model in m1.MODELS:
    tv=m1.load('m1',model)
    rows=[(tv[paths[0]],i) for i,r in enumerate(manifest) for paths in r['crops'].values() if paths[0] in tv]
    X=np.stack([x for x,y in rows]); y=np.array([y for x,y in rows])
    datasets={}
    for ds in ('f2','g1'):
      for kind in ('original','real'):
        directory=(ROOT/'data'/ds) if kind=='original' else (ROOT/'TEMP/data'/ds)
        man=json.loads((directory/'manifest.json').read_text())
        data=np.load(directory/f'emb_{model}.npz'); vv=data['vectors']; vv=vv/np.linalg.norm(vv,axis=1,keepdims=True); lookup=dict(zip(data['paths'].tolist(),vv.astype(np.float32)))
        rows2=[r for r in man if r['singer'] not in train_singers]
        sex={r['singer']:r['sex'] for r in rows2}
        kv=[((r['singer'],r['song'],v),lookup[p[0]]) for r in rows2 for v,p in r['crops'].items() if p[0] in lookup]
        keys=[k for k,v in kv]; V=np.stack([v for k,v in kv]); variants=sorted(set(k[2] for k in keys)-{'clean'})
        datasets[(ds,kind)]=(sex,keys,V,variants)
    for seed in (0,1,2):
      m1.SEED=seed; net=m1.train(X,y)
      for (ds,kind),(sex,keys,V,variants) in datasets.items():
        with torch.no_grad(): adapted=F.normalize(net(torch.from_numpy(V)),dim=1).numpy()
        family={'all':variants} if kind=='original' else {}
        for f in ('codec','mp3','mixsep','reverb','eq'):
          vs=[v for v in variants if v.startswith(f+'@')]
          if vs: family[f]=vs
        if kind=='original':
          old=next(x for x in json.loads((ROOT/'pilots/f2_results_final.json').read_text()) if x['model']==model)
          family['selected_failing']=[v for v,s in old['levels'].items() if s['effect_pts']>=10 and s['ci95_pts'][0]>0 and v in variants]
        spaces={'raw':dict(zip(keys,V)),'m2':dict(zip(keys,adapted))}
        per={space:{v:m1.trial_errors(emb,sex,v) for v in ['clean']+variants} for space,emb in spaces.items()}
        singers=sorted(set(per['raw']['clean']) & set(per['m2']['clean']))
        def values(space,vs):
          return np.array([np.mean([per[space][v][s] for v in vs if s in per[space][v]]) for s in singers])*100
        dataout={}
        rng=np.random.default_rng(42)
        for name,vs in {'clean':['clean'],**family}.items():
          if not vs: continue
          a,b=values('raw',vs),values('m2',vs); d=b-a
          boot=np.mean(d[rng.integers(0,len(d),(5000,len(d)))],axis=1)
          dataout[name]={'n_variants':len(vs),'raw':float(a.mean()),'m2':float(b.mean()),'difference_pts':float(d.mean()),'singer_bootstrap_ci95':np.percentile(boot,[2.5,97.5]).tolist()}
        results.append({'model':model,'seed':seed,'dataset':ds,'kind':kind,'n_singers':len(singers),'metrics':dataout})
        print(model,seed,ds,kind, {k:(round(v['raw'],1),round(v['m2'],1)) for k,v in dataout.items()},flush=True)
        (ROOT/'exploratory_m2/results.json').write_text(json.dumps(results,indent=2))
