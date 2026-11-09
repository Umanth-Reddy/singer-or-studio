"""Exploratory clean-teacher ridge adapter, selected using held-out training singers only."""
import json,sys,time
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
VALID={'female3','female4','male4','male5'}
assert VALID.issubset({r['singer'] for r in manifest})
TRAIN={r['singer'] for r in manifest}-VALID
print('train singers',sorted(TRAIN),'validation singers',sorted(VALID),flush=True)

def norm(x): return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def data(model):
 v=m1.load('m1',model); rows=[]
 for i,r in enumerate(manifest):
  clean=v[r['crops']['clean'][0]]
  for n,p in r['crops'].items():
   if p[0] in v: rows.append((v[p[0]],clean,i,r['singer'],r['sex'],r['source'],n))
 return rows

def fit(rows,lam):
 torch.manual_seed(0)
 X=torch.from_numpy(np.stack([r[0] for r in rows])); y=torch.tensor([r[2] for r in rows]); clean=torch.tensor([i for i,r in enumerate(rows) if r[6]=='clean'])
 net=torch.nn.Sequential(torch.nn.Linear(X.shape[1],512),torch.nn.ReLU(),torch.nn.Linear(512,256)); opt=torch.optim.AdamW(net.parameters(),lr=0.001)
 teacher=X[clean]@X[clean].T
 for _ in range(300):
  opt.zero_grad(); z=F.normalize(net(X),dim=1); gram=z[clean]@z[clean].T
  loss=m1.supcon(z,y)+lam*torch.mean((gram-teacher)**2); loss.backward();opt.step()
 return net.eval()

def transform(X,net):
 with torch.no_grad(): return F.normalize(net(torch.from_numpy(X.astype('float32'))),dim=1).numpy()

def val_metric(rows,W):
 # Compare crops from distinct source recordings, same singer versus same-sex other singer.
 emb=norm(np.stack([r[0] for r in rows])) if W is None else transform(np.stack([r[0] for r in rows]),W)
 cleanidx=[i for i,r in enumerate(rows) if r[6]=='clean']
 per={s:{'clean':[],'aug':[]} for s in VALID}
 for q in cleanidx:
  rq=rows[q]; sims=emb@emb[q]
  neg=np.array([sims[n] for n in cleanidx if rows[n][3]!=rq[3] and rows[n][4]==rq[4]])
  for p,rp in enumerate(rows):
   if rp[3]==rq[3] and rp[5]!=rq[5]:
    per[rq[3]]['clean' if rp[6]=='clean' else 'aug'].append(float(np.mean(neg>=sims[p])))
 return {k:float(np.mean([np.mean(per[s][k]) for s in VALID])*100) for k in ('clean','aug')}

def evalset(model,net,W,ds,kind):
 directory=ROOT/('data' if kind=='original' else 'TEMP/data')/ds
 man=json.loads((directory/'manifest.json').read_text()); d=np.load(directory/f'emb_{model}.npz')
 lookup=dict(zip(d['paths'].tolist(),norm(d['vectors']).astype('float32')))
 rows=[r for r in man if r['singer'] not in TRAIN|VALID]; sex={r['singer']:r['sex'] for r in rows}
 pairs=[((r['singer'],r['song'],n),lookup[p[0]]) for r in rows for n,p in r['crops'].items() if p[0] in lookup]
 keys=[p[0] for p in pairs]; V=np.stack([p[1] for p in pairs]); variants=sorted({k[2] for k in keys}-{'clean'})
 with torch.no_grad(): adapter=F.normalize(net(torch.from_numpy(V)),dim=1).numpy()
 spaces={'raw':V,'m2':adapter,'distill':transform(V,W)}
 families={'all':variants} if kind=='original' else {}
 for f in ('codec','mp3','mixsep','reverb','eq'):
  vs=[v for v in variants if v.startswith(f+'@')]
  if vs: families[f]=vs
 out={}
 for name,space in spaces.items():
  em=dict(zip(keys,space)); per={v:m1.trial_errors(em,sex,v) for v in ['clean']+variants}; singers=sorted(per['clean'])
  out[name]={k:float(np.mean([np.mean([per[v][s] for v in vs]) for s in singers])*100) for k,vs in {'clean':['clean'],**families}.items()}
 return {'n_singers':len(sex),'n_conditions':len(variants),'metrics':out}

allout={}
for model in m1.MODELS:
 t=time.time(); rows=data(model); fitrows=[r for r in rows if r[3] in TRAIN]; valrows=[r for r in rows if r[3] in VALID]
 table=[]
 for lam in (0.0,1.0,5.0,20.0):
  W=fit(fitrows,lam); metrics=val_metric(valrows,W); table.append({'lambda':lam,**metrics})
 baseline=val_metric(valrows,None)
 eligible=[r for r in table if r['clean'] <= baseline['clean']+3]
 selected=min(eligible,key=lambda x:(x['aug'],x['clean'])) if eligible else None
 print(model,'validation baseline',baseline,'candidates',table,'selected',selected,flush=True)
 if selected is None:
  allout[model]={'validation_raw':baseline,'candidates':table,'selected':None}; continue
 W=fit(rows,selected['lambda'])
 X=np.stack([r[0] for r in rows]); y=np.array([r[2] for r in rows]); m1.SEED=0; net=m1.train(X,y)
 result={'validation_raw':baseline,'candidates':table,'selected':selected,'tests':{}}
 for ds in ('f2','g1'):
  for kind in ('original','real'):
   val=evalset(model,net,W,ds,kind); result['tests'][ds+'_'+kind]=val
   print(model,ds,kind,val['metrics'],flush=True)
 allout[model]=result
 (ROOT/'exploratory_distill/results.json').write_text(json.dumps(allout,indent=2))
 print('seconds',round(time.time()-t,1),flush=True)
