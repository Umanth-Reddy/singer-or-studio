#!/usr/bin/env python3
"""Fixed-strength paired residual adapter; six sorted artists train, six held out."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE/"results"
STRENGTH={"cvsm_art":1000.0,"cvsm_a":1000.0,"ssl":1.0}
def main():
 torch.set_num_threads(3)
 for name,strength in STRENGTH.items():
  d=np.load(R/f"{name}.npz");artists=d["artists"];names=sorted(set(artists.tolist()))
  cut=len(names)//2;train_names=set(names[:cut]);test_names=set(names[cut:])
  tr=np.flatnonzero(np.isin(artists,list(train_names)));te=np.flatnonzero(np.isin(artists,list(test_names)))
  clean=torch.from_numpy(d["native"][tr].astype(np.float32));aug=torch.from_numpy(d["mp3_32"][tr].astype(np.float32));y=artists[tr]
  pos=[[j for j in range(len(tr)) if y[j]==y[i] and j!=i] for i in range(len(tr))]
  neg=[[j for j in range(len(tr)) if y[j]!=y[i]] for i in range(len(tr))]
  anchors=np.array([i for i in range(len(tr)) if pos[i] and neg[i]])
  if len(anchors)!=len(tr):raise RuntimeError("Every training clip needs same-artist positives and other-artist negatives")
  raw_native=d["native"][te].astype(np.float32);raw_low=d["mp3_32"][te].astype(np.float32)
  np.savez_compressed(R/f"heldout_raw_{name}.npz",native=raw_native,mp3_32=raw_low,ids=d["ids"][te],artists=artists[te])
  for seed in (0,1,2):
   torch.manual_seed(seed);rng=np.random.default_rng(seed)
   dim=clean.shape[1];down=nn.Linear(dim,64,bias=False);up=nn.Linear(64,dim,bias=False);nn.init.zeros_(up.weight)
   opt=torch.optim.Adam(list(down.parameters())+list(up.parameters()),lr=.001)
   def transform(x):return F.normalize(x+up(down(x)),dim=-1)
   for _ in range(400):
    a=rng.choice(anchors,256);pidx=np.array([rng.choice(pos[i]) for i in a]);nidx=np.array([rng.choice(neg[i]) for i in a])
    v=transform(clean[a]);p=transform(aug[pidx]);n=transform(clean[nidx])
    rank=F.softplus(((v*n).sum(1)-(v*p).sum(1)+.05)*10).mean()/10
    teacher=clean[a]@clean.T;adapted=v@transform(clean).T
    preserve=((teacher-adapted)**2).mean();align=(1-(transform(aug[a])*v).sum(1)).mean()
    loss=rank+strength*preserve+.1*align
    opt.zero_grad();loss.backward();opt.step()
   with torch.no_grad():
    z=transform(torch.from_numpy(np.concatenate([raw_native,raw_low]))).numpy();n=len(te)
    state={"down.weight":down.weight.detach().numpy(),"up.weight":up.weight.detach().numpy()}
   np.savez_compressed(R/f"adapter_{name}_s{seed}.npz",native=z[:n],mp3_32=z[n:],ids=d["ids"][te],artists=artists[te])
   torch.save({"state_dict":state,"strength":strength,"seed":seed,"steps":400,
    "train_artists":sorted(train_names),"test_artists":sorted(test_names),
    "objective":"soft ranking + strength*clean-score preservation + 0.1*paired alignment",
    "negative_control":"other artist; Music4All has no singer-sex labels"},R/f"adapter_{name}_s{seed}.pt")
   print(name,"seed",seed,"train/test artists",len(train_names),len(test_names),flush=True)
if __name__=="__main__":main()
