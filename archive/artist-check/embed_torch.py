#!/usr/bin/env python3
"""Run local TorchScript, SpeechBrain, WavLM, MERT, and Yakura encoders."""
import argparse,sys
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
HERE=Path(__file__).resolve().parent;BASE=HERE/"models"
CACHE=Path("/tmp/glass-breaker-cross-check-inputs.npz")
W={"ssl":BASE/"singer_identity_ssl/checkpoints/ssl_contrastive_vc/model.ts","ecapa":BASE/"ecapa/weights","wavlm":BASE/"wavlm/weights","mert":BASE/"mert/weights","yakura":BASE/"yakura/checkpoints/ckpt_epoch_100.pth"}
def rs(x,a,b):
 from math import gcd
 g=gcd(a,b);return resample_poly(x,b//g,a//g).astype(np.float32)
def build(name,torch):
 if name=="ssl":return torch.jit.load(str(W[name]),map_location="cpu").eval()
 if name=="ecapa":
  from speechbrain.inference.speaker import EncoderClassifier
  return EncoderClassifier.from_hparams(source=str(W[name]),savedir=str(W[name]),run_opts={"device":"cpu"})
 if name=="wavlm":
  from transformers import AutoFeatureExtractor,WavLMForXVector
  return AutoFeatureExtractor.from_pretrained(str(W[name])),WavLMForXVector.from_pretrained(str(W[name])).eval()
 if name=="mert":
  from transformers import AutoModel
  m=AutoModel.from_pretrained(str(W[name]),trust_remote_code=True).eval()
  return m
 if name=="yakura":
  sys.path.insert(0,str(BASE/"yakura/repo"));from lib.crnn import CRNN2D_elu
  c=torch.load(str(W[name]),map_location="cpu",weights_only=False);o=c["opts"];m=CRNN2D_elu(1+o.n_fft//2,o.feat_dim,dropout=0)
  m.load_state_dict({k.partition("module.")[2]:v for k,v in c["model"].items()},strict=True);return m.eval(),o
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",choices=W,required=True);a=p.parse_args()
 import torch;torch.set_num_threads(3);obj=build(a.model,torch);cache=np.load(CACHE);out={}
 for cond in ("native","mp3_32"):
  vv=[]
  for wav in cache[cond]:
   if a.model=="ssl":
    x=rs(wav,48000,44100);x/=max(float(np.max(np.abs(x))),1e-8)
    with torch.inference_mode():v=obj(torch.from_numpy(x[None])).numpy().reshape(-1)
   elif a.model=="ecapa":
    x=rs(wav,48000,16000)
    with torch.inference_mode():v=obj.encode_batch(torch.from_numpy(x)[None])[0,0].numpy()
   elif a.model=="wavlm":
    fe,m=obj;x=rs(wav,48000,16000);batch=fe(x,sampling_rate=16000,return_tensors="pt")
    with torch.inference_mode():v=m(**batch).embeddings[0].numpy()
   elif a.model=="mert":
    m=obj;states=[]
    hooks=[m.encoder.layer_norm.register_forward_hook(lambda _m,_i,o:states.append(o[0] if isinstance(o,tuple) else o))]
    hooks += [layer.register_forward_hook(lambda _m,_i,o:states.append(o[0] if isinstance(o,tuple) else o)) for layer in m.encoder.layers]
    x=rs(wav,48000,24000);x=torch.from_numpy(((x-x.mean())/(x.std()+1e-7))[None])
    with torch.inference_mode():m(x)
    for h in hooks:h.remove()
    if len(states)!=13:raise RuntimeError(f"MERT returned {len(states)} hidden states, expected 13")
    v=torch.stack(states).mean(0)[0].mean(0).numpy()
   else:
    m,o=obj;import torchaudio;x=rs(wav,48000,16000);n=o.input_len;x=torch.from_numpy(x)[None];x=torch.nn.functional.pad(x,(0,max(0,n-x.shape[1])))[:,:n]
    with torch.inference_mode():v=m(torchaudio.transforms.Spectrogram(n_fft=o.n_fft)(x).unsqueeze(0)).numpy().reshape(-1)
   v=np.asarray(v,dtype=np.float32).reshape(-1);v/=max(float(np.linalg.norm(v)),1e-12)
   if not np.isfinite(v).all():raise RuntimeError("non-finite embedding")
   vv.append(v)
  out[cond]=np.stack(vv);print(cond,len(vv),flush=True)
 path=HERE/"results"/f"{a.model}.npz";np.savez_compressed(path,**out,ids=cache["ids"],artists=cache["artists"],checkpoint=np.array(str(W[a.model])));print(path)
if __name__=="__main__":main()
