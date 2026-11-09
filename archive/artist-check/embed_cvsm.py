#!/usr/bin/env python3
"""Run released CVSM ART/A encoders on native and 32 kbps crops."""
import argparse,sys
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
HERE=Path(__file__).resolve().parent; REPO=HERE/"models/cvsm/repo"
CACHE=Path("/tmp/glass-breaker-cross-check-inputs.npz")
MODELS={"cvsm_art":REPO/"models/htdemucs/m4all_cvsmart","cvsm_a":REPO/"models/open-unmix/m4all_cvsma"}
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",choices=MODELS,required=True);a=p.parse_args()
 import tensorflow as tf
 tf.config.set_visible_devices([], "GPU");tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
 sys.path.insert(0,str(REPO));from cola import constants;from mscol import network
 model=network.get_contrastive_network(embedding_dim=512,temperature=.2,pooling_type="max",similarity_type=constants.SimilarityMeasure.BILINEAR)
 ckpt=tf.train.latest_checkpoint(str(MODELS[a.model]))
 if not ckpt: raise FileNotFoundError(MODELS[a.model])
 status=model.load_weights(ckpt);status.assert_existing_objects_matched();status.expect_partial()
 encoder=model.embedding_model.get_layer("encoder");cache=np.load(CACHE);out={}
 for cond in ("native","mp3_32"):
  vectors=[]
  for wav in cache[cond]:
   x=resample_poly(wav,1,3).astype(np.float32).reshape(4,16000);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
   spec=tf.abs(tf.signal.stft(x,400,160,fft_length=1024));mel=tf.tensordot(spec,tf.signal.linear_to_mel_weight_matrix(64,513,16000,60,7800),1)
   feat=tf.expand_dims(tf.math.log(tf.clip_by_value(mel,1e-5,1e8)),-1);v=encoder(feat,training=False).numpy().mean(0);v/=max(float(np.linalg.norm(v)),1e-12);vectors.append(v.astype(np.float32))
  out[cond]=np.stack(vectors);print(cond,len(vectors),flush=True)
 path=HERE/"results"/f"{a.model}.npz";np.savez_compressed(path,**out,ids=cache["ids"],artists=cache["artists"],checkpoint=np.array(str(ckpt)));print(path)
if __name__=="__main__":main()
