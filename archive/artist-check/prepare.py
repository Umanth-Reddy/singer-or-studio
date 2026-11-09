#!/usr/bin/env python3
"""Create a fresh artist-balanced Music4All sample and temporary 48 kHz audio cache."""
import csv,json,random,subprocess
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/"music4all"; ARCHIVE=DATA/"audios.zip"; OUT=Path(__file__).resolve().parent
CACHE=Path("/tmp/glass-breaker-cross-check-inputs.npz")
SEED,N_ARTISTS,TRACKS_PER_ARTIST,SR=20261109,12,6,48000
N=4*SR

def run(args,data=None):
 p=subprocess.run(args,input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if p.returncode: raise RuntimeError(f"{args[0]} failed: {p.stderr[-500:].decode(errors='replace')}")
 return p.stdout
def decode(data):
 return np.frombuffer(run(["ffmpeg","-v","error","-i","pipe:0","-ac","1","-ar",str(SR),"-f","f32le","pipe:1"],data),dtype="<f4").copy()
def main():
 members=run(["zipinfo","-1",str(ARCHIVE)]).decode(errors="replace").splitlines()
 ids={Path(m).stem for m in members if m.startswith("audios/") and m.endswith(".mp3")}
 tracks=defaultdict(list)
 with (DATA/"id_information.csv").open(newline="",encoding="utf-8-sig") as f:
  for row in csv.DictReader(f, delimiter="\t"):
   ident,artist=row.get("id","").strip(),row.get("artist","").strip()
   if ident in ids and artist: tracks[artist].append(ident)
 eligible=sorted(a for a,s in tracks.items() if len(set(s))>=TRACKS_PER_ARTIST)
 rng=random.Random(SEED); artists=rng.sample(eligible,N_ARTISTS); rows=[]
 for artist in artists:
  rows += [{"id":i,"artist":artist} for i in rng.sample(sorted(set(tracks[artist])),TRACKS_PER_ARTIST)]
 native=[]; low=[]; details=[]
 with (OUT/"manifest.csv").open("w",newline="",encoding="utf-8") as f:
  writer=csv.DictWriter(f,fieldnames=["id","artist","start_sample","source_bitrate"]);writer.writeheader()
  for idx,row in enumerate(rows):
   mp3=run(["7z","x","-so",str(ARCHIVE),f"audios/{row['id']}.mp3"])
   meta=subprocess.run(["ffprobe","-v","error","-select_streams","a:0","-show_entries","stream=bit_rate,sample_rate,channels","-of","json","pipe:0"],input=mp3,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   try: info=json.loads(meta.stdout)["streams"][0]
   except Exception: info={}
   wave=decode(mp3)
   if len(wave)<N: raise RuntimeError(f"short audio {row['id']}")
   maxstart=max(0,min(len(wave)-N,60*SR)); starts=list(range(0,maxstart+1,2*SR)) or [0]
   start=max(starts,key=lambda x:float(np.mean(wave[x:x+N]**2)))
   context_start=max(0,start-SR//2); context_end=min(len(wave),start+N+SR//2)
   context=wave[context_start:context_end]
   encoded=run(["ffmpeg","-v","error","-f","f32le","-ar",str(SR),"-ac","1","-i","pipe:0","-codec:a","libmp3lame","-b:a","32k","-f","mp3","pipe:1"],context.tobytes())
   dec32=decode(encoded); offset=start-context_start; crop32=dec32[offset:offset+N]
   if len(crop32)<N: crop32=np.pad(crop32,(0,N-len(crop32)))
   native.append(wave[start:start+N].astype(np.float32));low.append(crop32.astype(np.float32))
   row["start_sample"]=start;row["source_bitrate"]=int(info.get("bit_rate") or 0);writer.writerow(row)
   details.append(info)
   if (idx+1)%12==0: print(f"prepared {idx+1}/{len(rows)}",flush=True)
 np.savez_compressed(CACHE,native=np.stack(native),mp3_32=np.stack(low),ids=np.array([r["id"] for r in rows]),artists=np.array([r["artist"] for r in rows]))
 summary={"seed":SEED,"artists":len(artists),"clips":len(rows),"eligible_artist_count":len(eligible),"tracks_per_artist":TRACKS_PER_ARTIST,"source_bitrate_histogram":dict(Counter(str(x.get("bit_rate") or "unknown") for x in details)),"source_sample_rates":dict(Counter(str(x.get("sample_rate") or "unknown") for x in details)),"cache":str(CACHE),"cache_bytes":CACHE.stat().st_size}
 (OUT/"dataset_summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
