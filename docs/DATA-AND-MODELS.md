# Dataset and checkpoint setup

The offline evidence audit runs without datasets or checkpoints. For waveform and model reruns, put the following inputs in the ignored datasets directory, or pass --datasets to the preparation command. Use the original archives and honor each provider's terms.

| Input | Required layout | Source and exact selection |
| --- | --- | --- |
| VocalSet | datasets/vocalset/female1/excerpts/straight/… | [VocalSet release](https://doi.org/10.5281/zenodo.1193957). Point the dataset folder at the archive's FULL directory. The 59 selected paths and offsets are in [F2 inputs](../evidence/f2-rerun-inputs.json). |
| GTSinger | datasets/gtsinger/Chinese/ZH-Alto-1/… | [GTSinger full-corpus downloads](https://huggingface.co/datasets/GTSinger/GTSinger). Use the language archives named on the dataset card. The viewer sample is incomplete for this benchmark. [G1 inputs](../evidence/g1-rerun-inputs.json) records all 76 paths and offsets. |
| JVS-MuSiC | datasets/jvs/jvs_music_ver1.zip | [Publisher download and terms](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music). Keep the original ZIP, 598,098,044 bytes. The preparer checks the official singer groups and evaluates the 200 raw vocal tracks listed in the [inventory](../archive/jvs-initial/jvs_music_inventory.json). |
| CoVox | datasets/covox/S01_AL_pop_L.wav, etc. | [Publisher repository](https://osf.io/cgexn/) and [dataset paper](https://doi.org/10.3758/s13428-025-02664-9). Put the 44 selected files directly under covox/. |
| NUS-48E | datasets/nus/ADIZ/sing/01.wav, etc. | [Original corpus download folder](https://drive.google.com/drive/folders/12pP9uUl0HTVANU3IPLnumTJiRjPtVUMx) and [dataset paper](https://ieeexplore.ieee.org/abstract/document/6694316). Use the unmodified sung recordings. |
| MIT IR Survey | datasets/mit-ir/h001_Bedroom_65txts.wav, etc. | [MIT IR Survey download](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html). Point the folder at the Audio directory of the archive. The matched pairs and fingerprints are [here](../archive/room-followup/room_pairs.json). The earlier sweep uses the separate [fixed selection](../archive/measured-response/selection.json). |
| MUSDB18 | datasets/musdb18/train/*.stem.mp4 and test/*.stem.mp4 | [MUSDB18 STEMS release](https://doi.org/10.5281/zenodo.1117372). The codec and augmentation comparisons with backing tracks use the original test split masters. The [recorded mixture manifest](../evidence/musdb18-crop-manifest.json) tracks all 150 inputs. |
| Music4All | Original audios.zip and id_information.csv | [Provider request page](https://sites.google.com/view/contact4music4all). The [sample manifest](../archive/artist-check/manifest.csv) identifies the 72 recordings and offsets. This artist-only auxiliary result is available without the corpus. |

CoVox and NUS source SHA-256 values are in the [recording manifest](../evidence/response-crop-manifest.json). The render step checks them. GTSinger and VocalSet use the fixed source paths and start samples. Their recorded source fingerprints are verified before rendering. The experimental crops and audio stay local.

## Frozen encoders

Install the optional fetch dependency, then request the needed models from the repository root.

```sh
python -m pip install huggingface-hub
python workflow/fetch_models.py cvsm_art cvsm_a ssl ecapa mert
```

The fetcher uses the [CVSM official repository](https://github.com/cgaroufis/CVSM) at commit ffbb3ae0799b42d83e202db7f9ac04b8c6bdbdcf, Torres's [contrastive-vc TorchScript release](https://huggingface.co/BernardoTorres/singer-identity/tree/e60d96b/contrastive-vc), [SpeechBrain ECAPA VoxCeleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb), and [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M). It checks all loaded weight files against [the SHA-256 list](checkpoints.sha256). An existing matching checkout or manually fetched snapshot also works at the paths encoded in the hash list. A missing or mismatched weight is an error. MERT loads the modeling code in its pinned provider snapshot.

The fixed processing stages use [EnCodec](https://github.com/facebookresearch/encodec)'s 24 kHz mono model and [HTDemucs](https://github.com/facebookresearch/demucs)'s htdemucs release with their standard framework downloads and caches. The optional speech baseline used [WavLM Base Plus SV](https://huggingface.co/microsoft/wavlm-base-plus-sv) at feb593a6c23c1cc3d9510425c29b0a14d2b07b1e. The auxiliary Music4All run used a locally trained Yakura epoch-100 checkpoint whose SHA-256 is recorded. That checkpoint and C1's fine-tuned CVSM weights are author-local assets. The archived score tables, source scripts and checks describe those pilots. Exact inference for those pilots requires either authorized access to the historical checkpoints or retraining and checking against the archived results.

The repository tracks identifiers and fingerprints for model assets. Raw weights, audio, rendered crops and speaker vectors are kept in ignored paths. Upstream dataset and model terms apply independently.

For optional training replication, the C1 source used K1 FMA Medium shards s00–s02, with only crops at or above the recorded vocal-RMS cutoff and a fixed MUSDB18 augmentation recipe. See the [FMA release](https://github.com/mdeff/fma), the [K1 selection script](../archive/methods/k1_separate.py), the [K1 manifest](../archive/methods/k1-training-manifest.json) and the [C1 render](../archive/methods/c1_render.py). The original processed training cache and fine-tuned checkpoints are large author-local files. M2 uses the [M1 training selection](../archive/methods/m1-training-manifest.json) and the original [adapter fitting script](../archive/methods/m2_train_eval.py). The archives show the results and limitations of these analyses, including the separate Music4All verification run.
