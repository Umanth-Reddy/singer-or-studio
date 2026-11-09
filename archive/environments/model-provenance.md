# Model provenance

All inference outputs were generated from frozen weights. SHA-256 values for copied model weights are in models.sha256.

| Encoder | Upstream model / paper | Local material used |
|---|---|---|
| CVSM-ART, CVSM-A | [CVSM repository](https://github.com/cgaroufis/CVSM), [paper preprint](https://arxiv.org/abs/2510.03025) | Source modules and selected released checkpoints copied into models/cvsm/repo from the local official clone at commit ffbb3ae0799b42d83e202db7f9ac04b8c6bdbdcf: htdemucs/m4all_cvsmart and open-unmix/m4all_cvsma. |
| Torres contrastive SSL | [Singer Identity Representation Learning using Self-Supervised Techniques](https://hal.science/hal-04186048v1) | TorchScript checkpoint in models/singer_identity_ssl/checkpoints/ssl_contrastive_vc/model.ts; authors' source and license in models/singer_identity_ssl/repo/. |
| ECAPA-TDNN | [SpeechBrain ECAPA VoxCeleb model](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | models/ecapa/weights/; inference through installed SpeechBrain. |
| WavLM | [WavLM paper](https://arxiv.org/abs/2110.13900) | models/wavlm/weights/, WavLM-Base-Plus-SV configuration and safetensors checkpoint. |
| MERT | [MERT paper and model](https://arxiv.org/abs/2306.00107) | models/mert/weights/, MERT-v1-95M model and local modeling code. |
| Yakura et al. | [Self-Supervised Contrastive Learning for Singing Voices](https://doi.org/10.1109/TASLP.2022.3169627) | models/yakura/repo/ and models/yakura/checkpoints/ckpt_epoch_100.pth. |

CVSM source modules and all other tested weights/source files are included under models/; checkpoint hashes are recorded. No model was fetched during this test. The sample's model-training overlap is a limitation, especially for CVSM, whose official training data includes Music4All.
