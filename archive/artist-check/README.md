# Independent cross-check on Music4All

This folder contains a fresh, reproducible check of whether production mismatch changes artist-embedding rankings, using released encoders and a sample selected directly from the Music4All archive. It does not consume earlier result JSON or review documents.

## Result

The 32 kbps condition means a crop decoded from the Music4All MP3 and re-encoded at 32 kbps. The source audio is already compressed; it is not a clean master. “Native” below means the as-distributed archive crop.

| Encoder | Native ranking error | Native query / 32 kbps positive | Both sides 32 kbps | Change vs native (artist-bootstrap 95% interval) |
|---|---:|---:|---:|---:|
| CVSM-ART | 10.27% | 45.53% | 15.14% | +35.26 pp [+26.23, +43.28] |
| CVSM-A | 12.64% | 22.15% | 17.23% | +9.50 pp [+7.53, +11.24] |
| Torres contrastive SSL | 25.69% | 40.52% | 28.68% | +14.84 pp [+8.57, +21.89] |
| ECAPA-TDNN | 19.14% | 27.83% | 21.78% | +8.69 pp [+5.96, +11.16] |
| MERT | 20.34% | 44.79% | 20.32% | +24.46 pp [+18.28, +30.37] |
| WavLM-Base-Plus-SV | 26.79% | 28.24% | 26.69% | +1.45 pp [+0.17, +2.66] |
| Yakura et al. | 47.29% | 47.64% | 44.94% | +0.35 pp [-5.25, +6.01] |

The change is large for CVSM-ART and MERT, moderate for CVSM-A, Torres SSL, and ECAPA, small for WavLM, and indistinguishable from zero for Yakura in this sample. Applying the same 32 kbps encoding to the query and candidates brings most models closer to their native ranking error, especially MERT and WavLM; it does not fully restore the native score for every model.

**Assessment:** this is evidence that codec mismatch can cause real ranking failures for several tested music and voice encoders. It does not support a universal “all singer models fail” claim. It also does not establish a singer-only result: Music4All supplies artist labels and full mixes, and several sampled artists are bands. Treat this as an independent artist-level codec stress test, not a confirmatory singer biometrics result or an EER estimate.

## Protocol

- A fixed seed (20261109) selects 12 artists with at least six available tracks, then six tracks per artist (72 tracks total) from the full archive. See manifest.csv; selection was made before encoder inference.
- For each song, decode the archive MP3 to mono 48 kHz. In the first 60 seconds, choose the highest-energy 4-second crop on a 2-second grid. The original file is untouched; encode the same crop plus 0.5-second context on each side to MP3 32 kbps, decode it, and crop the corresponding 4 seconds.
- CVSM uses the released author checkpoints and 1-second log-mel windows, with unit-norm window inputs and averaged encoder outputs. The other runners use the local project preprocessing for each checkpoint (sample rate and pooling documented in embed_torch.py).
- For every query track, positives are all other tracks by the same metadata artist; negatives are all tracks by other artists. Error is the fraction of positive-negative comparisons where negative cosine is at least positive cosine. Average across queries within each artist, then macro-average artists.
- “Native” compares all as-distributed crops. “Native query / 32 kbps positive” compares an unmodified query and negative against positives re-encoded at 32 kbps. “Both sides 32 kbps” applies the same 32 kbps encoding to queries, positives, and negatives.
- The interval resamples 12 artist-level error changes 10,000 times. Trials within an artist share tracks, so the thousands of pair comparisons are not independent.

## Adapter follow-up

fit_adapter.py evaluates the previously explored 64-rank residual metric with fixed strengths (ART/A 1000, SSL 1), 400 steps, and three initializations (seeds 0, 1, 2). It trains on the first six artist names in sorted order and tests on the other six. No hyperparameter search uses the held-out half. Across seeds, the held-out 32 kbps mismatch errors were: CVSM-ART raw 50.17%, adapted 5.96–17.02% (40.25-point average reduction); CVSM-A raw 23.09%, adapted 9.09–10.20% (13.33-point average reduction); Torres SSL raw 38.98%, adapted 22.89–23.70% (15.76-point average reduction). Native-condition error changes were small: +0.24 to +0.74 points for ART, +1.61 to +1.72 for A, and +1.22 to +1.63 for SSL.

| Encoder | Held-out raw native / mismatch / both | Adapter mismatch across seeds | Mean mismatch change |
|---|---:|---:|---:|
| CVSM-ART | 12.89 / 50.17 / 17.59% | 5.96–17.02% | −40.25 pp |
| CVSM-A | 13.81 / 23.09 / 20.46% | 9.09–10.20% | −13.33 pp |
| Torres SSL | 25.06 / 38.98 / 31.43% | 22.89–23.70% | −15.76 pp |

For all nine seed/model runs, the six-artist bootstrap interval for the mismatch change stayed below zero. The three seeds repeat initialization on the same six held-out artists; they are not independent samples, and the intervals remain uncertain with only six held-out artists. Seed-specific intervals are in results/adapter_comparison.json; results/metrics.json has the full per-artist outputs. Trained adapter weights are saved in results/adapter_*.pt. This is a promising transfer check, not confirmation: encoder pretraining can overlap Music4All, negatives are not sex-matched, artist labels do not identify a singer in bands, and the held-out set is just six artists.

## Source models and files

- The tested CVSM source modules and checkpoints are copied into models/cvsm/repo from the sibling clone at commit ffbb3ae0799b42d83e202db7f9ac04b8c6bdbdcf. Only ART and A checkpoints are included.
- The tested checkpoint weights and source material are included in models/; see MODEL_PROVENANCE.md, models.sha256, and ENVIRONMENT.md.
- Music4All audio is read in place from ../music4all. No dataset audio is copied into this folder. The temporary decoded crops live at /tmp/glass-breaker-cross-check-inputs.npz during a run and are removed when run_all.sh finishes.
- Existing trained weights are tested frozen. The adapter follow-up is the only training step; it uses its own six-artist training half.

## Reproduce

From this machine run:

    bash "/legacy/credible-work-bro/cross checking/run_all.sh"

The script uses /legacy/ps/.venv-ssl for NumPy analysis, /legacy/ps/.venv-gate1 for TensorFlow CVSM, and /legacy/ps/claude-research/.venv for PyTorch / Transformers / SpeechBrain. It needs 7z, zipinfo, ffmpeg, and ffprobe. It runs on CPU; no GPU was visible. Model outputs and logs are kept in results/; the audio cache is removed after a successful run.

## Limits

The sample is modest, labels are artist-level, there is no vocal separation, and the archive audio is already MP3 (71 of 72 sampled streams report 128 kbps; one reports 64 kbps). CVSM checkpoints were trained on Music4All, so their pretraining exposure overlaps this test corpus; exact track overlap was not audited. Exposure for other checkpoints was not exhaustively audited either. One crop per song is not a deployment evaluation. This cannot establish population-level singer performance, causality of every studio-production factor, or paper acceptance. The result does support moving to a singer-verified, singer-disjoint, clean-master/controlled-codec evaluation before making the paper claim.


## Singer-level datasets added to this cross-check (2026-11-09)

I added an independent fresh-inference and rescoring pass on the VocalSet and GTSinger datasets already stored under `/legacy/ps`. Raw datasets and source folders were read in place and left unchanged; no dataset audio is copied here. The pass uses selected rendered crops from the original manifests, re-embeds all five local frozen models, and independently rescored the same singer-level pairwise task. All selected fresh scores match the prior stored embeddings at reported precision.

See [`singer_recheck/REPORT.md`](singer_recheck/REPORT.md) for protocol, results, limits, and corrected assessment. Fresh vectors, subset manifests with original source paths, full scorer outputs, and scripts are in that folder. `music4all_rerun/` contains the repeated Music4All metric analysis. The Music4All check stays artist-level and is not pooled with singer-level results.


## Method reruns (2026-11-09)

I also reran the M2 seed check, clean-reference ridge adapter, nuisance attenuation, and clean-similarity distillation from scratch copies using the saved VocalSet/GTSinger embeddings in `/legacy/ps`. Each output matches its earlier result file to floating-point precision (M2 and distillation exact). The method rerun report is [`method_recheck/REPORT.md`](method_recheck/REPORT.md). These reuse the same test singers, so they check reproducibility, not independent method validation.


## Daily progress log

See [`PROGRESS_2026-11-09.md`](PROGRESS_2026-11-09.md) for today’s rechecks, human-study decision, and prioritized external-validation plan.
