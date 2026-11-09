# Singer benchmark cross-check: fresh inference and independent rescoring

**Run date:** 2026-11-09. **Status:** completed on CPU.

## What was checked

The VocalSet and GTSinger manifests, rendered crop audio, and saved embeddings live in `/legacy/ps`. I read those inputs in place and did not write into or copy either raw dataset. I created temporary manifests that select the clean clip and five representative production variants, re-embedded each selected crop with the frozen local model checkpoints, and independently recomputed the pairwise ranking errors. The stored embeddings were also independently rescored across all their available conditions, including the original failing-condition set and C1 arms.

This is a **fresh model-inference and scoring check on the existing rendered crops**; it is not a fresh rendering of every processing condition from original raw audio. That distinction matters for EnCodec and source separation, which can be nondeterministic. Each selected crop was embedded (354 VocalSet and 456 GTSinger rows per model, with zero skipped). No training was done in the fresh benchmark pass.

## Data and metric

- VocalSet: 20 singers, 59 singer-song entries (3 excerpts each in the selected subset).
- GTSinger: 19 singers, 76 singer-song entries (4 songs each).
- Models: CVSM-ART, CVSM-A, Torres singer-identity SSL, ECAPA-TDNN, and MERT-v1-95M.
- Conditions: clean; synthetic reverb wet .3; EnCodec 6 kbps; MP3 64 kbps; mix-and-separate at 0 dB; EQ ±12 dB.
- Error is pairwise ranking error: clean query, same-singer other-song candidate under the named condition, clean different-singer same-sex negative; error when negative cosine is at least the positive cosine. Scores are singer-macro-averaged. These are **not EERs**.
- The matched-processing control applies the same production variant to query, positive, and negative.
- Uncertainty intervals in JSON use 2,000 singer bootstrap samples. The Music4All check separately bootstraps artists.

## Freshly recomputed errors

Percent ranking error. Each condition column is the processed-positive/clean-query error; lower is better. The first number after the model is clean baseline.

### VocalSet (20 singers)

| Model | Clean | Reverb .3 | EnCodec 6 | MP3 64 | Mix + separate 0 dB | EQ 12 |
|---|---:|---:|---:|---:|---:|---:|
| CVSM-ART | 8.1% | 45.0% | 74.2% | 23.4% | 92.4% | 16.8% |
| CVSM-A | 10.6% | 63.2% | 71.6% | 12.7% | 26.3% | 18.7% |
| Torres SSL | 9.5% | 100.0% | 99.2% | 59.0% | 83.5% | 28.1% |
| ECAPA-TDNN | 16.6% | 20.9% | 26.4% | 17.5% | 26.7% | 17.5% |
| MERT-v1-95M | 15.1% | 27.5% | 29.0% | 15.4% | 20.0% | 23.6% |

### GTSinger (19 singers)

| Model | Clean | Reverb .3 | EnCodec 6 | MP3 64 | Mix + separate 0 dB | EQ 12 |
|---|---:|---:|---:|---:|---:|---:|
| CVSM-ART | 8.9% | 39.4% | 33.6% | 20.4% | 83.1% | 16.2% |
| CVSM-A | 7.0% | 39.3% | 25.5% | 9.3% | 17.8% | 13.4% |
| Torres SSL | 7.8% | 95.2% | 67.2% | 26.6% | 29.8% | 13.2% |
| ECAPA-TDNN | 6.4% | 9.7% | 8.8% | 7.1% | 12.4% | 7.4% |
| MERT-v1-95M | 15.3% | 22.2% | 20.3% | 15.2% | 20.1% | 19.7% |

### Fresh vs saved embeddings

For every model and each of the five selected conditions on both datasets, the fresh inference produced the same reported error percentage as rescoring the prior stored embeddings (maximum absolute difference: **0.0 percentage points at the stored precision**). The full Singer-level result file contains exact values, bootstrap intervals, and matched-processing control rates. This strongly confirms the embedding/scoring pipeline for these checked conditions.

### Matched-processing control: specialized models

When the query and both candidates share the processing, errors return near clean baselines for CVSM-ART, CVSM-A, and SSL. Examples on VocalSet: CVSM-ART 92.4% cross-processed mix-and-separate vs 12.8% with all sides similarly processed (clean 8.1%); SSL 100.0% cross-reverb vs 16.2% both-processed (clean 9.5%). On GTSinger, CVSM-ART mix-and-separate is 83.1% cross-processed vs 8.8% both-processed (clean 8.9%). This supports a mismatch effect for those encoder/condition pairs. The same control is not uniformly successful for the general-purpose baselines, so it should not be claimed as a universal property.

## Recheck of the stored full-condition results

An independent scorer (separate from the original pilot scorer) recomputed F2 and G1 from the saved embeddings and manifests. The selected VocalSet conditions match the archived results exactly. On the original fixed G1 failing-condition set, the replicated mean effects were CVSM-ART **+37.2 pp** (95% singer-bootstrap CI 32.9 to 41.2), CVSM-A **+25.8 pp** (22.0 to 29.9), and SSL **+44.5 pp** (36.8 to 53.0). ECAPA was +4.4 pp (2.8 to 6.5) and MERT +5.2 pp (3.6 to 7.0) on the pre-specified union of failing conditions. The family effect is therefore substantial for these specialized models on the controlled benchmark, with the small singer count limiting population generalization.

### C1 causal fine-tuning check

Independent rescoring of saved C1 arm embeddings reproduces the partial result: production-varied positives (B) reduce the effect vs matched positives (A) by about **7.0 pp on VocalSet** and **4.4 pp on GTSinger**, with clean error nearly unchanged. The preregistered requirement was at least a 50% reduction on both datasets; it was not met. Treat this as evidence for a partial causal contribution, not a successful robust method.

## Music4All cross-check re-run

The Music4All artist-level audit remains separate because Music4All labels artists, contains full mixes, and CVSM pretraining overlaps this corpus. I re-ran the metric analysis and adapter summary from the current cross-check embeddings. The recomputed raw model metrics and artist-bootstrap intervals match the existing report exactly (0.0 numerical delta across 114 comparisons). It checks reproducibility of the Music4All analysis; it does not turn this into a singer-verified test.

Its key result remains mixed: native-to-32-kbps mismatch rises by +35.3 pp for CVSM-ART, +9.5 for CVSM-A, +14.8 for Torres SSL, +8.7 for ECAPA, +24.5 for MERT, +1.5 for WavLM, and +0.4 for Yakura (the Yakura interval includes zero). Thus codec mismatch is not exclusive to singer models and the effect is not universal. The six-artist held-out adapter check is promising but too small and potentially exposed to pretraining overlap to validate a method claim.

## Material limits and corrections to the paper claim

1. The current controlled singer evaluation is 39 singers total across two datasets, not a large population sample. Additional singers would improve precision and external validity; they would not automatically prove the causal story or make the effect universal.
2. The fresh pass re-embedded selected **existing rendered crops**. The raw dataset audio was not copied or altered, but this pass did not regenerate every codec, separation, reverb, or EQ waveform from raw recordings.
3. Real-song checks in the earlier independent study narrowed the story: among 147 MUSDB songs, only CVSM-ART showed a clear separation effect (+16.8 pp); CVSM-A, SSL, ECAPA, and MERT were near zero. Real-room tests similarly supported reverb failure mainly for SSL, not all singer encoders. These checks are not duplicated in the current fresh pass.
4. This evidence supports model- and condition-specific production mismatch on controlled singer tasks. It does **not** support “all singer encoders fail,” “real-world separation broadly breaks singer recognition,” or a strong universal method solution.
5. The local H1 package still describes a planned study. The two local `responses.jsonl` exports each contain one 14-answer record, which does not match the 42-trial H1 package. You have said the separate human study is completed and favorable; I have not re-scored or independently verified that result from the available files. Keep human-accuracy numbers out of this audit until the completed study's response export, exclusion rules, and scoring are attached here.
6. The latest source status files are stale (Oct 23/24); this report is the current audit result.

## Updated assessment

My corrected view: **the problem/benchmark evidence is a credible Q2 submission candidate**, and the fresh inference plus independent rescoring strengthen confidence that the key controlled results are not a stale-table or scoring artifact. It is not yet a submission-ready or guaranteed-acceptance paper: the claim must reflect the narrower real-song results, the singer sample remains small, human results need to be auditable in this package, and the method experiments do not establish a generally superior fix. I would frame it as a carefully controlled, model-specific robustness study with a partial mitigation result—not as a universal failure or a solved method. A larger singer-verified benchmark is the most valuable next evidence, but it could weaken or overturn the broad claim as well as strengthen it. Q2 status depends on the target journal, year, and review; no folder can certify acceptance.

## Reproduction artifacts

- `fresh_embeddings/f2/` and `fresh_embeddings/g1/`: fresh vectors from the six selected conditions.
- `manifests/`: subset manifests with paths pointing to the original crops in `/legacy/ps`; no dataset audio is bundled.
- `fresh_scores.json`: fresh vs saved comparison by model, dataset, condition, intervals, and matched-processing control.
- `full_saved_embedding_rescore.json`: all-condition stored embedding rescore for F2/G1 and C1 arms.
- `scripts/`: independent scoring and full saved-result reanalysis scripts. Re-run them from this machine with NumPy: `python scripts/score_fresh_embeddings.py` and `python scripts/recompute_saved_results.py`. Model embedding runner was `/legacy/ps/claude-research/claude-check-oct-26/embed.py`; it was read-only and not modified.
- `../music4all_rerun/`: re-run metric outputs and scripts from the separate artist-level Music4All audit.
