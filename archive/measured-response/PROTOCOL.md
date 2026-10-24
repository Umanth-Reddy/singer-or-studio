# TEMP robustness tests — protocol (frozen 2026-10-24, before any new render or embedding)

Workspace rule: everything for these tests lives in `TEMP/`. Files outside TEMP (datasets,
data/f1, data/f2, data/g1 manifests and crops, pilots/*.py, model caches) are only read.
Results are reported whatever they show; a failed rule narrows a claim, it is not hidden.

## T1 — production diversity (real rooms, more accompaniments)
Question: does the F2a/G1 production-mismatch effect survive when the single synthetic room
(noise RIR, RT60 1 s, seed 0) and the single MUSDB accompaniment ("Al James - Schoolboy
Facination") are replaced by several real rooms and accompaniments?

Data: the same clean 4 s windows as F2a (VocalSet, `data/f1/manifest.json`, 20 singers) and
G1 (GTSinger, `data/g1/sources.json`, 19 singers). Same peak normalisation as f2_render.

Reverb: 8 measured RIRs from the MIT IR Survey (Traer & McDermott 2016, 270 everyday spaces,
`TEMP/data/rirs_mit`). Selection fixed before rendering: estimate broadband T30 (Schroeder
backward integration, -5 to -35 dB, extrapolated to 60 dB) for every RIR; sort; take the RIR
closest to each of the 8 quantiles (1/16, 3/16, ..., 15/16). The RIR is resampled to 44.1 kHz,
energy-normalised and applied with the identical dry/wet mix as f1_render.reverb at wet
0.1, 0.2, 0.3, 0.5 -> 32 variants.

Mix-and-separate: 5 MUSDB18 *test* tracks other than Al James (random.Random(0).sample of the
sorted list; C1 training used MUSDB train tracks, so these are disjoint), 60 s from 30 s
(load_accomp), identical f1_render.make_mixsep at VAR +12, +6, 0, -6 dB -> 20 variants.

Encoders: CVSM-ART, CVSM-A, SSL, ECAPA, MERT, embedded with the unchanged pilot code.

Trial and statistic: identical to f2_analyze (clean query; positive = same singer, other song,
processed; negative = other same-sex singer, same song j, clean). Family effect = per-singer
mean error over all variants of the family (all strengths x all rooms / accompaniments; every
reverb and mixsep strength is in the fixed set S of E1) minus clean error; singer-macro;
2000x singer bootstrap 95 % CI.

PRIMARY rule, per family (reverb, mixsep): the effect HOLDS if on BOTH datasets at least 2 of
the 3 singer-specialised encoders (CVSM-ART, CVSM-A, SSL) have effect >= 10 pts with CI > 0.
Secondary (reported, no gate):
- per room / per accompaniment effects, and how many individually pass (>= 10, CI > 0);
- ratio of the new family effect to the original synthetic-recipe effect;
- both-processed control (negative gets the same room/accompaniment): error <= half the cross error;
- specialised-encoder mean effect > ECAPA's and MERT's.
Reading: HOLDS -> robustness evidence for the paper. FAILS for a family -> that family's claim is
reported as recipe-dependent. Any later explanation (e.g. effect vs room T30) is labelled
exploratory.

## T3 — second speaker-verification baseline + fusion
WavLM-Base-Plus-SV (`microsoft/wavlm-base-plus-sv`, x-vector head, 16 kHz, whole crop) on F2a,
G1 and T1 crops. Judged exactly as E1 (effect over S, both datasets) and on T1 families.
No layer or pooling choices after seeing results.
Score fusion (no tuned weights): similarity = mean of the cosine similarities of CVSM-ART and
ECAPA. Report clean error and effect over S on both datasets. Called a useful baseline if, on
both datasets, its effect <= 0.5 x CVSM-ART's and its clean error <= CVSM-ART's clean + 2 pts.

## T2 — listener analysis (waits for the offline H1 responses)
Score per H1 rule (docs/04 "H1"). Additionally: per-condition human accuracy with 95 % CIs from a
crossed bootstrap (resample listeners and trials) and a logistic mixed model
(correct ~ condition + (1|listener) + (1|trial)), unsure = 0.5 (fractional response; sensitivity
analysis with unsure dropped). Models on the same trials reported next to humans.

Amendment (2026-10-24 ~16:06, before any T1 embedding): VocalSet is embedded and analysed as
soon as its render finishes ("interim look", TEMP/t1_f2_results.json). It changes nothing:
the T1 verdict is still judged only on both datasets, with the rule above.
