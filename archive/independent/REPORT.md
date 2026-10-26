# Independent checks — 2026-10-26

New code only (`render.py`, `embed.py`, `analyze.py`); original pilot code imported, never edited.
Trial scoring re-implemented independently in `analyze.py`. Raw numbers: `results_<check>.json`.

## 1. Full rerun from source (VocalSet, 6 key conditions, 5 encoders) — REPRODUCES
Clean, reverb@0.3, mp3@64, eq@12 renders are bit-identical to the stored crops; codec@6 and
mixsep@0 differ at waveform level (neural codec / separator nondeterminism). Every error rate is
within 1 pt of `pilots/f2_results.json` (e.g. CVSM-ART mixsep@0 92.4 -> 92.8, SSL codec@6 99.2 -> 99.2).

## 2. Loudness — RULED OUT
RMS-matching every condition changes no effect (0.0 pts, all 5 encoders); pure gain of -12/-6/+6 dB
on clean gives 0.0 pts. By construction CVSM L2-normalises each 1 s window, SSL peak-normalises,
MERT standardises the input.

## 3b. Real songs (147 MUSDB songs: htdemucs vocals from the real mix vs ground-truth stem) — NARROWS CLAIM
Effect (pts, song-bootstrap CI): CVSM-ART +16.8 [14.2, 19.8]; SSL +2.6 [1.7, 3.7]; CVSM-A +0.9;
ECAPA +0.7; MERT +1.0. Easier trial than F2a (positive from the same song), so sizes are not
comparable to F2a, but only CVSM-ART shows a clear separation failure on real music.
3a (10 new synthetic accompaniments) was stopped by the user to save time; T1 already covered 5.

## 4. Speech models — HOLDS (with a correction)
Effect over S, paired singer bootstrap vs ECAPA: CVSM-ART +40.8 / +26.8, CVSM-A +25.2 / +13.1,
SSL +63.8 / +40.1 pts more than ECAPA (VocalSet / GTSinger, all CIs > 0).
Clean error vs ECAPA: VocalSet singer models 6-9 pts better (CIs borderline); GTSinger no
significant difference (CVSM-ART +2.5 [-0.3, 5.4]). Correction: ECAPA is "about as accurate" on
GTSinger, not "more accurate".
Second speech model, x-vector: also robust (effect +5.9 / +2.8) but less accurate clean (21.8 / 14.5 %).

## 5. Training-data overlap — NONE FOUND
CVSM: Music4All. SSL (Torres et al. 2023): VocalSet listed as an out-of-domain test set.
GTSinger (2024) postdates both. ECAPA/x-vector: VoxCeleb speech.

## What survives
- Codec/compression failure: CVSM and SSL (synthetic EnCodec/MP3 renders).
- Separation failure: CVSM-ART strongly, including on real songs; SSL only in synthetic mixes.
- Reverb failure: SSL only (T1, real rooms).
- Speech-verification models: robust, and about as accurate on clean singing as singer models.
Docs (`docs/06_paper_plan.md`, `docs/05_status.md`) predate T1 and these checks and overstate the claim.
