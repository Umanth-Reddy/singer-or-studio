# TEMP results (rules: PROTOCOL.md)

## T1 interim look — VocalSet only (2026-10-24 16:24, TEMP/t1_f2_results.json)
GTSinger not run yet (stopped by user to spare the GPU), so no official T1 verdict.
Effect = family error over all strengths x rooms/accompaniments minus clean error (pts, singer CI).

| Encoder | Reverb, 8 real rooms (orig synthetic) | Mix-sep, 5 new accomps (orig) |
|---|---|---|
| CVSM-ART | +5.2 [3.8, 6.6] (+32.2) | **+55.9** [49.7, 62.8] (+69.8) |
| CVSM-A | +7.7 [6.3, 9.1] (+39.0) | +6.3 [4.4, 8.5] (+16.5) |
| SSL | **+42.7** [40.2, 45.3] (+88.7) | **+42.9** [37.6, 48.1] (+64.0) |
| ECAPA | +1.6 (+4.9) | +4.6 (+9.2) |
| MERT | +2.6 (+13.2) | +2.6 (+5.3) |

Rule (>= 2 of 3 specialised encoders >= 10 pts, CI > 0, on both datasets):
- Reverb: 1/3 on VocalSet -> **FAILS** regardless of GTSinger.
- Mix-and-separate: 2/3 on VocalSet -> passes here; verdict needs GTSinger.

Per-room reverb effect vs T30 (s) [descriptive, exploratory]:
- SSL: 0.06 +14, 0.19 +2, 0.31 +34, 0.39 +46, 0.51 +34, 0.63 +54, 0.92 +79, 1.64 +78.
- CVSM-ART: +11, +0, +2, +3, +1, +3, +11, +10. CVSM-A: +9, -1, +4, +6, +2, +4, +16, +21.
Reading: SSL's reverb failure generalises to real rooms and grows with reverberation time.
CVSM's large reverb failure was mostly specific to the synthetic noise reverb; with real rooms it
appears only for long reverberation (T30 ~0.9-1.6 s, +10 to +21). Mix-and-separate failure of
CVSM-ART and SSL generalises across accompaniments (every accompaniment passes); CVSM-A's is small.
Both-processed controls still recover most of the error. ECAPA/MERT remain least affected.
Note: the H1 listening trials use the synthetic reverb@0.3; the human claim for reverb is about
that recipe.

## T3 — WavLM-SV + fusion (original crops, TEMP/t3_results.json)
WavLM-SV: clean 31.0 / 23.5 %, effect over S +8.6 / +7.2 (VocalSet / GTSinger).
Fusion CVSM-ART+ECAPA: clean 11.8 / 5.6 %, effect +21.5 / +9.2 (CVSM-ART +47.8 / +31.2).
Protocol rule: FAILS (VocalSet clean 11.8 > 8.1 + 2). Strong baseline with a clean cost on VocalSet.
WavLM on T1 renders: VocalSet in TEMP/data/f2/emb_wavlm.npz (analysed with the full T3 run).

## T1 FINAL — both datasets (2026-10-24 17:06, TEMP/t1_results.json)
| Encoder | Reverb VS / GT (orig) | Mix-sep VS / GT (orig) |
|---|---|---|
| CVSM-ART | +5.2 / +4.6 (+32 / +26) | **+55.9 / +46.5** (+70 / +57) |
| CVSM-A | +7.7 / +6.7 (+39 / +28) | +6.3 / +4.3 (+17 / +11) |
| SSL | **+42.7 / +27.4** (+89 / +75) | **+42.9 / +18.3** [9.0, 28.6] (+64 / +25) |
| ECAPA | +1.6 / +0.9 | +4.6 / +4.1 |
| MERT | +2.6 / +2.6 | +2.6 / +1.7 |
| WavLM-SV | +4.8 / +2.4 | +6.9 / +3.7 |

VERDICT (pre-registered rule):
- **Reverb: FAILS** (1/3 specialised encoders on each dataset; only SSL).
- **Mix-and-separate: HOLDS** (CVSM-ART and SSL on both datasets; every accompaniment passes for both).
- Specialised mean > ECAPA and MERT for both families on both datasets.
GTSinger per room, SSL: 0.06 +9, 0.19 +1, 0.31 +15, 0.39 +23, 0.51 +18, 0.63 +29, 0.92 +60, 1.64 +64;
CVSM-ART <= +9 in every room; CVSM-A +12 / +16 only in the two longest rooms (exploratory).
