# Singer or Studio? When Production Changes the Match

## Supplementary results and reproducibility

The five core encoders used identical trial membership for every treatment. VocalSet retained 9 female-labelled and 11 male-labelled singers and 59 recordings, giving 116 genuine pairs and 1,032 triplets per condition. GTSinger retained 10 female-labelled and 9 male-labelled singers and 76 recordings, giving 228 pairs and 1,944 triplets. Each manifest records the grouping and crop offset. The accompanying singer-count CSV records the denominator and two error numerators by condition and model.

The journal revision uses a single sorted singer order for bootstrapping shared-condition means. The original shared-set routine iterated singers in manifest insertion order and the original model-specific routine sorted singer IDs. These gave the same mean but different finite-draw percentile estimates with the same seed. The file trial-audit.json preserves both calculations and the common-set selection. The original separate x-vector and WavLM results remain available in the JSON and archive notes.

## S1. JVS-MuSiC factorial and ranking audit

The three digits encode the query, genuine candidate and impostor, in that order. Zero means clean and one means processed. For example, 010 processes only the genuine candidate, 011 processes both candidates, and 111 processes all three recordings. Comparisons retain the same performances and eligible candidates. Each song orientation includes 100 genuine pairs and 1,568 triplets, grouped by query singer.

To calculate a cell, count every trial whose impostor cosine equals or exceeds the genuine cosine. Divide the error count by the eligible impostor count for each query singer. Average those 100 rates and multiply by 100. Calculate paired contrasts by subtracting the two conditions at the trial level before resampling. For the symmetric matching contrast, compare cells 000−001, 010−011, 101−100 and 111−110 with equal weight.

For a reproducible render, the JVS source is 24 kHz mono. The reverb noise seed is 1000 + 2 × the numerical JVS singer identifier + 1 for a singer-specific song (or + 0 for the common song). The recipe uses a nominal one-second decay and 1.2 seconds of response. Process the full recording, center-crop four seconds with symmetric padding if needed, then peak-normalize to 0.5. The MP3 arm uses 64 kb/s mono and truncates or pads decoded audio to source length before cropping.

**Table A1.** Full ranking error (%) by model, query-song orientation, treatment and factorial cell. The eight cells appear in the digit order defined above.

| Treatment | Encoder | Query song | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reverb 20% | CVSM-ART | Common | 14.7 | 3.9 | 46.2 | 17.5 | 14.1 | 28.5 | 6.9 | 11.3 |
| MP3 64 kb/s | CVSM-ART | Common | 14.7 | 11.1 | 22.0 | 16.8 | 13.5 | 15.1 | 13.1 | 13.8 |
| Reverb 20% | CVSM-ART | Singer-specific | 12.6 | 5.2 | 42.2 | 13.0 | 13.2 | 33.9 | 4.0 | 10.0 |
| MP3 64 kb/s | CVSM-ART | Singer-specific | 12.6 | 10.8 | 17.2 | 12.9 | 13.1 | 15.8 | 10.1 | 11.5 |
| Reverb 20% | CVSM-A | Common | 14.9 | 4.7 | 47.9 | 16.7 | 12.6 | 19.4 | 11.2 | 13.7 |
| MP3 64 kb/s | CVSM-A | Common | 14.9 | 14.5 | 16.4 | 16.3 | 14.9 | 16.0 | 14.7 | 15.7 |
| Reverb 20% | CVSM-A | Singer-specific | 14.2 | 6.0 | 37.5 | 12.3 | 13.5 | 24.7 | 6.5 | 11.0 |
| MP3 64 kb/s | CVSM-A | Singer-specific | 14.2 | 13.7 | 14.0 | 13.7 | 14.1 | 14.8 | 12.9 | 13.5 |
| Reverb 20% | Torres SSL | Common | 18.5 | 1.0 | 98.9 | 36.3 | 31.7 | 97.3 | 1.0 | 19.9 |
| MP3 64 kb/s | Torres SSL | Common | 18.5 | 13.7 | 28.0 | 20.9 | 18.3 | 23.3 | 13.1 | 16.1 |
| Reverb 20% | Torres SSL | Singer-specific | 18.6 | 0.2 | 99.5 | 37.7 | 28.3 | 96.4 | 0.9 | 18.1 |
| MP3 64 kb/s | Torres SSL | Singer-specific | 18.6 | 12.9 | 29.5 | 19.2 | 19.0 | 22.0 | 16.5 | 17.0 |

**Table A2.** Audit example for CVSM-ART with the common song as query. Cosines are singer-macro averages. Ranking error includes a 95% interval.

| Cell | Genuine cosine | Impostor cosine | Error (%) and interval |
| --- | --- | --- | --- |
| 000 | 0.931149 | 0.883459 | 14.73 [10.15, 19.36] |
| 010 | 0.888241 | 0.883459 | 46.24 [39.01, 53.60] |
| 011 | 0.888241 | 0.840697 | 17.52 [12.99, 22.12] |
| 100 | 0.895950 | 0.846784 | 14.15 [10.12, 18.29] |
| 111 | 0.926267 | 0.866541 | 11.25 [7.27, 15.36] |

From 010 to 011 the genuine score stays at 0.888241, while the mean impostor score falls from 0.883459 to 0.840697. The displayed error change is 46.24 − 17.52 = 28.72 percentage points. This subtraction uses the rates obtained from the trial indicators. The attached score file allows that calculation to be independently repeated for each model, processing recipe and song orientation.



**Table A3.** Clean JVS cosine margin percentiles, with equal weight per query singer.

| Encoder | Query song | 10th | 50th | 90th |
| --- | --- | --- | --- | --- |
| CVSM-ART | Common | -0.0111 | 0.0443 | 0.1108 |
| CVSM-ART | Singer-specific | -0.0073 | 0.0461 | 0.1095 |
| CVSM-A | Common | -0.0114 | 0.0354 | 0.0883 |
| CVSM-A | Singer-specific | -0.0084 | 0.0355 | 0.0913 |
| Torres SSL | Common | -0.0232 | 0.0511 | 0.1964 |
| Torres SSL | Singer-specific | -0.0269 | 0.0509 | 0.2002 |

---

## S2. Core benchmark condition tables

The query stays clean throughout. Clean error is shown for each encoder. “G/B” gives the percentage errors with only the genuine candidate processed (G) and with both candidates processed (B). The shift is G minus the clean ranking error, in percentage points, with the 95% paired query-singer interval. A shared setting is marked with an asterisk. The first table retains the original model-specific selection and the remaining tables give every condition.


**Table B1.** Clean, genuine-only processed, and both-candidates-processed ranking error, in percent. The query is clean in all three. Shifts are paired changes from clean in percentage points with 95% intervals. For each singer encoder, the same VocalSet-selected condition set is used on both cohorts. ECAPA and MERT use the shared 17-condition union.

| Corpus | Encoder | Conditions | Clean | Genuine only | Both candidates | Shift (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| VocalSet | CVSM-ART | 14 | 8.1 | 65.2 | 18.2 | +57.1 [52.0, 62.5] |
| VocalSet | CVSM-A | 11 | 10.6 | 58.9 | 15.3 | +48.3 [42.9, 53.0] |
| VocalSet | Torres SSL | 17 | 9.5 | 80.2 | 30.7 | +70.8 [65.8, 75.6] |
| VocalSet | ECAPA-TDNN | 17 | 16.6 | 23.6 | 21.6 | +7.0 [4.9, 8.8] |
| VocalSet | MERT | 17 | 15.1 | 25.8 | 17.3 | +10.8 [8.6, 13.0] |
| GTSinger | CVSM-ART | 14 | 8.9 | 46.1 | 14.2 | +37.2 [32.9, 41.2] |
| GTSinger | CVSM-A | 11 | 7.0 | 32.8 | 10.1 | +25.8 [22.0, 29.9] |
| GTSinger | Torres SSL | 17 | 7.8 | 52.3 | 21.8 | +44.5 [36.8, 53.0] |
| GTSinger | ECAPA-TDNN | 17 | 6.4 | 10.8 | 9.5 | +4.4 [2.8, 6.5] |
| GTSinger | MERT | 17 | 15.3 | 20.5 | 16.4 | +5.2 [3.6, 7.0] |

**Table B2.** CVSM-ART. Clean error is 8.07% on VocalSet and 8.89% on GTSinger.

| Condition | VocalSet G/B (%) | VocalSet shift (pp) | GTSinger G/B (%) | GTSinger shift (pp) |
| --- | --- | --- | --- | --- |
| Reverb 0.1 wet * | 11.2 / 8.8 | +3.1 [1.6, 4.6] | 10.9 / 8.9 | +2.1 [1.0, 3.2] |
| Reverb 0.2 wet * | 21.3 / 9.2 | +13.2 [9.3, 17.3] | 21.1 / 10.4 | +12.2 [8.3, 16.5] |
| Reverb 0.3 wet * | 45.0 / 11.1 | +37.0 [29.2, 44.2] | 39.4 / 12.5 | +30.5 [22.1, 39.2] |
| Reverb 0.5 wet * | 83.5 / 15.0 | +75.4 [67.9, 82.1] | 67.8 / 14.9 | +58.9 [49.3, 68.2] |
| EnCodec 1.5 kb/s * | 87.4 / 26.0 | +79.3 [72.3, 85.9] | 51.8 / 19.6 | +42.9 [37.5, 48.6] |
| EnCodec 3 kb/s * | 86.0 / 23.3 | +77.9 [69.7, 85.4] | 45.0 / 15.6 | +36.1 [30.8, 42.6] |
| EnCodec 6 kb/s * | 74.2 / 20.1 | +66.1 [57.0, 75.1] | 33.6 / 14.7 | +24.7 [20.3, 29.3] |
| EnCodec 12 kb/s * | 55.8 / 19.0 | +47.7 [38.0, 57.6] | 23.1 / 13.9 | +14.2 [11.0, 18.0] |
| EnCodec 24 kb/s * | 44.2 / 16.5 | +36.1 [27.2, 46.2] | 17.4 / 12.6 | +8.5 [5.8, 11.8] |
| MP3 32 kb/s * | 80.6 / 21.0 | +72.6 [67.5, 77.1] | 63.0 / 18.8 | +54.1 [42.8, 64.9] |
| MP3 64 kb/s * | 23.4 / 11.4 | +15.4 [11.2, 19.8] | 20.4 / 11.8 | +11.6 [7.7, 15.5] |
| MP3 128 kb/s * | 9.1 / 8.3 | +1.0 [0.0, 2.2] | 9.9 / 9.7 | +1.0 [-0.3, 2.3] |
| EQ 3 dB | 8.4 / 8.1 | +0.3 [-0.5, 1.0] | 9.2 / 8.7 | +0.3 [-0.5, 1.0] |
| EQ 6 dB | 9.0 / 8.0 | +0.9 [-0.4, 2.3] | 10.6 / 9.3 | +1.7 [0.5, 3.3] |
| EQ 12 dB * | 16.8 / 8.9 | +8.7 [5.5, 11.7] | 16.2 / 9.8 | +7.3 [4.2, 10.8] |
| Mix/separate -6 dB * | 98.2 / 26.9 | +90.2 [84.6, 94.7] | 94.5 / 19.5 | +85.6 [78.7, 91.3] |
| Mix/separate 0 dB * | 92.4 / 21.3 | +84.3 [78.1, 89.6] | 83.1 / 14.1 | +74.2 [64.3, 82.3] |
| Mix/separate 6 dB * | 74.9 / 19.3 | +66.8 [59.2, 74.3] | 56.2 / 10.9 | +47.3 [39.1, 55.5] |
| Mix/separate 12 dB * | 45.9 / 14.7 | +37.9 [28.2, 47.5] | 28.3 / 9.6 | +19.4 [15.0, 24.1] |

**Table B3.** CVSM-A. Clean error is 10.60% on VocalSet and 7.00% on GTSinger.

| Condition | VocalSet G/B (%) | VocalSet shift (pp) | GTSinger G/B (%) | GTSinger shift (pp) |
| --- | --- | --- | --- | --- |
| Reverb 0.1 wet * | 13.5 / 9.5 | +2.9 [1.4, 4.6] | 8.1 / 7.0 | +1.1 [0.3, 2.0] |
| Reverb 0.2 wet * | 27.2 / 9.7 | +16.6 [11.4, 21.8] | 18.1 / 8.2 | +11.1 [7.9, 14.6] |
| Reverb 0.3 wet * | 63.2 / 12.6 | +52.6 [43.0, 62.0] | 39.3 / 10.2 | +32.3 [23.7, 41.9] |
| Reverb 0.5 wet * | 94.4 / 14.9 | +83.8 [77.5, 89.3] | 75.2 / 13.3 | +68.2 [59.1, 77.2] |
| EnCodec 1.5 kb/s * | 94.3 / 23.9 | +83.7 [75.6, 90.7] | 56.5 / 11.3 | +49.5 [42.5, 57.3] |
| EnCodec 3 kb/s * | 87.5 / 18.4 | +76.9 [69.3, 83.7] | 39.6 / 10.3 | +32.6 [27.0, 38.4] |
| EnCodec 6 kb/s * | 71.6 / 17.8 | +61.0 [52.9, 68.4] | 25.5 / 9.5 | +18.5 [14.7, 22.6] |
| EnCodec 12 kb/s * | 51.8 / 15.8 | +41.2 [33.2, 49.3] | 17.1 / 8.5 | +10.1 [7.2, 13.1] |
| EnCodec 24 kb/s * | 40.6 / 14.1 | +30.0 [23.6, 36.6] | 13.4 / 8.6 | +6.4 [3.8, 9.5] |
| MP3 32 kb/s * | 33.4 / 11.2 | +22.8 [17.5, 28.1] | 22.3 / 8.9 | +15.3 [11.0, 19.6] |
| MP3 64 kb/s * | 12.7 / 9.9 | +2.1 [0.8, 3.5] | 9.3 / 7.2 | +2.3 [0.9, 4.1] |
| MP3 128 kb/s * | 10.0 / 10.1 | -0.6 [-1.5, 0.2] | 6.7 / 7.3 | -0.3 [-0.9, 0.4] |
| EQ 3 dB | 11.3 / 11.0 | +0.7 [0.0, 1.7] | 6.9 / 6.6 | -0.1 [-0.7, 0.5] |
| EQ 6 dB | 12.6 / 11.5 | +2.0 [0.7, 3.8] | 8.1 / 7.0 | +1.1 [0.3, 1.8] |
| EQ 12 dB * | 18.7 / 12.9 | +8.1 [5.7, 10.4] | 13.4 / 8.3 | +6.4 [4.5, 8.6] |
| Mix/separate -6 dB * | 57.8 / 17.5 | +47.2 [38.2, 57.4] | 35.5 / 13.3 | +28.5 [20.8, 37.1] |
| Mix/separate 0 dB * | 26.3 / 12.1 | +15.7 [11.2, 20.9] | 17.8 / 9.3 | +10.8 [7.8, 14.1] |
| Mix/separate 6 dB * | 14.2 / 10.2 | +3.6 [1.7, 5.8] | 10.3 / 7.9 | +3.3 [1.9, 4.5] |
| Mix/separate 12 dB * | 10.0 / 10.4 | -0.6 [-2.1, 0.7] | 8.2 / 7.3 | +1.2 [0.4, 2.0] |

**Table B4.** Torres SSL. Clean error is 9.46% on VocalSet and 7.80% on GTSinger.

| Condition | VocalSet G/B (%) | VocalSet shift (pp) | GTSinger G/B (%) | GTSinger shift (pp) |
| --- | --- | --- | --- | --- |
| Reverb 0.1 wet * | 92.5 / 30.7 | +83.1 [78.1, 88.0] | 50.5 / 20.8 | +42.7 [30.3, 55.2] |
| Reverb 0.2 wet * | 100.0 / 36.5 | +90.5 [85.2, 95.2] | 87.7 / 28.1 | +79.9 [69.4, 88.7] |
| Reverb 0.3 wet * | 100.0 / 32.5 | +90.5 [85.2, 95.1] | 95.2 / 24.0 | +87.4 [80.7, 93.5] |
| Reverb 0.5 wet * | 100.0 / 31.0 | +90.5 [85.6, 95.1] | 96.2 / 23.0 | +88.4 [83.0, 93.5] |
| EnCodec 1.5 kb/s * | 100.0 / 44.8 | +90.5 [85.5, 95.4] | 81.1 / 28.2 | +73.3 [63.2, 82.9] |
| EnCodec 3 kb/s * | 99.9 / 39.5 | +90.4 [85.0, 95.0] | 74.2 / 26.5 | +66.4 [54.4, 78.2] |
| EnCodec 6 kb/s * | 99.2 / 36.3 | +89.7 [84.8, 94.1] | 67.2 / 26.7 | +59.4 [47.2, 71.2] |
| EnCodec 12 kb/s * | 97.5 / 36.7 | +88.1 [82.3, 93.3] | 60.0 / 26.1 | +52.2 [40.2, 63.5] |
| EnCodec 24 kb/s * | 95.4 / 36.5 | +85.9 [79.3, 91.6] | 51.6 / 25.9 | +43.8 [32.5, 55.5] |
| MP3 32 kb/s * | 67.9 / 30.8 | +58.5 [47.5, 69.8] | 36.5 / 24.6 | +28.7 [17.8, 40.8] |
| MP3 64 kb/s * | 59.0 / 22.8 | +49.6 [37.6, 61.7] | 26.6 / 16.4 | +18.8 [12.9, 25.3] |
| MP3 128 kb/s * | 30.6 / 17.5 | +21.1 [13.4, 29.3] | 20.3 / 14.2 | +12.5 [8.3, 17.1] |
| EQ 3 dB | 10.0 / 10.9 | +0.5 [-0.3, 1.4] | 7.4 / 7.8 | -0.4 [-1.7, 0.6] |
| EQ 6 dB | 13.5 / 12.8 | +4.1 [1.9, 6.4] | 8.6 / 8.9 | +0.8 [-0.9, 2.3] |
| EQ 12 dB * | 28.1 / 17.4 | +18.6 [14.3, 23.4] | 13.2 / 12.4 | +5.4 [2.3, 9.0] |
| Mix/separate -6 dB * | 97.6 / 35.6 | +88.1 [82.5, 93.2] | 55.2 / 26.9 | +47.4 [34.7, 59.6] |
| Mix/separate 0 dB * | 83.5 / 30.5 | +74.1 [66.3, 81.3] | 29.8 / 17.5 | +22.0 [11.2, 33.7] |
| Mix/separate 6 dB * | 64.5 / 24.1 | +55.0 [48.4, 62.0] | 24.9 / 16.0 | +17.1 [6.8, 27.9] |
| Mix/separate 12 dB * | 48.3 / 19.0 | +38.9 [31.6, 45.8] | 19.3 / 13.3 | +11.5 [4.0, 19.1] |

**Table B5.** ECAPA-TDNN. Clean error is 16.65% on VocalSet and 6.41% on GTSinger.

| Condition | VocalSet G/B (%) | VocalSet shift (pp) | GTSinger G/B (%) | GTSinger shift (pp) |
| --- | --- | --- | --- | --- |
| Reverb 0.1 wet * | 19.5 / 17.7 | +2.9 [0.3, 5.8] | 7.7 / 7.3 | +1.3 [0.4, 2.5] |
| Reverb 0.2 wet * | 21.3 / 19.7 | +4.6 [1.0, 8.5] | 8.8 / 8.3 | +2.4 [0.9, 4.2] |
| Reverb 0.3 wet * | 20.9 / 21.1 | +4.2 [-0.4, 8.4] | 9.7 / 9.1 | +3.3 [1.3, 5.6] |
| Reverb 0.5 wet * | 24.4 / 23.3 | +7.7 [2.3, 13.2] | 11.7 / 10.6 | +5.3 [2.8, 8.0] |
| EnCodec 1.5 kb/s * | 36.4 / 27.5 | +19.7 [13.7, 26.0] | 26.4 / 18.8 | +20.0 [12.9, 27.0] |
| EnCodec 3 kb/s * | 28.6 / 24.3 | +11.9 [7.1, 16.9] | 12.4 / 10.0 | +6.0 [2.3, 9.6] |
| EnCodec 6 kb/s * | 26.4 / 21.8 | +9.7 [5.7, 13.9] | 8.8 / 9.5 | +2.4 [0.3, 4.6] |
| EnCodec 12 kb/s * | 24.0 / 20.3 | +7.3 [3.7, 11.0] | 8.1 / 8.6 | +1.7 [-0.2, 3.6] |
| EnCodec 24 kb/s * | 23.9 / 21.2 | +7.3 [3.8, 10.9] | 7.5 / 8.1 | +1.1 [-0.7, 2.9] |
| MP3 32 kb/s * | 20.2 / 17.7 | +3.5 [0.8, 6.5] | 9.7 / 8.3 | +3.3 [1.2, 5.6] |
| MP3 64 kb/s * | 17.5 / 17.2 | +0.9 [-0.6, 2.9] | 7.1 / 7.0 | +0.7 [0.1, 1.3] |
| MP3 128 kb/s * | 17.3 / 17.3 | +0.6 [-0.7, 1.9] | 6.9 / 6.7 | +0.5 [0.0, 0.9] |
| EQ 3 dB | 16.3 / 16.6 | -0.3 [-1.1, 0.4] | 6.7 / 7.0 | +0.3 [-0.4, 0.9] |
| EQ 6 dB | 16.5 / 17.0 | -0.2 [-1.1, 0.7] | 7.1 / 7.1 | +0.7 [-0.2, 1.8] |
| EQ 12 dB * | 17.5 / 17.9 | +0.8 [-0.6, 2.3] | 7.4 / 7.1 | +1.0 [-0.2, 2.5] |
| Mix/separate -6 dB * | 32.0 / 32.6 | +15.4 [9.2, 20.6] | 21.2 / 14.1 | +14.8 [9.5, 21.2] |
| Mix/separate 0 dB * | 26.7 / 27.0 | +10.1 [5.8, 14.3] | 12.4 / 10.8 | +6.0 [3.3, 9.3] |
| Mix/separate 6 dB * | 23.4 / 21.7 | +6.7 [3.9, 9.7] | 9.7 / 8.7 | +3.3 [1.1, 6.3] |
| Mix/separate 12 dB * | 21.4 / 19.6 | +4.8 [2.5, 7.2] | 8.4 / 9.0 | +2.0 [0.6, 3.8] |

**Table B6.** MERT. Clean error is 15.06% on VocalSet and 15.27% on GTSinger.

| Condition | VocalSet G/B (%) | VocalSet shift (pp) | GTSinger G/B (%) | GTSinger shift (pp) |
| --- | --- | --- | --- | --- |
| Reverb 0.1 wet * | 17.6 / 15.0 | +2.6 [1.6, 3.7] | 17.1 / 15.8 | +1.8 [1.0, 2.9] |
| Reverb 0.2 wet * | 22.0 / 15.3 | +6.9 [4.1, 9.9] | 18.7 / 15.8 | +3.5 [2.2, 4.9] |
| Reverb 0.3 wet * | 27.5 / 14.8 | +12.4 [9.2, 16.0] | 22.2 / 16.2 | +7.0 [4.5, 9.7] |
| Reverb 0.5 wet * | 46.0 / 17.1 | +31.0 [24.0, 38.1] | 32.8 / 16.6 | +17.5 [12.7, 21.9] |
| EnCodec 1.5 kb/s * | 53.0 / 23.7 | +38.0 [31.3, 44.8] | 27.1 / 17.6 | +11.9 [8.6, 15.1] |
| EnCodec 3 kb/s * | 35.8 / 18.2 | +20.7 [15.8, 25.6] | 22.2 / 17.0 | +7.0 [4.3, 9.8] |
| EnCodec 6 kb/s * | 29.0 / 18.3 | +14.0 [10.2, 18.2] | 20.3 / 17.2 | +5.0 [3.1, 7.2] |
| EnCodec 12 kb/s * | 26.1 / 18.5 | +11.1 [7.9, 14.7] | 19.3 / 17.4 | +4.0 [2.3, 6.0] |
| EnCodec 24 kb/s * | 24.5 / 18.3 | +9.5 [6.5, 12.6] | 18.8 / 17.3 | +3.5 [1.8, 5.6] |
| MP3 32 kb/s * | 22.0 / 16.2 | +6.9 [4.1, 9.9] | 20.0 / 15.9 | +4.8 [2.9, 7.0] |
| MP3 64 kb/s * | 15.4 / 14.9 | +0.3 [-0.4, 1.0] | 15.2 / 15.0 | -0.0 [-0.5, 0.5] |
| MP3 128 kb/s * | 15.0 / 15.3 | -0.0 [-0.6, 0.5] | 15.2 / 15.1 | -0.1 [-0.4, 0.1] |
| EQ 3 dB | 15.8 / 15.6 | +0.7 [0.2, 1.3] | 15.3 / 15.2 | +0.0 [-0.3, 0.4] |
| EQ 6 dB | 16.7 / 16.2 | +1.6 [0.7, 2.8] | 16.0 / 15.4 | +0.7 [0.0, 1.5] |
| EQ 12 dB * | 23.6 / 17.9 | +8.6 [5.9, 11.5] | 19.7 / 17.2 | +4.4 [2.5, 6.8] |
| Mix/separate -6 dB * | 28.3 / 20.5 | +13.2 [9.2, 18.0] | 25.3 / 17.1 | +10.0 [6.1, 14.3] |
| Mix/separate 0 dB * | 20.0 / 18.5 | +4.9 [3.0, 6.7] | 20.1 / 16.6 | +4.8 [2.5, 7.8] |
| Mix/separate 6 dB * | 17.2 / 16.2 | +2.2 [1.0, 3.5] | 17.5 / 15.5 | +2.3 [0.3, 4.6] |
| Mix/separate 12 dB * | 16.1 / 15.8 | +1.0 [0.2, 1.9] | 16.3 / 15.0 | +1.0 [-0.1, 2.4] |

---

## S3. Measured responses and gallery checks

Four pairs of measured responses were chosen for similar broadband T30 decay estimates. Each pair contains two different filters. Within a factorial comparison, only the impostor response varies. Query and genuine response assignments are each balanced across both filters. The study averages the four pairs equally.

CoVox and NUS contributed 924 and 1,584 eligible triplets per cell, respectively. In the gallery comparison there were 21 impostors per CoVox gallery and 11 per NUS gallery. For each query and genuine recording pair, the mixed condition used 16 nearly balanced assignments and their 16 exact complements. This gave each impostor equal exposure to the two response assignments. The same gallery recordings also formed the two reference conditions: all impostors share the query response, or all receive its paired alternate. The query, genuine candidate and gallery membership stay fixed in these comparisons.

**Table C1.** Response files from the MIT survey [17] and estimated T30 in seconds. The attachment records full SHA-256 fingerprints.


| Pair | First response file | T30 (s) | Second response file | T30 (s) |
| --- | --- | --- | --- | --- |
| 1 | h001_Bedroom_65txts.wav | 0.310 | h248_Classroom_1txts.wav | 0.319 |
| 2 | h032_Bedroom_6txts.wav | 0.464 | h259_Classroom_1txts.wav | 0.435 |
| 3 | h045_Livingroom_4txts.wav | 0.606 | h198_Hallway_MITCampus_1txts.wav | 0.597 |
| 4 | h036_Bathroom_5txts.wav | 0.794 | h065_Classroom_3txts.wav | 0.804 |

**Table C2.** Paired error shift at 20% wet, in percentage points with 95% intervals. A positive shift means more errors when the impostor shares the query response.

| Pair | CoVox Torres SSL | CoVox CVSM-ART | NUS-48E Torres SSL | NUS-48E CVSM-ART |
| --- | --- | --- | --- | --- |
| 1 | -0.19 [-0.89, 0.49] | +0.05 [-0.28, 0.61] | +0.30 [-0.39, 0.97] | -0.02 [-0.14, 0.15] |
| 2 | +5.06 [3.30, 6.83] | +0.41 [0.02, 0.93] | +7.73 [4.34, 12.40] | +0.19 [-0.06, 0.54] |
| 3 | +0.57 [-0.58, 1.67] | -0.03 [-0.36, 0.33] | +6.72 [3.53, 10.77] | +0.22 [-0.16, 0.85] |
| 4 | +3.73 [1.62, 5.77] | +0.38 [-0.28, 1.21] | +3.00 [1.59, 5.12] | +0.24 [-0.02, 0.66] |

**Table C3.** Paired error shift at 50% wet, in percentage points with 95% intervals. A positive shift means more errors when the impostor shares the query response.

| Pair | CoVox Torres SSL | CoVox CVSM-ART | NUS-48E Torres SSL | NUS-48E CVSM-ART |
| --- | --- | --- | --- | --- |
| 1 | +4.14 [1.83, 6.98] | +0.11 [-0.51, 0.74] | +4.92 [2.27, 8.64] | +0.30 [0.05, 0.66] |
| 2 | +29.68 [24.38, 34.67] | +4.46 [2.35, 6.89] | +16.92 [12.71, 23.17] | +2.16 [0.96, 3.85] |
| 3 | +25.65 [20.89, 30.79] | +1.52 [0.48, 2.86] | +23.74 [17.19, 31.09] | +1.56 [0.49, 3.05] |
| 4 | +30.14 [24.48, 35.79] | +5.55 [2.89, 8.68] | +11.99 [8.22, 17.85] | +2.16 [0.88, 3.83] |

**Table C4.** Absolute top-rank error (%) under the three gallery arrangements. Table 5 gives paired uncertainty for mixed minus all alternate.

| Corpus | Encoder | Wet | All alternate | Mixed | All matching |
| --- | --- | --- | --- | --- | --- |
| CoVox | Torres SSL | 20% | 57.24 | 60.06 | 61.51 |
| CoVox | Torres SSL | 50% | 53.98 | 74.95 | 79.55 |
| CoVox | CVSM-ART | 20% | 44.46 | 45.45 | 46.45 |
| CoVox | CVSM-ART | 50% | 42.61 | 49.30 | 53.27 |
| NUS-48E | Torres SSL | 20% | 40.58 | 49.59 | 54.17 |
| NUS-48E | Torres SSL | 50% | 39.24 | 61.89 | 69.18 |
| NUS-48E | CVSM-ART | 20% | 22.31 | 22.70 | 23.05 |
| NUS-48E | CVSM-ART | 50% | 20.36 | 24.60 | 28.08 |

**Table C5.** Singer-weighted clean margin percentiles and mean symmetric impostor cosine shift when the impostor shares the exact query filter. Both genuine response assignments enter the contrast with equal weight.

| Corpus | Encoder | 10th | 50th | 90th | 20% wet | 50% wet |
| --- | --- | --- | --- | --- | --- | --- |
| CoVox | CVSM-ART | 0.0073 | 0.0426 | 0.0886 | +0.0003 | +0.0057 |
| CoVox | Torres SSL | -0.0082 | 0.0471 | 0.1233 | +0.0074 | +0.0511 |
| NUS-48E | CVSM-ART | 0.0204 | 0.0933 | 0.1548 | +0.0008 | +0.0069 |
| NUS-48E | Torres SSL | 0.0015 | 0.1359 | 0.3947 | +0.0213 | +0.1025 |

## S4. Separation on recorded mixtures

**Table D1.** Track-macro ranking errors (%) for the paired MUSDB18 comparison. The query always uses the reference stem. The effect and interval refer to separated genuine minus stem, calculated before rounding.

| Encoder | Tracks | Stems (%) | Separated genuine (%) | Both candidates separated (%) | Shift (pp) | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| CVSM-ART | 147 | 1.1 | 18.0 | 4.6 | 16.8 | [14.2, 19.8] |
| CVSM-A | 147 | 1.8 | 2.7 | 2.1 | 0.9 | [0.6, 1.3] |
| Torres SSL | 147 | 1.1 | 3.7 | 3.3 | 2.6 | [1.7, 3.7] |
| ECAPA-TDNN | 147 | 3.3 | 4.0 | 4.3 | 0.7 | [0.3, 1.1] |
| MERT | 147 | 6.9 | 7.9 | 7.8 | 1.0 | [0.5, 1.4] |

The 147 tracks retained for each encoder consist of 100 marked train and 47 marked test in MUSDB18. The selection file identifies seven rejected crop locations across three tracks. The rule retained a track whenever the reference query and both versions of a different genuine window could be scored. The file musdb18-crop-manifest.json records the centered excerpt start and the chosen window offsets within that excerpt, along with validity by encoder. Track selection spans the separator's training domain and is an auxiliary analysis.

## S5. Verification and reproducibility

**Check the figures and tables in the PDF.** Table A1 supplies the eight JVS cells and Table A2 isolates the headline example with exact cosine means. Table 4 uses the 17 entries marked in S2. S3 supplies response IDs, pairwise effects and gallery error rates. Use the paired intervals when judging differences. Subtracting the endpoints of the separate cell intervals does not give a paired interval.

**Recompute results from saved scores.** Extract the separate supplementary ZIP, then run **python3 verify_results.py** from the extracted directory. The script uses the standard library. It recomputes JVS cell ranking rates, mean genuine and impostor scores, margins, and point contrasts from the anonymized trial scores. It checks the printed core effect arithmetic, the common 17-condition means, and the measured-response pooled and gallery contrasts against the full-precision archived results. It also checks the archive file hashes. The main-paper shared-condition intervals can be reproduced with python3 verify_core_counts.py using NumPy. JVS and measured-response intervals remain archived bootstrap estimates.

**Inspect the experiment record.** The archive contains the full-precision results JSON, table and checkpoint provenance, the recorded room-matching protocols and a README. The JSON carries both the outcome and the source-file SHA-256 digest for each major analysis. These digests identify the archived source outputs. The compact JVS score table stores pseudonymous query and impostor IDs and cosines only. For the room comparison, the eight source response names appear in Table C1, and the four leave-one-pair-out estimates are recorded for each corpus, encoder and wetness.

**Understand the uncertainty.** For VocalSet and GTSinger, 2,000 paired query-singer resamples retain the original gallery. For JVS and the primary room analysis, 5,000 resamples draw singer identities within the grouping used to build trials. A singer drawn more than once receives that multiplicity in both query and impostor roles. A singer never becomes an impostor for their own query. The same draws are used for coupled contrasts. Room-response intervals condition on the fixed response files. Gallery intervals for the secondary comparisons keep candidate assignments fixed and resample query singers.

**Inspect the underlying replay checks.** Core results were checked by recomputing tables from the saved embeddings, rerunning selected frozen encoders on existing crops, and regenerating those six VocalSet conditions from the source recordings. The first check tests score aggregation. The second also tests inference. The last includes signal rendering. Differences in the regenerated EnCodec and HTDemucs examples are recorded in the JSON. The tested checkpoints are identified there by released checkpoint name and stored file fingerprints. These replays were performed on the research workspace. Running them from scratch requires the original recordings, model weights, processing code and dependencies. The attachment supports independent calculation from saved scores and inspection of the reported outputs.

## S6. Mitigation and artist-level sensitivity checks

Fine-tuning with production-varied positives reduced the effect relative to matched positives by 7.0 points on VocalSet [6.0, 7.9] and 4.4 on GTSinger [3.6, 5.4]. Clean error changed by less than 0.2 points on each corpus. Neither reduction reached the original 50% criterion. The supplement reports the arm trained without reverberation. For CVSM-ART, the crop-consistency adapter reduced mean error over its selected shifts from 60.7% to 21.9% on VocalSet evaluation singers, with clean error increasing from 7.8% to 13.9%. On GTSinger, these changed from 46.1% to 23.0% and from 8.9% to 13.2%, respectively. The clean penalty differed across encoders.

An auxiliary Music4All [21] analysis used 72 full-mix excerpts from 12 artist labels. Positive-only 32 kb/s MP3 re-encoding increased artist-macro ranking error by 35.26 points for CVSM-ART, 14.84 for Torres SSL and 24.46 for MERT. For the locally trained implementation of Yakura et al. [5], native error was 47.29% and the change was +0.35 points [−5.25, 6.01], with an artist bootstrap interval. This is an inconclusive shift at a high native error. We have no corresponding Yakura F2/G1 production sweep. The prior local VocalSet technique pilot used five held-out singers, a technique change and an any-impostor error definition. It serves a different question. Music4All includes precompressed audio, artist attribution and potential CVSM pretraining overlap, so it is an auxiliary check on these recordings.
