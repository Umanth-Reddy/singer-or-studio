# Final decision on the shared-room response direction

**Decision: close the broad stand-alone Q1 direction tested here. Include the result in the existing benchmark paper.**

The room-matching effect survived on two cohorts outside the reverb discovery analysis. It did not clear the decision rule set before scoring. The main benchmark still gains an explanation of why some impostors outrank the genuine candidate and a measured limit on how large the effect is at the primary reverb setting.

## Test design

Two different singer cohorts and eight previously unused responses from the same MIT impulse-response survey. CoVox contributed 22 female singers singing the same two songs in pop style. NUS-48E contributed 12 singers with four recordings each. The NUS galleries use the same within-singer recording slot across singers, with no demographic match or verified title match. The trials average over all different-recording orders within singer. Both corpora were used previously for other questions. Encoder pretraining exposure is not independently established.

Both frozen encoders processed all 1,564 inputs generated from 92 source recordings. There are 924 CoVox and 1,584 NUS eligible triplets per cell. The same recording identities recur in the paired conditions. Each room pair uses different indoor room categories with an absolute broadband T30 gap no larger than 0.029 seconds. A common wet fraction and similar decay time leave other acoustic properties free to vary.

The primary contrast changes only the impostor's response from an alternate measured response to the exact query response. It averages both choices of query and genuine condition. Positive values mean choosing the matching response for the impostor increases ranking error. The 95% intervals resample singer identities in the query and gallery together, with selected responses fixed.

| Cohort | Encoder | Clean pairwise error % | Pairwise penalty at 20% wet (pp, 95% CI) | At 50% wet (pp, 95% CI) |
|---|---|---:|---:|---:|
| COVOX | Torres SSL | 13.31 | +2.29 [+1.45, +3.27] | +22.40 [+18.86, +25.90] |
| COVOX | CVSM-ART | 6.93 | +0.20 [-0.03, +0.52] | +2.91 [+1.60, +4.50] |
| NUS | Torres SSL | 9.53 | +4.44 [+2.54, +7.01] | +14.39 [+10.70, +19.54] |
| NUS | CVSM-ART | 4.55 | +0.16 [+0.01, +0.44] | +1.55 [+0.67, +2.70] |

## Decision against the frozen rule

To advance, Torres SSL needed a point estimate above +5 pp and a lower 95% bound above +2 pp on both cohorts at 20% wetness. At least three of four response pairs had to be positive on both cohorts, and the mixed galleries had to favor matching impostors on both.

- **CoVox:** +2.29 pp [1.45, 3.27], three of four response pairs positive. The fixed-response interval lies below +5 pp.
- **NUS:** +4.44 pp [2.54, 7.01], four of four response pairs positive. The point estimate misses +5 pp. The interval still admits that magnitude.
- Both mixed galleries favor matching impostors. That secondary endpoint does not replace the primary gate.

The decision applies to the broad effect at the primary wet fraction across datasets. The effect plainly exists under these conditions and grows at the stronger fraction. This does not establish the size of the effect across room types, recording systems, or a population of deployments.

## What happened to the top-ranked singer

Candidate identities are held fixed. Impostors are split roughly 50/50 across the two responses, with complementary assignments to balance the odd gallery sizes. The genuine recording is assigned either response with equal weight. In the table, ‘wrong winner excess’ is the difference between the fractions of all trials won by matching and nonmatching impostors. It describes which impostor wins. The final column changes the impostor pool from all nonmatching to balanced mixed, which measures total top-1 error change under that counterfactual. It is a post-hoc comparison with a query-singer bootstrap that holds the gallery and response set fixed.

| Cohort | Encoder | Wet | Mixed top-1 error % | Wrong winner excess (pp) | Mixed minus all-nonmatching top-1 error (pp, 95% CI) |
|---|---|---:|---:|---:|---:|
| COVOX | Torres SSL | 20% | 60.06 | +10.21 | +2.81 [+0.78, +4.90] |
| COVOX | Torres SSL | 50% | 74.95 | +55.88 | +20.97 [+18.29, +23.73] |
| COVOX | CVSM-ART | 20% | 45.45 | +1.03 | +0.99 [+0.40, +1.69] |
| COVOX | CVSM-ART | 50% | 49.30 | +14.62 | +6.69 [+4.91, +8.35] |
| NUS | Torres SSL | 20% | 49.59 | +18.54 | +9.01 [+7.48, +10.62] |
| NUS | Torres SSL | 50% | 61.89 | +47.53 | +22.65 [+20.71, +24.76] |
| NUS | CVSM-ART | 20% | 22.70 | +0.56 | +0.39 [+0.09, +0.74] |
| NUS | CVSM-ART | 50% | 24.60 | +6.59 | +4.25 [+2.93, +5.65] |

At 20% wet, Torres SSL's wrong winner was already skewed toward the matching room. Total top-1 error rose by 2.81 pp on CoVox and 9.01 pp on NUS in the post-hoc all-nonmatching comparison. At 50% wet, that change exceeded 20 points on both. The stronger reverb result belongs in the benchmark with its wet fraction and gallery design stated. The simulations convolve existing singing recordings with measured responses. They do not observe people actually recording in those rooms. No mitigation was fit.

## Integrity and reproduction

- All eight response hashes were unique and excluded the discovery response hashes. All 92 source recording hashes were distinct.
- Both embeddings and the rendered manifest are checked against stored SHA-256 values by the scorer. The original frozen checkpoint file hashes are in the two provenance files.
- Identical inputs on the two gallery branches produce exactly equal wrong-winner attribution and identical top-1 failure across gallery compositions. Independent direct scoring reproduced the primary CoVox SSL estimate exactly to the reported precision.
- The [protocol](DECISION_PROTOCOL.md), [code](confirm.py), [full results](mixed_gallery_results.json), [room pairs](room_pairs.json), [source manifest](manifest.json) and two local embedding caches are included. Source audio and rendered audio are omitted.

Run the scorer from this folder with the existing Python environment.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /legacy/ps/.venv-ssl/bin/python confirm.py score --out . --base ..
```

Data sources and attribution. [CoVox](https://doi.org/10.3758/s13428-025-02664-9), [NUS-48E](https://apsipa.com/proceedings/2013/papers/364_NUS-corpus-Duan-2927261.pdf), and [Traer and McDermott's MIT IR Survey](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html).
