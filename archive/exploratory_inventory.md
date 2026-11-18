# Exploratory, negative, intermediate, and superseded experiments

This index deliberately includes findings that weaken, qualify, or fail a hypothesis. Exact JSON data are preserved in the [core archive](raw_core_and_independent_checks.md), [methods archive](raw_methods_and_exploratory.md), and [shareable experiment archive](raw_shareable_experiment_data.md). “Exploratory” means the result is useful for shaping the claim but should not be presented as independent confirmatory evidence.

## Main PS pilots and diagnostics

| Experiment | Data/result | Status / caveat |
|---|---|---|
| F1 early broad pilot | CVSM-ART matched/cross 9.17/58.46% (+49.29 pp [46.16,52.17]); CVSM-A 10.89/37.82 (+26.93 [22.40,30.92]); SSL 13.09/69.55 (+56.46 [53.12,60.23]). | Superseded/refined by F2. `f1_results.json` and `f1_results_final.json` are duplicate outputs, not replications. |
| F2 VocalSet | 20 singers, five encoders, 19 conditions. | Primary controlled benchmark table. All results at [F2](f2_vocalset.md); recomputed exactly. |
| F3 low-pass follow-up | SSL baseline 9.46%; low-pass 16 kHz 17.30% (+7.84 pp [5.68,10.22]), 11 kHz 18.74% (+9.29 [5.97,12.99]); stored failure breakpoint not met. MP3-128 30.57% (+21.12 [13.44,29.61]). | Same VocalSet setup, SSL only, exploratory follow-up. |
| G1 GTSinger | 19 singers, five encoders, 19 conditions. | Second singer dataset; see [G1](g1_gtsinger.md); recomputed exactly. |
| K1 retrieval-pool diagnostic | 481 pool songs, 330 queries, 63 artists. SSL top-1 same-artist 30.9%, cross-album 13.0%; after debiasing 3.9% and 2.7%; predefined pass=false (cross-album ratio .209, CI [.026,.477]). | Not singer verification; interim files repeat final or are partial, not replications. |
| P1 generated-music collision test | 150 Suno, 150 Udio, 150 MUSDB reference, 149 GT stems; only 39 known same-artist cross-song pairs. CVSM-ART collision: MUSDB .54%, GT 1.89%, codec 3.15%, Suno 18.09%, Udio 10.60%; SSL 6.31%, 3.04%, 19.54%, 60.32%, 46.50%. | Distribution/collision analysis, not singer verification. CVSM-ART fails on both generative sets; SSL passes Suno but fails Udio. Interim and final files differ by completeness, not independent cohorts. |
| P2 embedding-distance descriptors | F2/G1 embedding cosine quantiles; e.g. CVSM-ART 5th percentile .728/.699, ECAPA .479/.420, SSL .840/.705. | Reuses F2/G1 embeddings; descriptive only. |
| R1 album shift | 45 artists. ART 15.79→19.16% (+3.36 pp, CI [−1.13,7.40]); A 24.13→29.79 (+5.66 [0.78,11.26]); SSL 22.72→31.26 (+8.55 [2.69,15.03]); ECAPA +4.55 [0.39,8.84]; MERT +3.32 [−1.70,8.11]. | Exploratory, within-project data; shows album effects can affect multiple encoders. |
| Real-song separation | 147 MUSDB tracks. CVSM-ART +16.8 pp [14.2,19.8], SSL +2.6 [1.7,3.7], CVSM-A +0.9, ECAPA +0.7, MERT +1.0. | The clear effect is model-specific (CVSM-ART); the same-song trial is easier and not directly comparable to F2. |
| Real-room reverb / accompaniment extensions | Results in `TEMP/t1_results.json` and `t1_f2_results.json`; synthetic accompaniment extension 3a was stopped. | Real-room result narrows the claim mainly to SSL rather than all singer encoders. See detailed report in [evidence reconciliation](../audit/evidence_reconciliation_2026-11-18.md). |
| T3 fusion | Speech-model fusion follow-up in `TEMP/t3_results.json`. | Exploratory; complete source numbers preserved in raw JSON. |

## Independent rechecks

- [`independent-checks/REPORT.md`](/legacy/ps/glass-breaker/independent-checks/REPORT.md) reports an independently reimplemented scorer on key VocalSet conditions. Deterministic crops were bit-identical; EnCodec and separation waveforms differed because those components are nondeterministic, while selected error rates remained within one point.
- Loudness/gain control found 0 pp effect.
- November 9 singer recheck ran fresh inference on selected existing crops and independently rescored stored full-condition embeddings. It did not freshly render all raw-audio transformations.
- Method reruns reproduced earlier outputs on the same singers and conditions; no new independent singer test was created.
- Human-study response files did not reconcile with the stated 42-trial package, so human outcome is not included as evidence.

## Separate CVSM Review-I exploratory work

[`credible-work-bro/CVSM_REVIEW_I_RESULTS.md`](/legacy/credible-work-bro/CVSM_REVIEW_I_RESULTS.md) and [raw Review-I result JSON](raw_review_i.md) describe a distinct small full-mix artist experiment: 480 one-second clips, 24 artists, artist-disjoint 16/4/4 split. Frozen CVSM-ART embeddings produced unseen-artist AUC .66863, EER 37.708%, top-1 50%; eight lightweight proxy experiments are not full CVSM retraining. Checkpoint ablations are confounded by checkpoint step/separation differences. This is not a replication of the singer-verification paper protocol and must remain separate from Music4All cross-check metrics.

## Dataset expansion checks

NUS and QMUL experiments are repeatedly used for method development and are no longer untouched confirmatory cohorts. QMUL follow-up has only four singers after excluding 11 ambiguous recordings. NUS has 12 singers and 48 recordings. The `hybrid_cohort`, `paired_metric`, `virtual_augmentation`, `nus_paired`, and `qmul_*` exact results are in [raw shareable JSON](raw_shareable_experiment_data.md). No claim of a final external validation should be made from these reused pilots.
