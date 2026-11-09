# Method experiments: rerun and cross-check

**Run date:** 2026-11-09. Scripts ran from scratch copies under `/tmp`; original code, results, and dataset folders under `/legacy/ps` were read-only. The M2 script initially had a hardcoded original output path; that write was rejected by the filesystem, then the scratch copy's output root was corrected and rerun. The source M2 result file remained dated Oct 30 and unchanged.

## What was rerun

- M2 crop-consistency adapter: seeds 0, 1, and 2 across VocalSet/GTSinger and the original/real-room/accompaniment variants.
- Clean-reference ridge adapter.
- Production-shift nuisance attenuation.
- Clean-similarity distillation.

Inputs are the existing M1 training embeddings and VocalSet/GTSinger benchmark embeddings/manifests under `/legacy/ps`. No raw audio or dataset was copied. These method runs retrain adapters from stored embeddings and rescore existing test embeddings; they do not create a fresh independent test set. All method experiments reuse already-inspected singers and conditions.

## Reproduction check

New outputs compared against the prior JSON files:

| Experiment | Maximum absolute numeric difference | Numeric values compared |
|---|---:|---:|
| M2 seeds | 0 | 1,260 |
| Clean-reference ridge | 1.4e-14 | 264 |
| Nuisance attenuation | 1.4e-14 | 264 |
| Clean-similarity distillation | 0 | 255 |

Thus the previous method results were reproducible from the saved inputs and copied scripts. The differences are floating-point rounding only.

## Findings

- **M2 remains the most consistent of the tested adaptations.** It lowers all-condition error for all three encoder families on both singer datasets, but usually raises clean-condition error. Across seeds, all-condition error ranges are CVSM-ART 16.8–20.9% / 17.9–20.6%, CVSM-A 15.8–19.4% / 18.9–20.7%, and SSL 17.4–19.1% / 21.2–26.8% (VocalSet / GTSinger). These are repeated fits on the same singers, not independent test replications.
- **Ridge is a trade-off, not a universal improvement.** On CVSM-A/GTSinger, ridge gives 16.9% all-condition error with 6.9% clean error, compared with M2 at 18.9% all / 14.5% clean. However, on CVSM-ART/GTSinger it is 28.1% all-condition error versus M2 at 20.6%; the robust ECAPA reference is 10.4% all / 6.4% clean. No paired inferential comparison or fresh held-out singers were run.
- **Nuisance attenuation is the most interesting follow-up, but not a validated solution.** It reaches 15.5% all-condition error for CVSM-A/GTSinger (7.7% clean), compared with raw 22.7% (7.0% clean), M2 18.9% (14.5% clean), and ridge 16.9% (6.9% clean). Yet it is not uniformly better across encoders or VocalSet, and ECAPA remains stronger on GTSinger. Hyperparameters were selected with four development singers; the test sets were already examined.
- **Distillation did not add a new method.** Selection chose the zero distillation-weight setting for every encoder, making the selected objective M2. The run reproduces that conclusion.

## Updated method-paper view

There is no credible general-purpose method-paper result yet. Nuisance attenuation merits a properly preregistered follow-up on fresh singers and unseen production families, with paired uncertainty, clean-condition constraints, compute cost, and ECAPA included. A winning point estimate on this reused 19-singer set is not enough to claim superiority or novelty. The current results support presenting M2 and these alternatives as baselines/partial mitigations in the problem paper, not as a solved method contribution.

## Files

- `results/`: exact rerun JSONs.
- `scripts/`: code snapshots used for the rerun. They rely on this machine's source embeddings and benchmark files under `/legacy/ps`; the datasets are not bundled.
