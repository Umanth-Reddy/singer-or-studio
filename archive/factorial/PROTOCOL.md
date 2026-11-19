# SOL Research exploratory protocol

This protocol was written before calculating the new contrasts. Prior project results are already known. Both analyses reuse existing cohorts and frozen embeddings.

## Question

Does sharing a production condition raise impostor similarity, and does that shift explain part of the observed ranking error? For each trial, the ranking margin is genuine cosine minus impostor cosine. A margin at or below zero is a ranking error.

## Synthetic reverb and MP3 on JVS-MuSiC

Use all 100 documented singers and the original six voice-range groups. Hold the trial identities fixed across conditions. Score the full binary factorial over query, genuine candidate, and impostor production. A zero means clean and a one means the selected transformation. Run the original common-song query and reverse to singer-specific queries against the common-song gallery. Treat those as two arrangements of the same singers. Synthetic reverb is primary and MP3 64 kb/s is secondary.

Report the direct ranking contrast between genuine-only processing (010) and query-only processing (100). In addition, hold the query and genuine candidate fixed and switch the impostor's condition. Average the matched-minus-unmatched impostor cosine and ranking-error differences over both query conditions and genuine conditions. This symmetric contrast cancels a constant score shift associated with processing the impostor alone. Report the genuine-score match contrast and the score and ranking changes in each factorial cell. This design measures a score effect under the chosen fixed transforms.

## Measured room responses on VocalSet and GTSinger

Use the existing eight measured acoustic responses applied by convolution to the source clips. Pair adjacent responses after sorting by broadband T30, independent of model scores. Use the existing wet fraction 0.2 as the primary level to match the JVS investigation and report 0.1, 0.3 and 0.5 as sensitivity analyses. The matching is approximate and does not control direct-to-reverberant ratio, coloration, or all decay differences. Report the individual pairs and their T30 gaps.

For each pair of responses, score the full two-response factorial with same-sex cross-song trials from the original protocol. Apply both response orientations and both genuine conditions with equal weights so the impostor matches the query response in half the cells. The causal comparison changes only the impostor response, with query, genuine candidate, trial identity, and wet fraction fixed. Report change in impostor cosine and ranking error, plus genuine-score match contrast. Show pooled summaries across the four fixed pairs and per-pair results. Include the clean versus wet-genuine-only contrast for context, averaged over the eight measured responses. Include ECAPA and MERT as reference encoders on both datasets.

## Uncertainty and checks

Average over trials within query singer and then across singers. Use 5,000 fixed-seed paired singer bootstrap resamples stratified by the original matching groups, resampling identities in query and gallery roles together. Intervals for pooled room contrasts describe singers with the measured responses held fixed. No multiplicity correction is applied to the descriptive secondary contrasts. Require complete embeddings for every trial included in a comparison, verify embedding hashes for JVS, and reproduce the archived JVS headline cells and T1 clean and positive-only results within 0.01 percentage points.

Advance to a new cohort only if the estimated effect survives the JVS role swap and extends to at least one specialized encoder on both measured-response datasets. The effect must show a positive impostor-score match contrast and a meaningful paired ranking penalty. A small penalty or broad inconsistency motivates reporting the scope of the effect in the existing paper. New singers and genuinely independent acoustic responses would be needed for confirmation. No adapter, threshold, or production predictor is fit in this analysis.
