SINGER OR STUDIO? WHEN PRODUCTION CHANGES THE MATCH

1. Read supplement.md for the result tables, interpretation, and audit trail.
2. Run python3 verify_results.py to check the archive hashes and recompute the
   96 JVS factorial cells and 84 point contrasts. The script also checks the
   arithmetic of the measured-response summaries.
3. Run python3 verify_core_counts.py in an environment with NumPy to
   reconstruct the core trial membership from crop manifests, recalculate
   200 model-condition rates in each scoring arrangement, and reproduce all
   ten harmonized shared-condition confidence intervals.

Each JVS CSV row represents one query, genuine, and impostor combination
for one model, song orientation, and treatment. Query and impostor IDs
are pseudonyms local to the archive. "unique" means the singer-specific song.
The gAB and iAB columns hold genuine and impostor cosine values, with A
the query state and B the candidate state. 0 means clean and 1 means processed.
Equal scores count toward error. Each query singer receives equal weight.

The trial-audit JSON describes the core trial masks and preserves the earlier
and harmonized bootstrap intervals. SHA256SUMS verifies the consistency of
this supplement. Source and checkpoint fingerprints in the JSON identify
archived upstream files. JVS and room intervals are supplied as saved estimates.
Waveform rendering and model inference require access to the original corpora,
checkpoints, software dependencies, and the corresponding research pipeline.
This supplement contains no audio recordings, model weights, or embeddings.
