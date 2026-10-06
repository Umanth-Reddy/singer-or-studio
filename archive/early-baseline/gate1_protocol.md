# Gate 1 pilot protocol (frozen before model inference)

Frozen on 2026-10-06. This is the first, limited baseline experiment using the
two author repositories supplied by the researcher. No performance results were
inspected before this protocol was written.

## Models

- CVSM: authors' `models/htdemucs/m4all_cvsmart` checkpoint; frozen encoder
  representation from the author's artist-similarity path. The authors' paper
  evaluation depends on Music4All, which was not supplied, so this is a new
  dataset evaluation and is not a reproduction of their published scores.
- Singer Identity SSL: authors' pretrained `contrastive-vc` model; frozen
  1,000-dimensional encoder output at 44.1 kHz. No training or fine tuning.

## Primary comparison

Use VocalSet `FULL/*/excerpts/{straight,vibrato}` only. Each of the 20 singers
has three named excerpt songs recorded with straight and vibrato singing,
except any missing file is excluded along with its matched pair. For each
singer-song, the straight clip is the query, the same singer's vibrato clip is
the identity positive, and every *other* singer's straight clip for the same
song is an impostor candidate. This holds song and candidate technique fixed
while testing whether a technique change disrupts the identity match.

The primary metric is the fraction of queries where at least one impostor has
greater cosine similarity than the identity positive (ties count as errors).
There are at most 60 queries, but the 20 singers, not the clips, are the
independent units. Report the mean singer error rate and a 95% bootstrap
interval resampling singers (10,000 resamples; seed 20261006). The material
failure threshold is a 20% error rate: more than one failed identity match in
five controlled queries is consequential for a retrieval benchmark. A model
family counts toward Gate 1 only if the *lower* interval bound is at least
0.20. No threshold adjustment after inspecting outcomes.

## Audio and representations

- Decode source WAVs to mono; use a deterministic central four seconds,
  zero-padding shorter clips. Do not fit normalizers on evaluation clips.
- CVSM: resample to 16 kHz, split the four seconds into four one-second
  windows, L2-normalize each waveform (matching the authors' artist similarity
  evaluation), extract the frozen EfficientNet encoder outputs, average them,
  and L2-normalize the resulting clip vector.
- Singer Identity SSL: resample to 44.1 kHz, pass the entire four-second
  waveform through the official model, and L2-normalize its encoder output.
- Cache all clip embeddings with model revision/checkpoint SHA-256, manifest
  SHA-256, preprocessing description, and source clip IDs.

## Interpretation limits

This is a prespecified *pilot* for the supplied two models. VocalSet has only
20 singers and a narrow repertoire; its repeated three excerpts cannot be
treated as 60 independent singers. A Gate 1 `GO` from this pilot would be a
reason to continue to MERT, Yakura, GTSinger, and mixture interventions, not a
full paper result. A failure to meet the threshold is evidence against this
particular test, not a verdict on every formulation of the idea.

## Audit addendum (after the initial results)

The frozen protocol omitted one operation present in the SSL authors'
`eval.py`: peak-normalizing each waveform before inference. The first SSL run
was therefore not author-faithful and is superseded by a corrected run using
peak-normalized four-second crops. The query design, threshold, checkpoint,
and singer-bootstrap procedure did not change. This correction was made after
the first results were inspected; it must not be portrayed as part of the
originally frozen preprocessing plan.

The original CVSM central-four-second and corrected SSL central-four-second
results remain the mechanical pilot gate inputs. Full-excerpt CVSM pooling,
SSL BYOL, and three-crop SSL pooling were added after the initial results as
sensitivity checks. They are reported separately and do not create additional
independent model families or prespecified gate evidence.
