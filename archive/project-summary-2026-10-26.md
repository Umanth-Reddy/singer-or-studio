# Singer or Studio? Project summary (as of 26 Oct 2026)

This one file covers the whole project so far. **Part 1** is in plain language. **Part 2** has the
technical detail and the numbers behind every statement in Part 1.

> **Assumption used for the submission odds:** the offline listening study (about 50 listeners) is
> taken as fully in our favour, meaning people could still tell which recording was the same singer
> where the programs could not. Those responses have **not** been analysed yet. Every statement that
> depends on them is marked **[human study]**.

---

# PART 1: Plain language

## 1. The problem

Many programs claim they can recognise **who is singing** from a recording. People want to use them to:
- find songs by the same singer,
- credit or protect artists,
- catch AI copies of a singer's voice.

But a singer never sounds exactly the same in two recordings. The room, the microphone, whether the
song was streamed or compressed, whether the voice was pulled out of a full band mix: all of these
change the sound without changing *who* is singing.

**Our question:** when only the *recording* changes, do these programs still recognise the singer,
or do they get fooled?

## 2. What we did

1. Took clean recordings of about 40 singers from two public collections: 20 VocalSet singers and
   19 GTSinger singers, singing several songs each.
2. Changed only how they sound: added echo, compressed them like a low-quality stream, changed the
   tone, or mixed the voice into a band and pulled it back out (as apps that "remove vocals" do).
3. Gave each program a simple test: *"Here is singer A. Which of these two is also singer A?"*
   - Option 1: the same singer, but with the sound changed.
   - Option 2: a different singer, with clean sound like the first recording.
4. Ran this on five programs. Three were built specifically for singers. The other two were general
   ones: a program for recognising people's speaking voices, and a general music program.
5. Played the same kind of test to about 50 people. **[human study]**
6. Re-checked everything: real rooms instead of a fake echo, real songs instead of our own mixes,
   rerunning from scratch, and ruling out volume differences.

## 3. What we found

- **The singer-specific programs get fooled badly.** After a change in sound they often pick the
  *wrong* singer, simply because that recording "sounds like the same studio." Example: one program
  goes from about 8% wrong on clean audio to about 92% wrong after the voice is mixed into a band
  and pulled back out.
- **The information about the singer is still in there.** When both options get the same sound
  change, the programs mostly recover. So they don't forget the singer; they let the
  recording conditions outweigh the voice.
- **Surprise: a program made for *speaking* voices does much better.** It is far less fooled, and on
  clean singing it is about as accurate as the singer programs. A second speech program confirmed this.
- **People are not fooled.** Listeners still pick the right singer when the sound changes.
  **[human study]**

## 4. What did NOT hold up (being honest)

- **The echo result was mostly caused by our test setup.** Our first echo was fake and identical for
  every clip. With real recorded rooms, the echo only fools one of the three singer programs, not all three.
- **On real songs, only one program clearly fails at "pulled out of a mix."** The other singer
  program failed on our artificial mixes but barely on real songs.
- **Not "all singer programs fail."** Two families of singer programs fail. The general programs don't.
- **We have no fix yet.** Our attempts to retrain the programs helped only partly, or made them
  worse on clean audio.
- **Small scale:** about 40 singers, 4-second clips, and sound changes we made ourselves on a computer.

## 5. The core issue, in one sentence

**Singer-recognition programs partly listen to the *recording* instead of the *voice*, and nobody
had measured how much.** The likely reason: they were trained on music where each singer's songs
tend to share a "studio sound," so the recording became a shortcut. Speech programs were trained on
messy real-world audio (phones, rooms, noise), so they learned to ignore it.

## 6. Why this matters, and why it's Q2-worthy

- **It's a clear, useful warning.** Anyone using these programs for search, artist credit or
  AI-copy detection could get wrong answers from a simple change in sound quality.
- **It's new.** We found no earlier study that measures this for singer recognition and compares
  it with people.
- **It's well tested.** Two singer collections, five programs plus a second speech program, results
  reproduced from scratch, volume ruled out, real rooms and real songs checked, rules written before
  testing.
- **It has a practical takeaway:** a speech-voice program is a much safer choice today.
- **What keeps it from being clearly Q1:** no fix, a small number of singers, mostly computer-made
  sound changes, and only two kinds of singer programs tested.

## 7. Chances (assuming the human study is fully in our favour)

| Venue level | Chance | Why |
|---|---|---|
| **Q2 journal** (e.g. EURASIP JASMP, TISMIR-level) | **~65–75%** | A new, honest, well-checked finding with a human comparison. |
| **Top conference** (ISMIR / ICASSP) | **~60–70%** | Fits the "benchmark + analysis" papers these accept. |
| **Q1 journal** (e.g. IEEE/ACM TASLP) as it stands | **~20–25%** | Reviewers will ask for a fix, more singers and more models. |
| **Q1 with the upgrades below** | **~35–45%** | Fills the gaps Q1 reviewers usually point to. |

**Upgrades that would raise the Q1 chance:**
1. A fix that makes singer programs robust **without** losing accuracy on clean audio.
2. One or two more singer programs from other research groups, and more singers.
3. Real-world recordings, e.g. the same singer live vs. in the studio, not just computer-made effects.
4. Human-study questions with **real** room echo, not the fake one.

---

# PART 2: Technical detail

## A. Setup

- **Data:** VocalSet (20 singers, songs caro/dona/row, "straight" excerpts) and GTSinger
  (19 singers, Control_Group files). Loudest 4 s window per clip, peak-normalised to 0.5.
  Real-song check: 150 MUSDB18 songs (htdemucs-separated vocals vs ground-truth stem).
- **Encoders (all frozen):**
  - CVSM-ART (htdemucs vocal tags) and CVSM-A (label-agnostic), both from Music4All, arXiv 2510.03025;
  - SSL singer identity (Torres et al., ISMIR 2023, contrastive-vc);
  - ECAPA-TDNN (speechbrain, VoxCeleb);
  - MERT-v1-95M;
  - extras: WavLM-Base-Plus-SV and x-vector (speechbrain).
- **Trial:** clean query (singer s, song i). Positive: s, song j≠i, processed. Negative: another
  same-sex singer, song j, clean. Error = cos(q, neg) ≥ cos(q, pos). Singer-macro error, effect =
  processed minus clean, 2000× singer-bootstrap 95% CI. Pass rule: effect ≥ 10 pts with CI > 0.
- **Both-processed control:** the negative gets the same processing as the positive.
- **Fixed failing set S** (17 conditions: codec 1.5–24 kbps, MP3 32–128, reverb 0.1–0.5,
  mixsep −6 to +12 dB, EQ 12 dB), chosen from VocalSet before GTSinger was run.

## B. Main results

**F1 (VocalSet, matched vs cross-production error):** CVSM-ART 9.2 → 58.5%, CVSM-A 10.9 → 37.8%,
SSL 13.1 → 69.5%. Compression and EQ have almost no effect (≤ 3 pts, except SSL EQ +10.6).

**F2a dose-response (VocalSet error %, clean → condition):**

| Encoder | clean | reverb 0.3* | codec 6 kbps | MP3 64 | mixsep 0 dB | EQ 12 dB |
|---|---|---|---|---|---|---|
| CVSM-ART | 8.1 | 45.0 | 74.2 | 23.4 | 92.4 | 16.8 |
| CVSM-A | 10.6 | 63.2 | 71.6 | 12.7 | 26.3 | 18.7 |
| SSL | 9.5 | 100.0 | 99.2 | 59.0 | 83.5 | 28.1 |
| ECAPA | 16.6 | 20.9 | 26.4 | 17.5 | 26.7 | 17.5 |
| MERT | 15.1 | 27.5 | 29.0 | 15.4 | 20.0 | 23.6 |

\*synthetic noise reverb; see T1.

**G1 replication (GTSinger, effect over S):** CVSM-ART +37.2 [32.9, 41.2], CVSM-A +25.8,
SSL +44.5 [36.8, 53.0], ECAPA +4.4, MERT +5.2. The pre-registered replication passes for CVSM-ART and SSL.

**Both-processed control:** errors fall to near baseline. For example, CVSM-ART on VocalSet falls
from 65.2% cross-processed to 18.2% both-processed (clean 8.1%). Identity information survives;
the failure is a mismatch effect.

## C. Robustness checks

**T1: real rooms (8 MIT IR Survey RIRs) and 5 held-out MUSDB accompaniments.** Effect in pts,
VocalSet / GTSinger, with the original synthetic recipe in brackets:

| Encoder | Reverb (real rooms) | Mix-separate (new accompaniments) |
|---|---|---|
| CVSM-ART | +5.2 / +4.6 (+32 / +26) | **+55.9 / +46.5** (+70 / +57) |
| CVSM-A | +7.7 / +6.7 | +6.3 / +4.3 |
| SSL | **+42.7 / +27.4** | **+42.9 / +18.3** |
| ECAPA | +1.6 / +0.9 | +4.6 / +4.1 |
| MERT | +2.6 / +2.6 | +2.6 / +1.7 |

The reverb family FAILS the rule (1 of 3 specialised encoders). Mix-separate HOLDS (2 of 3 on both
datasets). SSL's reverb effect grows with room T30. CVSM's appears only for long rooms (T30 ≈ 0.9–1.6 s).

**T3:** WavLM-SV is robust (effect +8.6 / +7.2) but has high clean error (31.0 / 23.5%).
CVSM-ART + ECAPA score fusion: clean 11.8 / 5.6%, effect +21.5 / +9.2. It fails the rule on VocalSet
clean cost.

**Independent checks, 26 Oct** (`claude-check-oct-26/REPORT.md`):

1. **Rerun from source:** clean, reverb, MP3 and EQ renders are bit-identical; codec and mixsep
   differ at waveform level (neural nondeterminism). All error rates are within 1 pt of the stored results.
2. **Loudness:** RMS-matching and ±6/−12 dB gain change nothing (0.0 pts, all encoders). By
   construction, CVSM L2-normalises each 1 s window, SSL peak-normalises and MERT standardises.
3. **Real songs (147 MUSDB songs, separated vs ground-truth stem):**
   - CVSM-ART +16.8 [14.2, 19.8];
   - SSL +2.6 [1.7, 3.7];
   - CVSM-A +0.9, ECAPA +0.7, MERT +1.0.
   
   This is an easier trial (same-song positive), so sizes don't compare directly to F2a.
4. **Speech-model comparison (paired bootstrap vs ECAPA, effect over S):**
   - CVSM-ART +40.8 / +26.8, CVSM-A +25.2 / +13.1, SSL +63.8 / +40.1 pts more than ECAPA, all CIs > 0.
   - Clean error: on VocalSet the singer models are 6–9 pts better (borderline CIs). On GTSinger
     there is no significant difference (CVSM-ART +2.5 [−0.3, 5.4]).
   - x-vector: robust (+5.9 / +2.8) but higher clean error (21.8 / 14.5%).
5. **Training overlap:** none found. CVSM was trained on Music4All. The Torres SSL paper lists
   VocalSet as an out-of-domain test set. GTSinger (2024) postdates both models. ECAPA and x-vector
   were trained on VoxCeleb.

## D. Mitigation attempts (no clean fix yet)

- **C1** (fine-tune CVSM-A with production-varied positives): reduces the effect by 7.0 pts
  (VocalSet) and 4.4 pts (GTSinger) vs the control arm. It **does not pass** the pre-set rule
  (partial effect only).
- **M2** (label-free crop-consistency adapter on CVSM-ART): error on the failing conditions falls
  from 60.7% to 21.9% (VocalSet) and from 46.1% to 23.0% (GTSinger). But clean error rises
  7.8 → 13.9% (VocalSet), a real trade-off. Report it as a baseline, not a fix.
- **Score fusion** with ECAPA: lowers the effect but costs clean accuracy on VocalSet.

## E. Ideas tested earlier and dropped

- **P1** (AI-generated vocals are more "homogeneous"): codec alone explains 45–82% of the gap. Failed.
- **K1** (artist retrieval after album debiasing): failed its rule.
- **R1** (album effect): small, +3 to +9 pts, mostly inside the CIs.

## F. Human study (H1) **[human study]**

- **Design:** VocalSet trials in 6 conditions (clean, reverb@0.3 synthetic, codec@6, MP3@64,
  mixsep@0, EQ@6), 6 per condition, with 3 repeats and 3 catch trials. Two counterbalanced forms
  (A/B), models scored on the identical trials.
- **Pre-set pass rule:**
  - human accuracy ≥ 80% on effect trials;
  - within 10 pts of clean accuracy;
  - singer models ≥ 20 pts below humans;
  - exclude listeners who fail ≥ 2 of 3 catch trials.
- **Status:** run offline with about 50 listeners and assumed favourable. Still to do:
  - digitise the responses;
  - confirm which clip set was played;
  - score with the pre-set rule (crossed listener × trial bootstrap, mixed-effects logistic model);
  - report consent and listening conditions.
- **Caveat:** the reverb trials use the synthetic recipe. Report the human reverb result as recipe-specific.

## G. Claims: what to write and what not to write

| Claim | Status |
|---|---|
| Codec/MP3 mismatch breaks CVSM and SSL; general encoders stay robust | ✅ Supported (2 datasets) |
| Mix-then-separate breaks CVSM-ART, including real songs | ✅ Supported |
| Mix-then-separate breaks SSL | ⚠️ Synthetic mixes only |
| Real-room reverb breaks SSL (grows with T30) | ✅ Supported |
| Reverb breaks CVSM | ❌ Only the synthetic recipe or long rooms |
| Identity survives; the failure is a mismatch effect | ✅ Supported |
| Speech SV models are robust and about as accurate on clean singing | ✅ Supported (VocalSet: slightly less accurate) |
| "Current singer encoders fail" (in general) | ❌ Narrow to CVSM and Torres SSL |
| Listeners ignore these changes | **[human study]** assumed ✅ |
| A fix exists | ❌ Baselines only (C1, M2, fusion) |

## H. Novelty and related work

- **Closest prior work:**
  - Hsieh et al. 2020, accompaniment confounds in singer ID (overlaps the mixsep angle);
  - Deng et al., ICASSP 2025, sensitivity of general audio embeddings to effects (not singer
    identity, no humans);
  - the album/producer effect in artist ID (long known);
  - speech channel-robustness literature.
- **Our gap:**
  - production-mismatch benchmark specifically for singer identity;
  - dose-response, real-room and real-song checks;
  - both-processed control;
  - speech-SV comparison;
  - matched human listening test.
- No direct prior found in a targeted search, but the search was not a full systematic review.

## I. Paper plan (revised)

- **Working title:** *Singer or Studio? Singer-Identity Embeddings Confuse Production With Voice.*
- **Primary target:** a Q2 journal (EURASIP JASMP / TISMIR-level) or ISMIR. TASLP only after the upgrades in Part 1, §7.
- **Before submitting:**
  1. score the human study;
  2. update `docs/06_paper_plan.md` and `docs/05_status.md`, which still describe the pre-T1 claims;
  3. commit protocols and results to git, so the pre-registration dates can be checked;
  4. confirm audio rights (VocalSet CC BY 4.0; check GTSinger and MUSDB terms);
  5. make the figures.

## J. Where things are

- Pilot code and results: `pilots/` (F1, F2a, G1, C1, M1/M2, P1, K1, R1, H1).
- Real-room and fusion tests: `TEMP/` (`PROTOCOL.md`, `RESULTS.md`).
- Independent checks, 26 Oct: `claude-check-oct-26/` (`REPORT.md`, `results_*.json`).
- Listening trials: `listening_h1/`. `PRIVATE_trial_key.csv`, `models_on_trials.json` and
  `study_h1.json` reveal the answers: do not share them with participants.
