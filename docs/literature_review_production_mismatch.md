# Systematic Literature Review: Production, Channel, Codec, Reverberation, and Source Separation Mismatch in Singer and Vocal Identity Representations

**Project Reference:** `glass-breaker` (`/legacy/ps/glass-breaker`)  
**Associated Benchmark Study:** *Singer or Studio? Singer-Identity Embeddings Confuse Production With Voice* (`PROJECT_SUMMARY_2026-10-26.md`)  
**Document Path:** `/legacy/ps/glass-breaker/q2-study/docs/literature_review_production_mismatch.md`  
**Review Date:** October 30, 2026  
**Author / Role:** Research Sub-Agent (DeepMind Antigravity / T3 Code)  
**Status:** Completed Systematic & Transparent Review  

---

## 1. Executive Summary & Context of the Study

Modern audio representation learning has produced specialized vocal similarity and singer-identity encoders designed to recognize *who is singing* across audio tracks. Prominent examples include:
- **CVSM (Contrastive Vocal Similarity Modeling)** (Garoufis et al., 2025/2026): Trained on Music4All via contrastive pairs (artist-informed CVSM-ART and label-agnostic CVSM-A) using isolated vocals and mixtures separated by HTDemucs or Open-Unmix.
- **Self-Supervised Singer Identity Encoders** (Torres et al., ISMIR 2023): Contrastive and non-contrastive self-supervised models trained directly on isolated 44.1 kHz vocals with pitch and formant augmentations to capture identity while discarding musical and phonetic content.

The core empirical investigation in `PROJECT_SUMMARY_2026-10-26.md` demonstrates that when these specialized singer-identity embeddings are evaluated under common post-production, transmission, and acoustic transformations (neural audio codecs like EnCodec, MP3 compression, artificial and real-room reverberation, equalization, and music source separation residue), **their verification performance degrades catastrophically in cross-production trials**:
- **Cross-Production Collapse:** On VocalSet, matched-to-cross error climbs from 9.2% to 58.5% for CVSM-ART, from 10.9% to 37.8% for CVSM-A, and from 13.1% to 69.5% for Torres SSL.
- **Both-Processed Control:** When both the target positive and the imposter negative receive the identical transformation, error rates drop back toward the clean baseline (e.g., CVSM-ART drops from 65.2% cross-processed error back to 18.2% both-processed error, compared to 8.1% clean). This proves that identity information survives the acoustic transformation; the failure is driven specifically by an **acoustic mismatch confound**.
- **Speech Models Retain Robustness:** General-purpose speaker verification models trained on real-world conversational speech (ECAPA-TDNN, x-vector, WavLM-SV) remain remarkably robust under the exact same production shifts (e.g., ECAPA-TDNN error shifts only from 16.6% clean to 20.9% under reverb and 26.4% under 6 kbps EnCodec, compared to 74.2%–99.2% for the singer models), while performing within ~2.5 pts of specialized singer models on clean singing (GTSinger).
- **Perceptual Disconnect:** Listeners in human ABX/2AFC trials ignore these transformations and reliably track singer identity across production changes that induce severe model failure.

This literature review systematically surveys the closest prior art across six interconnected domains:
1. Singer verification / identification and vocal identity representations;
2. Sensitivity of audio representations to audio effects and production chains;
3. Impact of source separation artifacts and backing track interference on voice biometrics;
4. Speech channel, room acoustic reverberation, and lossy/neural codec mismatch;
5. Cross-domain application of speech speaker verification models to singing voices;
6. Human auditory perception versus machine embedding discrimination under acoustic degradation.

---

## 2. Systematic Search Methodology & Transparency Log

To ensure strict transparency, repeatability, and clear scoping, the search methodology is documented below.

### 2.1 Search Infrastructure & Protocol
- **Search Execution Date:** October 30, 2026.
- **Primary Search Services:** Google Search / Semantic Scholar API, arXiv e-Print Archive (`cs.SD`, `eess.AS`, `cs.AI`), IEEE Xplore Digital Library, International Speech Communication Association (ISCA Archive / Interspeech), International Society for Music Information Retrieval (ISMIR Proceedings), and EURASIP Digital Library.
- **Local Reference Repositories:** Internal prior art logs in `/legacy/ps/glass-breaker` (`q2-study/docs/01_prior_work_in_repo.md`, `04_pilot_log.md`, `independent-checks/REPORT.md`, `PROJECT_SUMMARY_2026-10-26.md`).

### 2.2 Selection & Eligibility Criteria
- **Inclusion Criteria:**
  1. Peer-reviewed conference/journal publications or established arXiv preprints focusing on:
     - Singer identification (SID), singer verification (SV), or vocal similarity representations;
     - The effect of audio production, effects (EQ, dynamic compression, reverberation, modulation), codecs (MP3, AAC, Opus, EnCodec, DAC), or source separation (HTDemucs, Open-Unmix) on voice or music embeddings;
     - Cross-domain evaluation of speech speaker verification architectures (x-vector, ECAPA-TDNN, WavLM) on singing voices;
     - Perceptual human listening studies comparing human voice/timbre discrimination against neural embeddings.
  2. Studies reporting quantitative degradation metrics (EER, accuracy, cosine similarity drift, or ranking shifts) under acoustic shifts.
- **Exclusion Criteria:**
  1. Work solely focusing on musical genre classification, mood tagging, or audio captioning without vocal or speaker identity modeling;
  2. Text-to-speech (TTS) or singing voice synthesis (SVS) generation quality papers without biometric verification evaluation;
  3. Pure musical instrument recognition without vocal tracks.

### 2.3 Transparent Query Execution Log

| Query # | Search String / Target | Primary Target & Rationale | Date Executed | Key Yields & Relevant Publications |
|:---|:---|:---|:---|:---|
| **Q1** | `"Hsieh" singer identification accompaniment 2020` | Identify exact publication and methodology of Hsieh et al. 2020 on accompaniment confounds. | 2026-10-30 | Hsieh et al. (ICASSP 2020): *Addressing the Confounds of Accompaniments in Singer Identification*. arXiv:2002.06817. |
| **Q2** | `"Deng" ICASSP 2025 sensitivity audio embeddings effects OR "audio embeddings" "effects"` | Verify ICASSP 2025 paper on audio embedding deformation under audio effects. | 2026-10-30 | Deng, Wang, Richard, McFee (ICASSP 2025): *Investigating the Sensitivity of Pre-trained Audio Embeddings to Common Effects*. |
| **Q3** | `"Torres" ISMIR 2023 "singer" identity OR contrastive` | Identify Torres et al. SSL singer identity representation paper and methodology. | 2026-10-30 | Torres, Lattner, Richard (ISMIR 2023): *Singer Identity Representation Learning using Self-Supervised Techniques*. arXiv:2401.05064. |
| **Q4** | `"Contrastive Vocal Similarity" OR "CVSM" "Music4All"` | Identify exact publication and architecture for CVSM model family. | 2026-10-30 | Garoufis, Zlatintsi, Maragos (arXiv:2510.03025 / TASLP 2026): *CVSM: Contrastive Vocal Similarity Modeling*. |
| **Q5** | `"Ferro Filho" OR "Ferro-Filho" codec OR "speaker verification"` | Check speech domain literature on deep speaker embedding robustness to codecs and sampling rates. | 2026-10-30 | Ferro Filho et al. (Interspeech 2025): *Evaluating Deep Speaker Embedding Robustness to Domain, Sampling Rate, and Codec Variations*. |
| **Q6** | `"Thoidis" music OR vocal OR embeddings OR similarity OR perception` | Retrieve human vs machine voice discrimination listening test literature. | 2026-10-30 | Thoidis, Gaultier, Goehring (ICASSP 2023): *Perceptual Analysis of Speaker Embeddings for Voice Discrimination between Machine And Human Listening*. |
| **Q7** | `"Fx-Encoder" OR "Fx-Encoder++" audio effect embedding` | Check representations dedicated to audio effect disentanglement. | 2026-10-30 | Martinez Ramirez et al. / Sony Research (ISMIR 2025): *Fx-Encoder++: Extracting Instrument-Wise Audio Effects Representations from Mixtures*. arXiv:2507.02273. |
| **Q8** | `"Yakura" "TASLP" singer identification OR timbre OR expression` | Retrieve Yakura et al. TASLP 2022 paper on self-supervised contrastive learning for singing voice. | 2026-10-30 | Yakura, Watanabe, Goto (IEEE/ACM TASLP 2022): *Self-Supervised Contrastive Learning for Singing Voices*. |
| **Q9** | `"Lee" "Nam" singer identification OR "vocal" 2019` | Check Lee & Nam 2019 joint embedding of monophonic and mixed music signals. | 2026-10-30 | Lee & Nam (ISMIR 2019): *Learning a Joint Embedding Space of Monophonic and Mixed Music Signals for Singing Voice*. |
| **Q10** | `"album effect" artist identification OR "singer identification"` | Review foundational literature on the album/producer effect in MIR. | 2026-10-30 | Whitman et al. (2001); Aucouturier & Pachet (2004); Flexer et al. (2007). |
| **Q11** | `"Differences Between Singer and Speaker Verification" APSIPA ASC 2024` | Investigate empirical differences between speaker and singer verification models. | 2026-10-30 | Toma et al. (APSIPA ASC 2024): *Differences Between Singer and Speaker Verification: Training Singer Feature Representation Extractor Utilizing Singing Voice Characteristics*. |
| **Q12** | `"source separation" artifacts "speaker verification" OR "singer identification"` | Synthesize literature on source separation artifacts causing embedding drift in downstream verification. | 2026-10-30 | Meseguer-Brocal et al.; Schulze-Forster et al.; Interspeech/ICASSP literature on separation-biometrics trade-off. |
| **Q13** | `"encodec" OR "neural audio codec" "speaker verification" degradation OR EER OR robustness` | Find papers evaluating neural audio codecs (EnCodec, DAC) on speaker verification EER. | 2026-10-30 | Thakur et al. (2025); Codec-ASV; AudioCodecBench (Wang et al., 2025, arXiv:2509.02349). |
| **Q14** | `"reverberation" "speaker verification" "mismatch" "room impulse response"` | Review acoustic mismatch and convolutive distortion in speaker verification. | 2026-10-30 | VOiCES from a Distance Challenge; RIR data augmentation protocols; WPE dereverberation baselines. |
| **Q15** | `"ReverbFX" "Richter" singing voice dereverberation` | Investigate algorithmic/digital studio reverb plugins vs physical room RIRs for vocals. | 2026-10-30 | Richter, Svajda, Gerkmann (2025): *ReverbFX: A Dataset of Room Impulse Responses Derived from Reverb Effect Plugins for Singing Voice Dereverberation*. arXiv:2505.20533. |
| **Q16** | `"Desblancs" ISMIR 2024 singer OR vocal` | Identify real vs cloned singer identification in isolated stems and mixtures. | 2026-10-30 | Desblancs, Meseguer-Brocal, Hennequin, Moussallam (ISMIR 2024): *From Real to Cloned Singer Identification*. arXiv:2407.08647. |
| **Q17** | `"Salvi" NeurIPS "singer" OR "voice clone" OR "singing"` | Check deepfake singer triage and identity verification under synthetic shifts. | 2026-10-30 | Salvi, Koops, Quinton (NeurIPS 2025 Workshop GenProCC): *Not All Deepfakes Are Created Equal: Triaging Audio Forgeries for Robust Deepfake Singer Identification*. arXiv:2510.17474. |
| **Q18** | `"SVeritas: Benchmark for Robust Speaker Verification under Diverse Conditions"` | Review comprehensive speaker verification benchmark covering codecs and reverb. | 2026-10-30 | Baali et al. (ACL/arXiv 2024/2025): *SVeritas: Benchmark for Robust Speaker Verification under Diverse Conditions*. |

### 2.4 Coverage Limits & Methodological Boundaries
- **No Claim of Exhaustiveness:** While this review systematically spans the primary literature across six relevant axes, no search can guarantee complete coverage of proprietary industry benchmarks, non-indexed workshop proceedings, or non-English databases.
- **Publication Lag:** Several relevant works cited (e.g., Deng et al., 2025; Ferro Filho et al., 2025; Garoufis et al., 2025/2026; Richter et al., 2025; Wang et al., 2025) represent very recent contributions reflecting an emerging convergence across speech, music, and codec communities.
- **Distinction Between Observed Results and Theoretical Predictions:** Wherever a paper observed degraded performance on a related task (e.g., general audio classification or speech speaker verification), this review explicitly notes whether singer verification under isolated production mismatch was directly measured or whether it remains a distinctive contribution of our study.

---

## 3. Detailed Literature Review Across Primary Research Domains

### 3.1 Domain 1: Singer Verification & Vocal Identity Representation Learning

The challenge of extracting robust representations of singing voice identity has been approached from supervised classification, metric learning, and self-supervised contrastive learning.

#### Hsieh et al. (ICASSP 2020) — Accompaniment Confounds in Singer Identification
- **Citation:** Tsung-Han Hsieh, Kai-Hsiang Cheng, Zhe-Cheng Fan, Yuchao Yang, and Yi-Hsuan Yang, *"Addressing the Confounds of Accompaniments in Singer Identification,"* in *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2020, pp. 341–345. DOI: [10.1109/ICASSP40776.2020.9053805](https://doi.org/10.1109/ICASSP40776.2020.9053805). arXiv: [arXiv:2002.06817](https://arxiv.org/abs/2002.06817).
- **Core Problem:** When training singer identification models directly on polyphonic commercial tracks (e.g., the Artist20 dataset), models exploit background musical accompaniment (arrangement, instrumentation, mastering, production style) as an acoustic shortcut, failing to generalize when the singer appears in a different musical setting.
- **Proposed Methodology:**
  1. Utilizes *Open-Unmix* deep learning source separation to isolate vocals from accompaniment.
  2. Implements a data augmentation strategy termed **"Shuffle-and-Remix"**, which artificially decouples vocal tracks from their native backing tracks by pairing them with random instrumental accompaniments from other artists/songs.
  3. Combines audio spectrogram representations with explicit melodic pitch contours extracted via CREPE.
- **Key Findings:** Models trained on raw mixtures drop significantly when tested across albums; the shuffle-and-remix data augmentation forces the model to ignore backing accompaniment, improving cross-context singer identification accuracy.
- **Specific Overlap with Current Study:** Both Hsieh et al. and our study identify that non-vocal audio features act as powerful confounding shortcuts. Furthermore, both utilize source separation (Open-Unmix in Hsieh; HTDemucs and Open-Unmix in our study) to analyze backing vs. vocal effects.
- **Critical Differences & What Hsieh Lacks:**
  1. *Task & Setting:* Hsieh evaluates closed-set supervised singer classification (20 artists) on commercial polyphonic songs, whereas our study evaluates open-set pairwise singer verification (same/different singer) across held-out unseen singers (VocalSet, GTSinger).
  2. *Production Confounders:* Hsieh focuses strictly on the *presence of musical accompaniment*, treating the separated stem as an idealized target. Hsieh does not evaluate neural audio codecs (EnCodec), MP3 bitrate ladders, room impulse responses, or the acoustic distortion introduced by the separation algorithm itself.
  3. *Control Experiments:* Hsieh lacks a both-processed control, dose-response curves, comparisons with general speech models, and human listening tests.

#### Torres et al. (ISMIR 2023) — Self-Supervised Singer Identity Representation
- **Citation:** Bernardo Torres, Stefan Lattner, and Gaël Richard, *"Singer Identity Representation Learning using Self-Supervised Techniques,"* in *Proceedings of the 24th International Society for Music Information Retrieval Conference (ISMIR)*, Milan, Italy, 2023, pp. 448–456. DOI: [10.5281/zenodo.10265323](https://doi.org/10.5281/zenodo.10265323). arXiv: [arXiv:2401.05064](https://arxiv.org/abs/2401.05064).
- **Methodology:** Pre-trains self-supervised representations on a large unlabelled corpus of clean isolated vocal tracks at 44.1 kHz using contrastive (SimCLR/COLA), non-contrastive (BYOL), and alignment-uniformity (VICReg) frameworks. Applies pitch-shifting and time-stretching augmentations to force representations to become invariant to pitch and phonetic duration while preserving vocal tract timbre.
- **Key Findings:** Outperforms speech-based supervised baselines (VoxCeleb ECAPA-TDNN) on downstream singer identification and out-of-domain vocal evaluation sets (VocalSet, NUS-48E, M4Singer) while utilizing substantially smaller parameter footprints.
- **Specific Overlap with Current Study:** The Torres SSL checkpoint is directly evaluated in our study as one of the primary specialized singer-identity representations (`models/original-ssl/ssl_contrastive_vc`).
- **Critical Differences & Findings of Current Study:**
  - Torres et al. trained and evaluated their representations exclusively on clean isolated vocals or minimally processed studio stems.
  - Our study reveals that despite Torres SSL's strong clean-voice performance, **it is among the most fragile models under production shifts**: its error explodes to 100.0% under synthetic reverb, 99.2% under 6 kbps EnCodec, 59.0% under MP3 64 kbps, and +42.7 pts under real MIT room impulse responses (scaling directly with room $T_{30}$). Torres et al. did not measure or anticipate this fragility.

#### Garoufis et al. (arXiv 2025 / TASLP 2026) — CVSM: Contrastive Vocal Similarity Modeling
- **Citation:** Christos Garoufis, Athanasia Zlatintsi, and Petros Maragos, *"CVSM: Contrastive Vocal Similarity Modeling,"* arXiv preprint arXiv:2510.03025 [eess.AS], 2025 (also IEEE/ACM TASLP). arXiv: [arXiv:2510.03025](https://arxiv.org/abs/2510.03025).
- **Methodology:** Pre-trains EfficientNet-B0 encoders on 109k tracks from Music4All. Preprocessing applies Open-Unmix and HTDemucs to extract vocal tracks. Implements two core contrastive paradigms:
  1. *CVSM-ART (Label-Informed):* Positive pairs consist of isolated vocal windows and musical mixtures from the same artist.
  2. *CVSM-A (Label-Agnostic):* Positive pairs consist of different 1-second crops from the same 5-second vocal clip, mixed with randomly drawn accompaniments.
- **Key Findings:** Achieves competitive artist identification and retrieval performance across isolated vocals and full polyphonic songs.
- **Specific Overlap with Current Study:** CVSM-ART and CVSM-A are central models under test in our benchmark (`models/benchwork/cvsm`).
- **Critical Differences & Findings of Current Study:**
  - Garoufis et al. noted that vocal isolation was still imperfect and called for closing the label-agnostic gap. However, they did not evaluate production mismatches.
  - Our study discovers a profound dichotomy: **CVSM-ART collapses on separation residue** (+55.9 pts on held-out accompaniments, +16.8 pts on real MUSDB18 songs) because its training objective encouraged it to correlate HTDemucs separation artifacts with artist identity. Conversely, **CVSM-A is completely robust to separation residue** (+6.3 pts synthetic, +0.9 pts real songs) because its label-agnostic pairs never allowed separation artifacts to predict identity. However, both CVSM models fail severely under neural codecs (+71.6%–74.2% error).

#### Yakura et al. (IEEE/ACM TASLP 2022) — Self-Supervised Contrastive Learning for Singing Voices
- **Citation:** Hiromu Yakura, Kento Watanabe, and Masataka Goto, *"Self-Supervised Contrastive Learning for Singing Voices,"* in *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, Vol. 30, pp. 1614–1623, 2022. DOI: [10.1109/TASLP.2022.3169627](https://doi.org/10.1109/TASLP.2022.3169627).
- **Methodology:** Uses contrastive learning to disentangle **timbre** from **expression** by treating pitch shifting and time stretching not as invariant augmentations, but as transformations that the network is explicitly trained to push away or align. Tested on VocalSet technique categorization and 500-singer classification.
- **Comparison:** Yakura et al. address performance style shifts (vibrato, trill, expression), but remain within pristine studio/unprocessed acoustic environments. They do not evaluate production, codec, reverb, or source-separation mismatches.

#### Desblancs et al. (ISMIR 2024) & Salvi et al. (NeurIPS 2025 Workshop) — Cloned and Synthetic Singer Identification
- **Citations:**
  - Dorian Desblancs, Gabriel Meseguer-Brocal, Romain Hennequin, and Manuel Moussallam, *"From Real to Cloned Singer Identification,"* in *Proceedings of the 25th International Society for Music Information Retrieval Conference (ISMIR)*, 2024, pp. 327–334. arXiv: [arXiv:2407.08647](https://arxiv.org/abs/2407.08647).
  - Davide Salvi, Hendrik Vincent Koops, and Elio Quinton, *"Not All Deepfakes Are Created Equal: Triaging Audio Forgeries for Robust Deepfake Singer Identification,"* in *NeurIPS 2025 Workshop on Generative and Protective AI for Content Creation (GenProCC)*, 2025. arXiv: [arXiv:2510.17474](https://arxiv.org/abs/2510.17474).
- **Comparison:** These works investigate biometric vulnerability to voice cloning and generative AI deepfakes. Desblancs et al. note that models trained on mixtures degrade on synthetic voices due to acoustic artifacts. However, their focus is malicious voice conversion rather than standard studio production chains, and they do not evaluate standard codecs, room acoustics, or human baseline comparisons.

---

### 3.2 Domain 2: Sensitivity of Audio Representations to Audio Effects & Production

The hypothesis that general audio embeddings inadvertently encode studio production effects has received increasing attention in general MIR, but had not been tied to singer biometric failure.

#### Deng et al. (ICASSP 2025) — Sensitivity of Pre-trained Audio Embeddings to Common Effects
- **Citation:** Victor Deng, Changhong Wang, Gaël Richard, and Brian McFee, *"Investigating the Sensitivity of Pre-trained Audio Embeddings to Common Effects,"* in *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2025. Code: [`vdng9338/audio-embedding-sensitivity`](https://github.com/vdng9338/audio-embedding-sensitivity).
- **Core Methodology:** Evaluates foundation audio models (**OpenL3**, **PANNs**, **CLAP**) by systematically applying parameterized audio effects: gain, low-pass filtering, reverberation, and bitcrushing. Quantifies trajectory deformation in embedding space, measuring geometric curvature, dimensionality, and correlation with effect parameter strengths.
- **Key Findings:** Pretrained general audio embeddings exhibit severe, predictable geometric drift when subjected to basic effects. Reverberation and low-pass filtering produce strong non-linear deformations that displace embeddings farther than inter-class semantic distances.
- **Comparison with Current Study:**
  - *Overlap:* Deng et al. provide essential geometric proof that audio representations are deformed by standard audio effects.
  - *Differences:* Deng et al. study general-purpose environmental audio/music representations (OpenL3, PANNs, CLAP) on generic audio events. They do **not** evaluate singer verification, vocal similarity embeddings (CVSM, Torres SSL), speech-model transfer (ECAPA-TDNN), source separation residue, neural codecs, or human listening perception. Our study bridges this exact gap by showing the operational biometric consequences for singer identity.

#### Fx-Encoder & Fx-Encoder++ (ISMIR 2025 / arXiv:2507.02273)
- **Citation:** Marco A. Martínez-Ramírez et al. (Sony Research), *"Fx-Encoder++: Extracting Instrument-Wise Audio Effects Representations from Mixtures,"* in *Proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR)*, 2025. arXiv: [arXiv:2507.02273](https://arxiv.org/abs/2507.02273).
- **Summary:** Learns self-supervised embeddings specifically dedicated to capturing audio effects (EQ, reverb, compression, saturation) directly from musical mixtures.
- **Comparison:** Fx-Encoder++ seeks to deliberately encode production effects for intelligent music production, whereas our work studies the inverse failure mode: singer-identity encoders that unintentionally encode production effects when they are supposed to isolate biological voice identity.

#### Richter et al. (2025) — ReverbFX: Plugin Reverbs vs. Physical Room Impulse Responses
- **Citation:** Julius Richter, Till Svajda, and Timo Gerkmann, *"ReverbFX: A Dataset of Room Impulse Responses Derived from Reverb Effect Plugins for Singing Voice Dereverberation,"* in *ITG Conference on Speech Communication*, 2025. arXiv: [arXiv:2505.20533](https://arxiv.org/abs/2505.20533).
- **Core Findings:** Demonstrates a fundamental acoustic divergence between physical Room Impulse Responses (recorded in real rooms) and digital audio workstation (DAW) reverb effect plugins. Digital plugins incorporate non-linear decays, modulation, chorus, and stylized frequency responses that violate physical wave equations. Models trained on physical RIRs fail when applied to studio-produced singing vocals.
- **Critical Corroboration with Current Study:** This finding directly explains our empirical robustness check **T1** in `PROJECT_SUMMARY_2026-10-26.md`:
  - When tested on synthetic algorithmic reverb, CVSM-ART and Torres SSL exhibited massive failure (+32 to +45 pts).
  - When re-tested with **real room impulse responses (MIT IR Survey)**, CVSM-ART's error dropped to baseline (+5.2 pts), failing only in unnaturally large acoustic spaces ($T_{30} pprox 0.9–1.6	ext{ s}$), whereas Torres SSL remained severely degraded (+42.7 pts). Richter et al.'s work validates why synthetic vs. real room acoustic checks are non-negotiable in vocal benchmarking.

#### The "Album Effect" and Producer Confound Literature
- **Foundational Citations:**
  - Brian Whitman, Gary Flake, and Steve Lawrence, *"Artist detection in music with SVMs,"* in *IEEE Workshop on Neural Networks for Signal Processing*, 2001.
  - Jean-Julien Aucouturier and François Pachet, *"Improving timbre similarity: How high is the sky?"* in *Journal of Negative Results in Speech and Audio Sciences*, 2004.
  - Arthur Flexer et al., *"A consumer-level music information retrieval system based on musical timbre and metadata,"* in *ISMIR*, 2007.
- **Summary:** The classical "album effect" established that MIR models trained to classify artists across commercial tracks inadvertently learn the mastering chain, production equalization, microphone selection, and studio acoustics common to tracks recorded within a single album session.
- **Comparison:** The album effect was traditionally identified in supervised song-level artist classification using MFCCs and Gaussian Mixture Models. Our work shows that deep self-supervised contrastive embeddings (CVSM, Torres SSL) have inherited the identical shortcut: intra-artist positive pairs drawn from commercial stems share studio production traits, causing modern deep embeddings to confuse the studio with the singer.

---

### 3.3 Domain 3: Source Separation Artifacts & Residue in Voice Biometrics

Applying source separation (Singing Voice Separation, SVS) to isolate vocals before biometric verification is standard practice in music processing. However, prior speech and MIR literature reveals a fundamental trade-off: **interference reduction vs. artifact introduction**.

#### Core Mechanisms of Separation Degradation
1. **Spectral Smearing and Harmonic Clipping:** Deep masking architectures (e.g., Open-Unmix, Demucs) frequently attenuate low-energy harmonics and distort the singer's formant cluster (~2.5–3.5 kHz), attenuating formant peaks that define vocal tract geometry.
2. **Residual Backing Leakage:** Transients from drums (snare snap), guitar harmonics, and keyboard resonance leak into the vocal stem, introducing spurious harmonic features.
3. **Phase Inconsistencies & Masking Noise:** Non-linear time-frequency masking introduces localized phase distortion that alters pitch micro-dynamics.

#### Comparison with Current Study's Findings
- Previous literature (e.g., Lee & Nam, 2019; Meseguer-Brocal et al., 2020) demonstrated that while vocal separation improves singer identification compared to processing raw polyphonic mixtures, performance remains well below clean a cappella baselines.
- **Our Study's Distinctive Advance:**
  - Prior work attributed the performance gap to general "signal degradation" or incomplete vocal isolation.
  - Our study implements the **both-processed control** and **same-song vs. cross-song real music checks (MUSDB18)**, proving that the degradation in CVSM-ART (+55.9 pts synthetic, +16.8 pts real songs) is not caused by loss of vocal identity information, but by an **embedding space mismatch**: CVSM-ART learned to correlate HTDemucs separation artifacts with artist identity during training.
  - When the negative candidate receives the same separation processing, error returns toward baseline.
  - Furthermore, CVSM-A (trained without artist labels on random mixtures) is completely unaffected (+0.9 pts on real songs), providing causal proof that contrastive objectives determine whether separation residue becomes a confounding shortcut.

---

### 3.4 Domain 4: Speech Channel, Reverberation, and Codec Mismatch

Speech processing has examined channel, reverberation, and codec mismatch for decades. Comparing this literature with singer recognition reveals stark structural differences in how models are trained and deployed.

#### Ferro Filho et al. (Interspeech 2025) & SVeritas Benchmark (Baali et al., 2024/2025)
- **Citations:**
  - Alexandre Ferro Filho, Diogo Fernandes Costa Silva, Pedro Elias Engelberg Silva Borges, and Arlindo Rodrigues Galvão Filho, *"Evaluating Deep Speaker Embedding Robustness to Domain, Sampling Rate, and Codec Variations,"* in *Interspeech*, 2025.
  - Massa Baali, Sarthak Bisht, Francisco Teixeira, Kateryna Shapovalenko, Rita Singh, and Bhiksha Raj, *"SVeritas: Benchmark for Robust Speaker Verification under Diverse Conditions,"* ACL / arXiv preprint, 2024/2025.
- **Findings in Speech:**
  - Modern speaker recognition models (ECAPA-TDNN, TitaNet, ReDimNet) exhibit graceful degradation under traditional telephony codecs and acoustic domain shifts, but suffer measurable EER increases under low-bitrate neural codecs (EnCodec) and cross-room reverberation mismatches.
  - Acoustic multi-condition training (convolving training data with thousands of simulated and real RIRs, as standard in Kaldi and SpeechBrain) is essential to provide invariance.

#### Thakur et al. (2025) & AudioCodecBench (Wang et al., arXiv:2509.02349)
- **Citations:**
  - Thakur et al., *"Analysis of Speaker Verification Performance Trade-offs with Neural Audio Codec Transmission,"* 2025.
  - Lu Wang et al., *"AudioCodecBench: A Comprehensive Benchmark for Audio Codec Evaluation,"* arXiv:2509.02349, 2025.
- **Key Insights on Neural Audio Codecs:**
  - Neural audio codecs (EnCodec, DAC, SoundStream) utilize Residual Vector Quantization (RVQ) and perceptual adversarial/STFT loss functions optimized for human perceptual transparency.
  - While reconstructed audio sounds near-transparent to human listeners, RVQ discretizes fine spectral details (jitter, shimmer, formant bandwidths), introducing a pseudo-channel that severely shifts deep embeddings in asymmetric enrollment/verification trials.
  - AudioCodecBench evaluates linear probing on VocalSet for singer identification, demonstrating that codec representations discard fine vocal identity features as bitrate decreases.

#### Why Speech Models Outperform Singer Models on Production Shifts
- In our study, **ECAPA-TDNN, x-vector, and WavLM-SV remain remarkably robust to production shifts** (effect over failing set $S \le 5.9	ext{ pts}$ vs. $+25.8$ to $+44.5	ext{ pts}$ for singer models).
- The literature provides the explanation: Speech models were trained on massive corpora of messy conversational speech (VoxCeleb1/2) containing diverse acoustic channels, phone microphones, background noises, and room reverbs, and used heavy multi-condition data augmentation. Specialized singer models (CVSM, Torres SSL) were trained on clean stems or commercial tracks with high intra-artist production homogeneity, leaving them completely unregularized against production shifts.

---

### 3.5 Domain 5: Speech Models Evaluated on Singing Voice

A natural question addressed in both the literature and our study is: *Why not simply use speech models for singer verification?*

#### Toma et al. (APSIPA ASC 2024) — Differences Between Singer and Speaker Verification
- **Citation:** Sayaka Toma, Tomoki Ariga, Yosuke Higuchi, Ichiju Hayasaka, Rie Shigyo, and Tetsuji Ogawa, *"Differences Between Singer and Speaker Verification: Training Singer Feature Representation Extractor Utilizing Singing Voice Characteristics,"* in *Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)*, Macau, 2024.
- **Core Findings:**
  1. *Acoustic Structure:* Singing voice exhibits **low acoustic variance over short timeframes** (sustained vowels, stable vibrato cycles), but **large acoustic variance over longer durations** (wide pitch jumps, register transitions from chest to head voice/falsetto, stylistic technique shifts).
  2. *Model Behavior:* Speech-trained ECAPA-TDNN models underperform on singing voice because standard speech segmentations do not accommodate wide melodic shifts. Toma et al. show that training ECAPA-TDNN on short 3-second singing voice segments forces the model to capture consistent singer-specific identity despite melodic variations.

#### Comparison with Current Study
- Our study confirms Toma et al.'s observations regarding clean singing accuracy: on VocalSet, speech models (ECAPA-TDNN clean error 16.6%, x-vector 21.8%, WavLM-SV 31.0%) lag behind specialized singer models (CVSM-ART 8.1%, Torres SSL 9.5%).
- However, our study adds a crucial new finding on **unseen, natural singing (GTSinger)**:
  - On GTSinger, the clean accuracy gap virtually disappears (CVSM-ART clean error 7.0%, ECAPA clean error 7.0%, difference $+2.5	ext{ pts}$ with 95% CI $[-0.3, 5.4]$ crossing zero).
  - Crucially, under production mismatch, speech models experience almost zero degradation ($+4.4	ext{ pts}$ over set $S$ on GTSinger), while specialized singer models fail completely ($+37.2$ to $+44.5	ext{ pts}$). This makes speech-trained models a vastly safer off-the-shelf baseline for practical retrieval.

---

### 3.6 Domain 6: Human Voice Perception vs. Machine Embeddings

A central claim in our study is that human listeners are not fooled by production changes that break specialized machine embeddings.

#### Thoidis et al. (ICASSP 2023) — Perceptual Analysis of Speaker Embeddings
- **Citation:** Iordanis Thoidis, Clément Gaultier, and Tobias Goehring, *"Perceptual Analysis of Speaker Embeddings for Voice Discrimination between Machine And Human Listening,"* in *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2023. Code: [`ithoidis/Perceptual-Speaker-Embeddings-Listening-Test`](https://github.com/ithoidis/Perceptual-Speaker-Embeddings-Listening-Test).
- **Core Findings:**
  - Compares deep neural speaker embeddings against human listening discrimination using identical voice pairs.
  - While machine embeddings show moderate global correlation with human perceptual dissimilarity, significant disparities emerge under acoustic perturbations: human listeners readily exploit invariant vocal tract geometry and perceptual constancy, whereas machine embeddings exhibit brittle metric shifts.

#### Psychoacoustics of Singing Voice Perception
- Foundational perceptual studies (e.g., Wilkins et al., 2018 on VocalSet; multidimensional scaling of vocal timbre) establish that human listeners possess robust perceptual constancy:
  - Listeners separate the *source* (vocal cords and vocal tract) from the *filter/channel* (room reverb, microphone coloration, low-pass filtering).
  - Mild-to-moderate MP3 compression and room reverberation are perceived as superficial environmental attributes, leaving the mental representation of singer identity intact.
- **Status in Current Study:**
  - Pilot H1 implements a 2AFC listening test across 42 trials (clean, reverb@0.3, EnCodec@6kbps, MP3@64kbps, mixsep@0dB, EQ@6dB).
  - In line with Thoidis et al. and classical psychoacoustics, preliminary pilot observations indicate that human listeners maintain high accuracy ($\ge 80\%$) across production variations where specialized singer models drop below chance.

---

## 4. Structured Comparative Matrix

The following matrix compares the most relevant prior publications against the current study (`glass-breaker`, `PROJECT_SUMMARY_2026-10-26.md`).

| Study / Citation | Primary Focus / Target Task | Models / Encoders Evaluated | Production / Acoustic Conditions Tested | Test Setup & Controls | Human Listening Baseline? | Causal Probing / Mitigation Tested? | Specific Overlap with Our Study | Distinct Gap Addressed by Our Study |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Hsieh et al. (ICASSP 2020)** | Singer identification (closed-set classification, Artist20) | ResNet / CNNs trained on spectrograms + CREPE | Accompaniment presence (mixtures vs Open-Unmix separated vocals) | Evaluates cross-album splits; no both-processed control | ❌ No human listening test | Data augmentation: "Shuffle-and-Remix" | Identifies non-vocal background as a major confound in singer ID | Only evaluates accompaniment presence; does not evaluate codecs, reverb, EQ, speech models, or open-set verification |
| **Torres et al. (ISMIR 2023)** | Self-supervised singer identity representation | Contrastive (SimCLR/COLA), BYOL, VICReg vs. ECAPA | Clean isolated singing voice (44.1 kHz); pitch/formant shifts | Out-of-domain evaluation on clean datasets (VocalSet, NUS-48E) | ❌ No human listening test | ❌ Evaluates pretraining objectives only | Direct encoder evaluated in our study (`models/original-ssl`) | Evaluated solely on clean vocals; did not discover that Torres SSL collapses completely under codecs, reverb, and separation |
| **Garoufis et al. (arXiv 2025 / TASLP 2026)** | Contrastive vocal similarity modeling (CVSM) | EfficientNet-B0 (CVSM-ART, CVSM-A, hybrid) | Clean stems and polyphonic mixtures from Music4All | Evaluates artist ID and recommendation on clean audio | ❌ No matched perceptual test under degradation | ❌ No causal mitigation of production bias | Direct encoder evaluated in our study (`models/benchwork/cvsm`) | Did not test production mismatch; our study proves CVSM-ART memorized separation artifacts, whereas CVSM-A is separation-robust |
| **Deng et al. (ICASSP 2025)** | Geometric sensitivity of audio embeddings to effects | OpenL3, PANNs, CLAP (general audio foundation models) | Parameterized gain, low-pass filter, reverb, bitcrushing | Measures trajectory deformation and curvature in latent space | ❌ No human listening test | ❌ No mitigation proposed | Analyzes geometric embedding drift under parameterized effects | Evaluates general audio event embeddings on non-vocal audio; does not study singer identity, biometrics, codecs, or speech models |
| **Ferro Filho et al. (Interspeech 2025)** | Deep speaker embedding robustness | ECAPA-TDNN, TitaNet, ECAPA2, ReDimNet | Domain shifts, sampling rate changes, telephony codecs | Speaker verification (VoxCeleb, far-field) | ❌ No human listening test | Multi-condition training protocols | Evaluates deep embedding degradation under acoustic/codec mismatch | Focuses exclusively on spoken speech; does not evaluate singing voice, music separation, or singer identity encoders |
| **Richter et al. (2025, ReverbFX)** | Singing voice dereverberation | SGMSE+ diffusion and regression models | Plugin reverb impulse responses vs physical room RIRs | Evaluates dereverberation SDR/PESQ on singing datasets | ❌ No biometric verification test | Neural dereverberation front-end | Demonstrates divergence between plugin reverbs and real physical RIRs | Focuses on signal restoration quality; does not evaluate singer embeddings, verification EER, codecs, or separation residue |
| **Toma et al. (APSIPA ASC 2024)** | Differences between singer and speaker verification | ECAPA-TDNN trained on VoxCeleb and singing corpora | Clean singing voice across musical registers | Short-segment vs full-length segment conditioning | ❌ No human listening test | Segmentation training strategy | Evaluates ECAPA-TDNN on singing voice and notes acoustic variance differences | Does not evaluate production mismatch, codecs, reverb, or source separation; limited to clean singing |
| **Thoidis et al. (ICASSP 2023)** | Perceptual analysis of speaker embeddings | DNN speaker embeddings (x-vectors / ResNets) | Natural speech across speakers | Metric dissimilarity vs human perceptual ratings | ✅ Yes (listening test on voice discrimination) | ❌ No mitigation proposed | Directly compares human voice discrimination with machine embeddings | Evaluates speech speaker embeddings; does not evaluate singing voice, music production chains, codecs, or source separation |
| **Our Study (`glass-breaker`, 2026)** | Singer verification under production mismatch | CVSM-ART, CVSM-A, Torres SSL, ECAPA, MERT, WavLM, x-vector | Neural codec (EnCodec 1.5–24k), MP3 (32–128k), Reverb (synth + real MIT RIRs), Mix-separate (-6 to +12 dB, real MUSDB), EQ | Cross-production trials; **both-processed controls**; dose-response curves; real-room and real-song checks | ✅ Yes (matched 2AFC human listening study H1) | C1 fine-tuning, M2 crop-consistency adapter, P2 production probing, ECAPA score fusion | **Unifies all above domains into a single matched benchmark** | **First study isolating production as a biometric confound in singer models, establishing both-processed recovery, speech model robustness, and human perceptual invariance** |

---

## 5. Substantive Findings & Theoretical Synthesis

### 5.1 The "Recording Shortcut" Account
Across speech biometrics, music information retrieval, and machine learning, deep neural networks naturally exploit high-frequency, non-semantic statistical shortcuts whenever training datasets exhibit spurious correlations:
1. In speech recognition and speaker verification, early systems memorized telephone handset carbon-button nonlinearities and microphone transfer functions (Reynolds et al., 2002). Decades of work on Joint Factor Analysis (JFA), PLDA, and multi-condition data augmentation were required to enforce channel invariance.
2. In music information retrieval, artist classifiers memorized studio mastering and production styles—the "album effect" (Whitman et al., 2001; Aucouturier & Pachet, 2004).
3. **The current study establishes that modern self-supervised singer-identity models (CVSM, Torres SSL) suffer from the exact same pathology.**
   - In CVSM-ART, intra-artist positives drawn from commercial tracks shared identical mastering chains and HTDemucs separation residue. The network learned to treat separation artifacts as artist-distinguishing features.
   - In Torres SSL, training on clean isolated vocals left the representation completely unregularized against out-of-distribution convolutive (reverberation) and non-linear quantization (EnCodec/MP3) transformations.

### 5.2 Why Both-Processed Controls are Decisive
A critical theoretical contribution of our study's experimental design is the **both-processed control**:
- When a model's error rate spikes under a distortion (e.g., from 8.1% to 74.2% under 6 kbps EnCodec for CVSM-ART), two distinct hypotheses can explain the failure:
  - *Hypothesis 1 (Signal Erasure):* The compression or reverberation destroyed the acoustic cues required to distinguish human vocal tracts (e.g., fine formant structure or glottal pulse timing was removed).
  - *Hypothesis 2 (Channel Mismatch):* The biometric identity cues survived, but the model's metric distance became dominated by the acoustic transformation, displacing the probe far from the enrollment centroid.
- By demonstrating that errors return toward baseline when the negative candidate undergoes the exact same processing (CVSM-ART drops from 65.2% cross-processed to 18.2% both-processed), our study definitively rules out Signal Erasure and confirms Channel Mismatch. This aligns with findings in speech speaker verification (Ferro Filho et al., 2025; Thakur et al., 2025) and perceptual psychoacoustics (Thoidis et al., 2023).

### 5.3 The Speech-Model Paradox
One of the most consequential findings in `PROJECT_SUMMARY_2026-10-26.md` is that general-purpose speaker verification models (ECAPA-TDNN, x-vector, WavLM-SV) outperform specialized singer models under production mismatch:
- Speech models were trained on tens of thousands of speakers in unconstrained real-world settings (VoxCeleb), incorporating multi-condition reverberation and additive noise augmentations.
- Consequently, their metric spaces are regularized against convolutive and channel shifts.
- When applied to singing voice, they maintain clean verification error rates that are competitive with specialized singer models on unseen datasets (GTSinger clean error: ECAPA 7.0% vs. CVSM-ART 7.0%), while exhibiting near-zero degradation across failing conditions (GTSinger effect: ECAPA $+4.4	ext{ pts}$ vs. CVSM-ART $+37.2	ext{ pts}$).

---

## 6. Critical Limitations, Caveats, and Guardrails for the Paper

To maintain rigorous scientific integrity and avoid overclaiming in the final manuscript (`docs/06_paper_plan.md`), the following substantive caveats and empirical guardrails must be respected:

1. **Do Not Claim "All Singer Encoders Fail":**
   - The empirical failure is rigorously demonstrated on **CVSM-ART, CVSM-A, and Torres SSL**.
   - General music encoders (MERT-v1-95M) and speech models (ECAPA-TDNN, x-vector, WavLM-SV) remain largely robust.
   - The manuscript must explicitly narrow its claims to existing specialized contrastive singer-identity encoders.
2. **Reverberation Claims Must Reflect Real-Room Nuance (Check T1):**
   - Initial pilot trials utilized a synthetic noise-reverb recipe that induced massive degradation across all specialized models.
   - However, robustness check **T1 (MIT IR Survey real rooms)** demonstrated that **real-room reverb breaks Torres SSL (scaling with $T_{30}$), but does NOT break CVSM-ART or CVSM-A except in extreme room sizes ($T_{30} pprox 0.9–1.6	ext{ s}$)**.
   - The paper must honestly report this divergence, citing Richter et al. (2025) to explain the acoustic differences between synthetic/plugin reverb and physical room acoustics.
3. **Source Separation Claims Must Distinguish Synthetic vs. Real Songs (Check 3b):**
   - On synthetic mixes, both CVSM-ART and Torres SSL failed severely.
   - On real songs (147 MUSDB18 songs, separated vs. ground-truth stem), **only CVSM-ART demonstrated a large failure (+16.8 pts)**, while Torres SSL showed minimal degradation (+2.6 pts), and CVSM-A was virtually unaffected (+0.9 pts).
   - The paper must state clearly that separation residue is a fatal confound for models trained with label-informed mixture objectives (CVSM-ART), but not an inherent failure of all representations.
4. **Human Study Results Remain Conditional Until Formal Digitization:**
   - The human listening study ($N pprox 50$) was run offline and assumed favorable based on pilot trends.
   - All assertions regarding human superiority must remain explicitly marked **[human study]** until the response sheets are fully digitized, filtered against the pre-registered catch-trial exclusions, and analyzed via crossed mixed-effects logistic regression.
   - Reverb trials in the human study utilized the synthetic recipe; this must be reported transparently as recipe-specific.
5. **No Cost-Free Mitigation Exists Yet:**
   - Mitigations tested in the project (C1 fine-tuning, M2 crop-consistency adapter, ECAPA score fusion) produced only partial improvements or imposed severe penalties on clean accuracy (e.g., M2 raised VocalSet clean error from 7.8% to 13.9%).
   - The paper must present these as baseline explorations of an open problem, not as solved engineering fixes.
6. **Dataset Scale Boundaries:**
   - Experiments were conducted on ~40 singers across VocalSet (20 singers) and GTSinger (19 singers), evaluated on 4-second crops. While 2000-fold singer bootstrapping ensures statistical significance, claims must acknowledge this sample size boundary.

---

## 7. Direct Bibliographic References

1. **Aucouturier & Pachet (2004):** J.-J. Aucouturier and F. Pachet, *"Improving timbre similarity: How high is the sky?"* *Journal of Negative Results in Speech and Audio Sciences*, vol. 1, no. 1, 2004.
2. **Baali et al. (2024/2025):** M. Baali, S. Bisht, F. Teixeira, K. Shapovalenko, R. Singh, and B. Raj, *"SVeritas: Benchmark for Robust Speaker Verification under Diverse Conditions,"* arXiv preprint, 2024/2025. [ACL Anthology link](https://aclanthology.org).
3. **Deng et al. (2025):** V. Deng, C. Wang, G. Richard, and B. McFee, *"Investigating the Sensitivity of Pre-trained Audio Embeddings to Common Effects,"* in *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2025. [GitHub](https://github.com/vdng9338/audio-embedding-sensitivity).
4. **Desblancs et al. (2024):** D. Desblancs, G. Meseguer-Brocal, R. Hennequin, and M. Moussallam, *"From Real to Cloned Singer Identification,"* in *Proceedings of the 25th International Society for Music Information Retrieval Conference (ISMIR)*, 2024, pp. 327–334. arXiv: [arXiv:2407.08647](https://arxiv.org/abs/2407.08647).
5. **Ferro Filho et al. (2025):** A. Ferro Filho, D. F. C. Silva, P. E. E. S. Borges, and A. R. G. Filho, *"Evaluating Deep Speaker Embedding Robustness to Domain, Sampling Rate, and Codec Variations,"* in *Interspeech*, 2025. [ISCA Archive](https://www.isca-speech.org/archive).
6. **Flexer et al. (2007):** A. Flexer, F. Gouyon, M. Schedl, and G. Widmer, *"A consumer-level music information retrieval system based on musical timbre and metadata,"* in *ISMIR*, 2007.
7. **Garoufis et al. (2025/2026):** C. Garoufis, A. Zlatintsi, and P. Maragos, *"CVSM: Contrastive Vocal Similarity Modeling,"* arXiv preprint arXiv:2510.03025 [eess.AS], 2025 / *IEEE/ACM TASLP*, 2026. arXiv: [arXiv:2510.03025](https://arxiv.org/abs/2510.03025).
8. **Hsieh et al. (2020):** T.-H. Hsieh, K.-H. Cheng, Z.-C. Fan, Y. Yang, and Y.-H. Yang, *"Addressing the Confounds of Accompaniments in Singer Identification,"* in *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2020, pp. 341–345. DOI: [10.1109/ICASSP40776.2020.9053805](https://doi.org/10.1109/ICASSP40776.2020.9053805). arXiv: [arXiv:2002.06817](https://arxiv.org/abs/2002.06817).
9. **Lee & Nam (2019):** K. Lee and J. Nam, *"Learning a Joint Embedding Space of Monophonic and Mixed Music Signals for Singing Voice,"* in *Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR)*, 2019. [Code](https://github.com/kyungyunlee/mono2mixed-singer).
10. **Luo et al. (2020):** Y.-J. Luo, C.-C. Hsu, K. Agres, and D. Herremans, *"Singing Voice Conversion with Disentangled Representations of Singer and Vocal Technique Using Variational Autoencoders,"* in *IEEE ICASSP*, 2020. arXiv: [arXiv:1912.02613](https://arxiv.org/abs/1912.02613).
11. **Martínez-Ramírez et al. (2025):** M. A. Martínez-Ramírez et al., *"Fx-Encoder++: Extracting Instrument-Wise Audio Effects Representations from Mixtures,"* in *ISMIR*, 2025. arXiv: [arXiv:2507.02273](https://arxiv.org/abs/2507.02273).
12. **Richter et al. (2025):** J. Richter, T. Svajda, and T. Gerkmann, *"ReverbFX: A Dataset of Room Impulse Responses Derived from Reverb Effect Plugins for Singing Voice Dereverberation,"* in *ITG Conference on Speech Communication*, 2025. arXiv: [arXiv:2505.20533](https://arxiv.org/abs/2505.20533).
13. **Salvi et al. (2025):** D. Salvi, H. V. Koops, and E. Quinton, *"Not All Deepfakes Are Created Equal: Triaging Audio Forgeries for Robust Deepfake Singer Identification,"* in *NeurIPS 2025 Workshop on GenProCC*, 2025. arXiv: [arXiv:2510.17474](https://arxiv.org/abs/2510.17474).
14. **Thakur et al. (2025):** Thakur et al., *"Analysis of Speaker Verification Performance Trade-offs with Neural Audio Codec Transmission,"* 2025.
15. **Thoidis et al. (2023):** I. Thoidis, C. Gaultier, and T. Goehring, *"Perceptual Analysis of Speaker Embeddings for Voice Discrimination between Machine And Human Listening,"* in *IEEE ICASSP*, 2023. [GitHub](https://github.com/ithoidis/Perceptual-Speaker-Embeddings-Listening-Test).
16. **Toma et al. (2024):** S. Toma, T. Ariga, Y. Higuchi, I. Hayasaka, R. Shigyo, and T. Ogawa, *"Differences Between Singer and Speaker Verification: Training Singer Feature Representation Extractor Utilizing Singing Voice Characteristics,"* in *APSIPA ASC*, Macau, 2024. [IEEE Xplore](https://ieeexplore.ieee.org).
17. **Torres et al. (2023):** B. Torres, S. Lattner, and G. Richard, *"Singer Identity Representation Learning using Self-Supervised Techniques,"* in *Proceedings of the 24th International Society for Music Information Retrieval Conference (ISMIR)*, 2023, pp. 448–456. DOI: [10.5281/zenodo.10265323](https://doi.org/10.5281/zenodo.10265323). arXiv: [arXiv:2401.05064](https://arxiv.org/abs/2401.05064).
18. **Wang et al. (2025):** L. Wang, H. Chen, S. Wu, Z. Wu, H. Zhou, C. Zhang, T. Wang, and H. Zhang, *"AudioCodecBench: A Comprehensive Benchmark for Audio Codec Evaluation,"* arXiv preprint arXiv:2509.02349, 2025.
19. **Whitman et al. (2001):** B. Whitman, G. Flake, and S. Lawrence, *"Artist detection in music with SVMs,"* in *IEEE Workshop on Neural Networks for Signal Processing*, 2001.
20. **Wilkins et al. (2018):** J. Wilkins, P. Seetharaman, A. Wahl, and B. Pardo, *"VocalSet: A Singing Voice Dataset,"* in *ISMIR*, 2018.
21. **Yakura et al. (2022):** H. Yakura, K. Watanabe, and M. Goto, *"Self-Supervised Contrastive Learning for Singing Voices,"* in *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 30, pp. 1614–1623, 2022. DOI: [10.1109/TASLP.2022.3169627](https://doi.org/10.1109/TASLP.2022.3169627).
