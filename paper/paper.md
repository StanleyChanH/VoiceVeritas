---
# JOSS submission draft — fill in the author block (name + ORCID of the
# corresponding author is required by JOSS at submission time) before opening
# the review issue at https://joss.theoj.org/papers/new
title: 'VoiceVeritas: open-source read-speech detection for interview anti-cheating'
tags:
  - Python
  - speech processing
  - audio classification
  - prosody
  - self-supervised learning
  - ethics
authors:
  - name: VoiceVeritas Contributors
    affiliation: 1
    corresponding: true
affiliations:
  - name: Independent
    index: 1
date: 17 November 2026
bibliography: paper.bib
---

# Summary

Online interviews are increasingly vulnerable to a simple cheat: the candidate
reads a prepared or LLM-generated answer from a hidden screen. Human reviewers
struggle to tell scripted from spontaneous speech reliably, yet the two styles
differ measurably in pause rhythm, F0 dynamics, disfluency patterns, and
spectral-temporal regularity. *VoiceVeritas* is, to our knowledge, the first
open-source implementation of read-speech detection aimed at this task: given
an interview-answer recording it returns
$P(\mathrm{read}\mid\mathrm{audio}) \in [0,1]$ together with per-segment
scores, a confidence value, an estimated SNR, and — for the interpretable
track — the prosodic features that drove the decision.

The package ships two complementary models behind one interface. *Track A* is
an interpretable baseline: energy-based voice-activity detection, 34 prosodic
features (F0 statistics, jitter, shimmer, pause distribution, rhythm variability,
syllable-rate dynamics) and a calibrated LightGBM classifier whose
feature attributions a human reviewer can inspect. *Track B* follows the
recipe validated by Elisha et al. [@elisha2024]: frozen wav2vec 2.0
[@baevski2020wav2vec] embeddings, chunk-level scoring, and median aggregation.
Any subset of models bundles into a portable median ensemble.

Beyond the models, the package emphasises honest deployment: a robustness
suite stresses every artifact under noise, telephony band-limiting, resampling,
and MP3 transcoding; each prediction carries an SNR estimate and refuses
confidence below 15 dB SNR; and evaluation reports EER and FPR@TPR95 because
false accusations are the costliest error. Per-utterance prediction artifacts
for the published benchmark are shipped in-repo so any AUC can be recomputed
without re-training. An ethics charter restricts use to assisting human review,
never automating rejection.

# Statement of need

Interview anti-cheating is a commercially active but closed field: vendors
market "read-speech scores" with no published method, no benchmark, and no
error bars. Researchers studying speech-style classification
[@elisha2024; @takam2024style] likewise release methods but not deployable
tooling. VoiceVeritas fills that gap for practitioners in hiring platforms,
universities, and proctoring teams who need an auditable baseline rather than
a black box: it is installable from PyPI (`pip install voiceveritas`), exposes
a CLI, a FastAPI service, and a Docker image, runs Track A in ~250 ms per
10 s of audio on CPU, and documents every known limitation — including a
cross-domain benchmark whose shortcut risks are explicitly postmortemed and a
held-out SBCSAE domain-shift evaluation for generalization claims.

The codebase is also a research substrate: data preparation for three public
corpora (LibriSpeech, AMI, SBCSAE), a noise-augmentation trainer, calibration,
and evaluation harnesses are all reusable modules, and every experiment is
reproducible from configuration files and seeds.

# State of the field

Scripted-vs-spontaneous classification has been studied with handcrafted
prosodic features and, more recently, self-supervised embeddings
[@elisha2024]. Existing work, however, (i) targets podcast or broadcast
domains rather than interview answers, (ii) evaluates within-corpus where
channel cues can act as shortcuts, and (iii) provides no maintained tooling.
VoiceVeritas contributes a benchmark with documented shortcut postmortems
(e.g. a forced-alignment feature family rejected after leaking corpus
identity), a noise-robustness report down to 5 dB SNR, and — uniquely —
per-utterance prediction releases that make the published numbers auditable.

# References