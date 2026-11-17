---
# JOSS submission draft — ready to submit. Open the review issue at
# https://joss.theoj.org/papers/new and reference this file + repository.
title: 'VoiceVeritas: open-source read-speech detection for interview anti-cheating'
tags:
  - Python
  - speech processing
  - audio classification
  - prosody
  - self-supervised learning
  - ethics
authors:
  - name: Chen Hao
    affiliation: 1
    corresponding: true
    orcid: 0009-0009-9228-8518
affiliations:
  - name: Independent
    index: 1
date: 17 November 2026
bibliography: paper.bib
---

# Summary

Online interviews are increasingly exposed to a simple cheat: the candidate
reads a prepared, or LLM-generated, answer from a hidden screen. Human
reviewers struggle to tell scripted from spontaneous speech reliably, yet
the two styles differ measurably in pause rhythm, F0 dynamics, disfluency
patterns, and spectral–temporal regularity [@gabler2023reconsidering].
*VoiceVeritas* is, to our knowledge, the first open-source implementation
of read-speech detection aimed at this task. Given an interview-answer
recording it returns $P(\mathrm{read}\mid\mathrm{audio}) \in [0,1]$ together
with per-segment scores, an estimated SNR, and — for the interpretable
track — the prosodic features that drove the decision.

The package ships two complementary detectors behind one interface
(Figure 1). *Track A* is an interpretable baseline: energy-based voice
activity detection, 34 prosodic features (F0 statistics, jitter, shimmer,
pause distribution, rhythm variability, syllable-rate dynamics), and a
calibrated LightGBM classifier [@ke2017lightgbm] whose feature
attributions a human reviewer can inspect. *Track B* follows the recipe
validated by @elisha2024: frozen wav2vec 2.0 [@baevski2020wav2vec]
embeddings, chunk-level scoring, and median aggregation. Any subset of
models bundles into a portable median ensemble, and the same artifacts run
through a CLI, a FastAPI service, or Docker. Track A scores a 10-second
answer in ~250 ms on CPU.

![Architecture: audio is preprocessed to 16 kHz mono, scored by both
tracks, and aggregated; every prediction carries an SNR reliability
estimate, and Track A additionally exposes per-feature
attribution.\label{fig:arch}](figures/architecture.pdf)

# Statement of need

Interview anti-cheating is a commercially active but closed field: vendors
market "read-speech scores" with no published method, no benchmark, and no
error bars, while researchers studying speech-style classification
[@elisha2024; @dash2023read] release methods but not deployable tooling.
VoiceVeritas fills that gap for hiring platforms, universities, and
proctoring teams who need an auditable baseline rather than a black box.
It is installable from PyPI, runs without GPUs, and — unusually for this
application area — documents every known limitation in-repo: a
shortcut-learning postmortem of its own benchmark, a noise-robustness
report down to 5 dB SNR, and per-utterance prediction artifacts for every
published number so any AUC can be recomputed without re-training.

The codebase is also a research substrate: data preparation for three
public corpora, a noise-augmentation trainer, calibration, and evaluation
harnesses are reusable modules, and every experiment is reproducible from
configuration files and seeds. An ethics charter restricts use to
assisting human review, never automating rejection.

# Benchmark and domain-shift behaviour

The default benchmark pairs LibriSpeech audiobook reading
[@panayotov2015librispeech] with AMI meeting speech [@carletta2006ami] on
a speaker-disjoint split, reporting EER and FPR@TPR95 because false
accusations are the costliest error. Because such a cross-domain pairing
invites channel shortcuts, the repository also publishes a fully
held-out domain-shift evaluation on corpora the models never saw in
training: SBCSAE conversations [@dubois2000sbcsae] as spontaneous speech
against unseen-speaker LibriSpeech `test-clean` readers. The two tracks
fail in opposite ways — a result we consider more useful than either
headline number:

| Model | Benchmark AUC | Domain-shift AUC |
|---|---|---|
| Track A · prosody + LightGBM | 0.998 | **0.868** |
| Track B · frozen wav2vec 2.0 | 1.000 | **1.000** |
| Median ensemble | 0.997 | **0.895** |

Track A's prosodic features entangle style with channel — roughly half of
genuine conversation crosses its 0.5 threshold in the shifted domain —
whereas Track B's self-supervised embeddings transfer almost perfectly
(3.3% false flags). The median ensemble smooths noise, not systematic
member failure. The practical guidance this implies (trust embeddings
cross-domain; recalibrate thresholds per traffic; keep humans in the
loop) is exactly the kind of statement only an auditable open tool can
make and defend.

# References