# Changelog

All notable changes to VoiceVeritas are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

_Nothing yet._

## [0.2.0] — 2026-11-17

### Added

- **SBCSAE held-out domain-shift evaluation** (`vv prepare sbcsae`): OpenSLR 155
  conversations as an eval-only spontaneous corpus (`split=test` by design),
  scored against unseen-speaker LibriSpeech `test-clean` read speech. The
  published results split the tracks: Track A 0.998 → **0.868** (prosodic
  features entangle channel; ~half of genuine conversation crosses its 0.5
  threshold), Track B holds at **1.000** (SSL embeddings transfer), the median
  ensemble lands between at 0.895 — ensembles smooth noise, not systematic
  member failure. Full protocol, per-conversation analysis, and prediction
  artifacts: [docs/in_domain.md](docs/in_domain.md).
- **Auditable per-utterance predictions**: `vv eval --split <s> --dump-predictions out.csv`
  writes one row per test utterance (score, label, SNR, speaker). The benchmark
  test-set predictions of all three released models ship in
  [docs/assets/predictions/](docs/assets/predictions/) so anyone can recompute
  the published AUC/EER numbers without re-training.
- CI: coverage reporting via Codecov; Python 3.13 in the test matrix;
  Windows and macOS runners alongside Linux.
- `CHANGELOG.md` (this file) and release-drafter automation.

### Fixed

- `vv eval`/`vv serve`/`vv robustness` now accept any `.joblib` artifact path
  directly (e.g. ensemble bundles); previously only run *directories* and
  literal `model.joblib` paths worked.

## [0.1.5] — 2026-11-17

### Added

- `pip install voiceveritas` — first PyPI release (trusted publishing).
- `vv download-model` — one command fetches the pretrained Track-A artifact
  (~140 KB) into the user cache.
- Deployable Hugging Face Space demo (Gradio, Track A) in `demo/hf-space/`.
- Bundled synthetic examples (`examples/`) for instant post-install smoke
  testing.

## [0.1.4] — 2026-11-16

### Added

- Character-level forced alignment capability (WhisperX recipe, zero extra
  dependencies).

### Evaluated & rejected

- Alignment-density features exposed a **coverage leak** (CTC decode density
  encodes corpus identity, not speech style: 157 vs 67 words/min) — capability
  shipped, feature rejected. Details in [docs/benchmark.md](docs/benchmark.md).

## [0.1.3] — 2026-11-13

### Added

- Noise-augmented Track-B (wav2vec2) training: white-noise 10 dB AUC
  **0.137 (inverted) → 0.998**; pink noise 0.169 → 0.988.
- Median ensembles (`vv ensemble m1 m2 -o ens.joblib`): portable bundles with
  member payloads inline — the recommended deployable.

## [0.1.2] — 2026-11-13

### Changed

- **Noise-aware training on by default** (white/pink noise 10–25 dB + gain
  jitter on the train split): Track A white-noise 10 dB AUC 0.644 → 0.965,
  clean 0.978 → 0.998.

## [0.1.1] — 2026-11-13

### Added

- Robustness suite (`vv robustness`): white/pink noise, telephony band,
  8 kHz round trips, MP3 transcoding — full report in
  [docs/robustness.md](docs/robustness.md).
- Pretrained Track-A artifact attached to the release.

## [0.1.0] — 2026-11-13

### Added

- Dual-track read-speech detection: Track A (34 prosodic features + LightGBM,
  interpretable) and Track B (frozen wav2vec2 embeddings + MLP).
- Speaker-disjoint cross-domain benchmark (LibriSpeech read vs AMI
  spontaneous), CLI (`vv train/predict/eval`), FastAPI service, Docker.
- Ethics charter, model card, benchmark & data documentation.

[Unreleased]: https://github.com/StanleyChanH/VoiceVeritas/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/StanleyChanH/VoiceVeritas/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/StanleyChanH/VoiceVeritas/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/StanleyChanH/VoiceVeritas/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/StanleyChanH/VoiceVeritas/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/StanleyChanH/VoiceVeritas/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/StanleyChanH/VoiceVeritas/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/StanleyChanH/VoiceVeritas/releases/tag/v0.1.0
