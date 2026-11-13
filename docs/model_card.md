# Model Card — VoiceVeritas Read-Speech Detector

Following the spirit of *Model Cards for Model Reporting* (Mitchell et al.,
2019). Applies to artifacts produced by this repository's default recipes.

## Model details

- **Task**: estimate P(audio is read/recited rather than spontaneously spoken)
- **Track A (`prosody_lgbm`)**: 34 handcrafted prosodic/acoustic features
  (F0 statistics & slope, jitter, shimmer, HNR, pause/speech-run
  distributions, nPVI rhythm, syllable-rate variability, spectral shape) →
  LightGBM + isotonic calibration. Fully interpretable (TreeSHAP
  contributions returned per prediction).
- **Track B (`wav2vec_head`)**: frozen `facebook/wav2vec2-base` encoder,
  time mean+std pooled embeddings (1536-d) → 2-layer MLP head.
- **Version**: 0.1.0 (see `model.joblib` metadata for per-run provenance).

## Intended use

- Assisting **human** review of recorded interviews by surfacing segments
  with scripted-speech characteristics.
- Research on speech style and paralinguistics.

## Out-of-scope / prohibited use

- Automated rejection of candidates without human review.
- Covert analysis of people not informed of its use.
- As standalone "proof of cheating" (a high score has benign explanations).

## Training data

LibriSpeech dev-clean (read; CC BY 4.0) + AMI Meeting Corpus (spontaneous;
CC BY 4.0). No candidate data. Details and licenses: [data.md](data.md).

## Performance

See [benchmark.md](benchmark.md) for full numbers
(Track A test AUC 0.978 / EER 4% / FPR@TPR95 4%; Track B test AUC 1.000 on
the same speaker-disjoint split).

## Known limitations & biases

- **Cross-domain gap**: trained on audiobooks vs meeting speech; single-mic
  interview audio may shift scores in either direction. Recalibrate before
  relying on absolute probabilities.
- **Accent / non-native speakers**: untested; disfluency patterns differ
  across L2 speakers and the model may mistake L2 prosody for scriptedness.
  This is the primary fairness risk — evaluate on your population before
  deployment.
- **Rehearsed-but-honest answers** are stylistically read-like; the model
  measures style, not honesty.
- **Duration shortcut (historical)**: an early benchmark leaked utterance
  length; fixed and regression-tested (see benchmark.md). Keep the guard
  when adding features or datasets.
- **English only, by design**: speech-style cues are language-specific;
  the project scopes to English interview speech (see README).

## Ethical considerations & deployment guidance

Read [ethics.md](ethics.md). In short: human-in-the-loop, informed
candidates, FPR-budget thresholds, appeal path, no audio retention.

## Caveats on artifacts

Model files (`.joblib`) pickle code-adjacent objects — only load artifacts
you trained yourself or obtained from a trusted source.
