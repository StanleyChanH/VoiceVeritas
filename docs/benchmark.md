# Benchmark Results / 基准结果

Reproduce with: `vv prepare librispeech --download && vv prepare ami --download`
then `vv train --config configs/baseline.yaml` and
`vv train-deep --config configs/deep.yaml` (same seeds as `configs/*.yaml`).

## Setup

| | |
|---|---|
| Read speech (label 1) | LibriSpeech dev-clean — 20 speakers, 399 answer units, ~2.5 h |
| Spontaneous (label 0) | AMI meetings — 12 sessions, 300 chunks, ~1.5 h |
| Split | **speaker/session-disjoint**, seeded (42), 70/15/15 |
| Unit lengths | randomized 12–25 s in *both* classes (see "the shortcut" below) |
| Test set | 65 utterances, unseen speakers/sessions |

## Results

| Model | Test AUC-ROC | AUC-PR | EER | FPR@TPR95 | Brier |
|---|---|---|---|---|---|
| Track A: prosody + LightGBM (isotonic) | **0.978** | 0.974 | 0.04 | 0.04 | 0.019 |
| Track B: frozen wav2vec2-base + MLP | **1.000** | 1.000 | 0.00 | 0.00 | — |

Track A's top features by gain: `zcr_std`, `zcr_mean`, `speechseg_mean`
(speech-run duration — the literature's #1 cue), `pause_total_ratio`,
`spec_centroid_std`, `f0_p10`, `spec_flatness_mean`, `hnr_db`.

## The shortcut we had to kill / 我们必须消灭的捷径

An earlier run reported AUC 0.98 — but feature importance showed
`duration_log` with gain ≈ 409 and every other feature ≈ 0. The benchmark
had leaked *utterance length*: AMI chunks all landed on exactly the 25 s
cap while LibriSpeech units ran 24–33 s. The classifier had learned
"long ⇒ read", a collection artifact that would collapse on real traffic.

Fixes (both verified by regression tests):

1. `duration_log` removed from the feature set
   (`tests/test_prosody.py::test_no_duration_shortcut`);
2. both corpora are now cut to randomized 12–25 s lengths
   (`tests/test_data_prep.py` asserts chunk lengths are not all identical).

The honest Track-A numbers above are *after* these fixes. We consider this
the most instructive artifact of the whole project: **read the feature
importance before you celebrate the metric.**

## Caveats (please read before citing) / 引用前请阅读

- **Cross-domain by construction**: read = studio audiobooks, spontaneous =
  meeting-room headset mixes. Some of the signal — especially ZCR/spectral
  features for Track A, and likely part of Track B's edge — reflects
  *channel and register*, not pure reading style. Expect a large drop on
  in-the-wild interview audio from a single microphone.
- **AMI speaker granularity**: sessions (not individual humans) are the
  split unit; see [data.md](data.md).
- Track B's perfect 1.0 on 65 test utterances is encouraging but small;
  treat as upper bound, not deployment expectation.
- English only. Non-native speakers and accented speech are untested —
  the known-bias risk is documented in [ethics.md](ethics.md).
- Before any deployment use: recalibrate on your own labeled traffic and
  pick thresholds by FPR budget (`metrics.threshold_at_fpr`).

## Latency (Track A, API service)

`scripts/bench_api.py`, Ryzen 7 8745HS (CPU-only), 10 s wav, 15 sequential
requests after one warm-up request (the first request after process start
pays ~5 s of lazy imports):

| metric | p50 | p95 |
|---|---|---|
| end-to-end | **243 ms** | 278 ms |
| inference only | 217 ms | 254 ms |

Comfortably within the project target of "10 s audio → ≤ 500 ms" on CPU.
Track B (wav2vec2) is ~30–50x slower per second of audio on CPU — use it
for offline/asynchronous review, keep Track A for interactive traffic.
