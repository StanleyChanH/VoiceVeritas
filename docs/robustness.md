# Robustness Evaluation / 鲁棒性评测

Stress the models under the perturbation family from the project plan
(noise at controlled SNR, telephony band-limiting, sample-rate round trips,
MP3 transcoding) on the speaker-disjoint test split (65 utterances).

Reproduce (augmentation is on by default in `configs/baseline.yaml`):

```bash
uv run vv train --config configs/baseline.yaml        # noise-augmented Track A
uv run vv robustness runs/<run> --manifest data/manifests/benchmark.jsonl --split test
```

## Track A · prosody + LightGBM

**Noise-aware training** (train-split augmentation, on by default) vs the
original unaugmented baseline:

| Condition | AUC (augmented) | AUC (original) | EER (aug) | FPR@TPR95 (aug) |
|---|---|---|---|---|
| clean | **0.998** | 0.978 | 0.04 | 0.04 |
| telephony band 300–3400 Hz | **0.990** | 0.975 | 0.12 | 0.12 |
| resample 8 kHz round trip | **0.988** | 0.887 | 0.05 | 0.04 |
| white noise 20 dB | **0.977** | 0.939 | 0.04 | 0.04 |
| pink noise 20 dB | **0.978** | 0.873 | 0.05 | 0.04 |
| white noise 10 dB | **0.965** | 0.644 | 0.125 | 0.20 |
| pink noise 10 dB | **0.931** | 0.604 | 0.16 | 0.32 |
| white noise 5 dB | **0.941** | 0.523 | 0.20 | 0.32 |
| pink noise 5 dB | **0.800** | 0.529 | 0.275 | 0.84 |

Augmentation (white/pink noise at 10–25 dB SNR + ±3 dB gain on training
audio only) transforms the picture: the 10 dB collapse is gone, 5 dB white
noise stays usable, and even *clean* performance improves (0.978 → 0.998)
— the perturbations act as a regularizer against channel overfitting.
Pink 5 dB remains the hardest condition (low-frequency noise corrupts the
energy VAD's pause segmentation).

## Track B · frozen wav2vec2-base + MLP

The original (unaugmented) head **inverted** under white noise — AUC
0.137 at 10 dB, i.e. it scored spontaneous audio as *more* read.
Noise-augmented embeddings (`augment: true` in `configs/deep.yaml`)
fixed it completely:

| Condition | AUC (augmented) | AUC (original) | EER (aug) | FPR@TPR95 (aug) |
|---|---|---|---|---|
| clean | 0.971 | **1.000** | 0.10 | 0.12 |
| white noise 20 dB | **0.989** | 0.885 | 0.08 | 0.08 |
| white noise 10 dB | **0.998** | 0.137 (inverted) | 0.025 | 0.00 |
| pink noise 10 dB | **0.988** | 0.169 | 0.04 | 0.04 |
| white noise 5 dB | **0.980** | 0.081 | 0.125 | 0.12 |
| pink noise 5 dB | **0.947** | 0.170 | 0.12 | 0.12 |
| telephony band 300–3400 Hz | **0.986** | 0.957 | 0.15 | 0.08 |
| resample 8 kHz round trip | 0.929 | 0.922 | 0.175 | 0.16 |

The trade is ~3 points of clean AUC (1.000 → 0.971) for near-immunity to
noise — under 10 dB white noise the augmented deep model (0.998) is now
the strongest single model in the project.

## Ensembles · median aggregation

`vv ensemble m1.joblib m2.joblib -o ens.joblib` bundles models into a
portable median-ensemble artifact. Median (not mean) so one failing
member cannot drag the aggregate. Findings from our three trained models:

| Condition | prosody-aug | deep-aug | **ensemble(2)** | ensemble(3, incl. ASR) |
|---|---|---|---|---|
| clean | 0.998 | 0.971 | **0.997** | 0.999 |
| white noise 10 dB | 0.965 | 0.998 | **0.994** | 0.926 |
| pink noise 10 dB | 0.931 | 0.988 | **0.986** | 0.987 |
| white noise 5 dB | 0.941 | 0.980 | 0.944 | — |

**Lesson (honest)**: the 3-member ensemble with the *un-augmented* ASR
model dropped to 0.926 at white 10 dB — median cannot outperform a
majority of noise-fragile members. **Only ensemble noise-robust
members**; the recommended deployable is `ensemble(prosody-aug,
deep-aug)`.

(MP3 conditions skipped in this run: no ffmpeg on the benchmark machine;
the harness supports them where ffmpeg is present.)

## Reading the results / 如何解读

1. **Noise-aware Track A now survives moderate noise** (AUC ≥ 0.93 down to
   10 dB, 0.94 at white 5 dB). Pink 5 dB is the remaining weak spot.

2. **Band-limiting is essentially free** (−0.009 AUC augmented).

3. **Track B's white-noise inversion is fixed by augmentation** (0.137 →
   0.998 at 10 dB). Mechanism (hypothesis): noise power scaled with
   whole-signal speech power made "noisiness" a class-correlated artifact;
   training on noisy variants breaks that correlation.

## Deployment guidance / 部署建议

- Every prediction now carries an **`estimated_snr_db`** (VAD-frame
  estimator; CLI prints it, the API returns it with a `warning` field).
  Audio below the **15 dB reliability floor** should be routed to human
  review — the warning text says so explicitly.
- Prefer denoising front-ends for very noisy capture; pink-noise-heavy
  environments remain hard for Track A.
- Never act on a single degraded-condition score; require corroboration
  (multiple segments, other signals, human review — [ethics.md](ethics.md)).

## Roadmap (mitigations)

- [x] Noise-aware training (augmentation) — shipped, on by default
- [x] SNR estimator + low-confidence routing — shipped in prediction paths
- [x] Augmented Track B embeddings — inversion fixed (0.137 → 0.998 @10 dB)
- [x] Median ensembles (`vv ensemble`) — deployable: ensemble(prosody-aug, deep-aug)
- [ ] Noise-augment the ASR track (lexical features under noise)
