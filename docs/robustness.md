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

| Condition | AUC-ROC | EER | FPR@TPR95 | ΔAUC |
|---|---|---|---|---|
| clean | **1.000** | 0.00 | 0.00 | — |
| pink noise 20 dB | **0.976** | 0.05 | 0.04 | −0.024 |
| telephony band 300–3400 Hz | 0.957 | 0.04 | 0.04 | −0.043 |
| resample 8 kHz round trip | 0.922 | 0.225 | 0.20 | −0.078 |
| white noise 20 dB | 0.885 | 0.16 | 0.36 | −0.115 |
| white noise 10 dB | **0.137** | 0.76 | 1.00 | −0.863 |
| pink noise 10 dB | 0.169 | 0.80 | 1.00 | −0.831 |
| white/pink noise 5 dB | ~0.1–0.2 | ~0.85 | 1.00 | ≈ −0.9 |

(MP3 conditions skipped in this run: no ffmpeg on the benchmark machine;
the harness supports them where ffmpeg is present.)

## Reading the results / 如何解读

1. **Noise-aware Track A now survives moderate noise** (AUC ≥ 0.93 down to
   10 dB, 0.94 at white 5 dB). Pink 5 dB is the remaining weak spot.

2. **Band-limiting is essentially free** (−0.009 AUC augmented).

3. **Track B (unaugmented) still fails differently**: it *inverts* under
   white noise (AUC below 0.5 — it scores spontaneous audio as more
   "read"). Mechanism (hypothesis): noise power scales with whole-signal
   speech power; read-class audio has a higher speech fraction, receives
   more absolute noise energy, and "noisiness" becomes a class-correlated
   artifact pointing the wrong way. Applying the same augmentation recipe
   to Track B (re-extracting embeddings of noisy audio) is future work.

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
- [ ] Augment Track B embeddings (re-extract noisy variants)
- [ ] Investigate Track B's white-noise inversion (per-band SNR matching,
      noise-conditioned calibration)
