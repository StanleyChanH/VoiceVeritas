# Robustness Evaluation / 鲁棒性评测

Stress both tracks under the perturbation family from the project plan
(noise at controlled SNR, telephony band-limiting, sample-rate round trips,
MP3 transcoding) on the speaker-disjoint test split (65 utterances).

Reproduce:

```bash
uv run vv robustness runs/<baseline> --manifest data/manifests/benchmark.jsonl --split test
uv run vv robustness runs/<deep>     --manifest data/manifests/benchmark.jsonl --split test
```

## Track A · prosody + LightGBM

| Condition | AUC-ROC | EER | FPR@TPR95 | ΔAUC |
|---|---|---|---|---|
| clean | **0.978** | 0.04 | 0.04 | — |
| telephony band 300–3400 Hz | **0.975** | 0.05 | 0.00 | −0.004 |
| white noise 20 dB | 0.939 | 0.12 | 0.12 | −0.040 |
| pink noise 20 dB | 0.873 | 0.225 | 1.00 | −0.105 |
| resample 8 kHz round trip | 0.887 | 0.225 | 1.00 | −0.091 |
| white noise 10 dB | 0.644 | 0.675 | 1.00 | −0.335 |
| pink noise 10 dB | 0.604 | 0.750 | 1.00 | −0.374 |
| white/pink noise 5 dB | ~0.52 | ~0.88 | 1.00 | ≈ −0.45 |

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

1. **Neither track survives ≤10 dB SNR.** This is the single most important
   deployment fact in this document. Below ~15 dB SNR the scores are noise
   — treat them as "no signal", never as evidence.

2. **Band-limiting barely matters** (−0.004 / −0.043 AUC). Both tracks are
   safe on telephony-band audio.

3. **The two tracks fail differently.**
   - Track A degrades *gracefully*: its energy VAD floods under noise,
     pause features degenerate, and AUC slides toward 0.5 (chance).
   - Track B *inverts* under white noise: AUC drops **below** 0.5 — under
     noise it systematically scores spontaneous audio as more "read".
     Mechanism (hypothesis, worth verifying): noise power scales with
     whole-signal speech power; read-class audio has a higher speech
     fraction and thus receives more absolute noise energy, so
     "noisiness" becomes a class-correlated artifact the embedding picks
     up — and it points the wrong way.

4. **Pink noise is kinder than white at 20 dB** for Track B (0.976 vs
   0.885): low-frequency-weighted noise corrupts fewer of the spectral
   cues the SSL representation relies on.

## Deployment guidance / 部署建议

- Measure or estimate SNR upstream; **route low-SNR audio (< ~15 dB) to
  human review** instead of the model.
- Prefer denoising front-ends or noise-robust VAD before Track A.
- Never act on a single degraded-condition score; require corroboration
  (multiple segments, other signals, human review — [ethics.md](ethics.md)).

## Roadmap (mitigations)

- [ ] Noise-aware training: add the same perturbation suite as training-time
      augmentation (the project plan's original Phase-4 mitigation).
- [ ] SNR estimator + automatic low-confidence routing.
- [ ] Investigate Track B's white-noise inversion (per-band SNR matching,
      noise-conditioned calibration).
