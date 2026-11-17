# Domain-shift evaluation / 域偏移评测

> The honest test: score the shipped models on speech **both sides of which
> they have never seen in training** — and publish whatever happens.

## The question this answers / 这个评测回答什么问题

The default [benchmark](benchmark.md) is cross-domain by construction:
*read* = studio audiobooks (LibriSpeech dev-clean), *spontaneous* = meeting
headset mixes (AMI). A classifier could reach high AUC there by detecting
**channel and register** (room reverb, mic distance, recording genre) rather
than reading style. This evaluation asks the uncomfortable question directly:

> If we swap the spontaneous corpus for a *completely different* one the
> models never saw — naturally-occurring conversations (SBCSAE) — and the
> read speech for *unseen speakers* (LibriSpeech test-clean), how much of
> the reported performance survives?

## Protocol / 协议

| Side | Corpus | Selection | Units |
|---|---|---|---|
| Read (1) | LibriSpeech **test-clean** (unseen speakers) | 6 speakers × up to 20 units | 117 |
| Spontaneous (0) | **SBCSAE** (OpenSLR 155, unseen corpus) | 12 conversations (every 5th of 60), 10 VAD chunks each | 120 |

- Neither corpus appears in training; the script **fails hard** if a
  test-clean speaker overlaps the training benchmark's speakers.
- Everything is `split=test` (SBCSAE is eval-only by policy — [data.md](data.md)).
- 237 utterances, 1.39 h, seed 42, VAD chunking identical to AMI's recipe.

Reproduce end-to-end:

```bash
uv run vv prepare sbcsae --download                # 6.1 GB, OpenSLR mirrors
# LibriSpeech test-clean: download + extract under data/raw/librispeech
uv run python scripts/build_in_domain.py --libri-raw data/raw/librispeech/LibriSpeech/test-clean
uv run vv eval --run runs/<baseline_augmented> --manifest data/manifests/in_domain.jsonl \
    --split test --dump-predictions docs/assets/predictions/in_domain_prosody_aug.csv
```

## Results / 结果

Same artifacts, two test beds — the in-family benchmark vs this fully
unseen pairing:

| Model | Benchmark AUC (EER, FPR@95) | **Domain-shift AUC (EER, FPR@95)** |
|---|---|---|
| Track A · prosody+LightGBM (augmented) | 0.998 (0.04, 0.04) | **0.868 (0.256, 0.358)** |
| Track B · frozen wav2vec2 + MLP | 1.000 (0.00, 0.00) | **1.000 (0.009, 0.00)** |
| Ensemble · median(A, B) | 0.997 (0.075, 0.04) | **0.895 (0.128, 0.158)** |

Per-utterance scores: [`docs/assets/predictions/`](assets/predictions/)
(`in_domain_*.csv`).

## What the numbers say / 数字意味着什么

**The two tracks fail in opposite ways — and that is the most useful result
in this repo.**

**Track A degrades: 0.998 → 0.868.** A real style signal remains, but
roughly a quarter of the headline performance does not survive the domain
shift, and the failure mode matters more than the AUC:

- **The read side transfers.** 99% of unseen test-clean readers are flagged
  (median score 1.00, p10 = 0.93). Reading aloud looks like reading aloud.
- **The spontaneous side splits by conversation.** Flag rates range from
  ~0% (`sbc016`, `sbc046`) to ~85% (`sbc001`, `sbc036`) at the default 0.5
  threshold — 49% of genuinely spontaneous conversation is flagged overall:

| conversation | mean P(read) | mean est. SNR dB |
|---|---|---|
| sbc016 | 0.03 | 20.0 |
| sbc046 | 0.03 | 11.4 |
| sbc011 | 0.09 | 19.3 |
| sbc031 | 0.24 | 11.6 |
| sbc021 | 0.34 | 8.0 |
| sbc056 | 0.49 | 18.1 |
| sbc026 | 0.55 | 13.3 |
| sbc041 | 0.70 | 24.7 |
| sbc051 | 0.73 | 19.1 |
| sbc036 | 0.83 | 25.7 |
| sbc006 | 0.82 | 11.8 |
| sbc001 | 0.86 | 17.2 |

- **Noise does not explain the split** — the high-scoring conversations
  average *better* SNR (19.7 dB) than the low-scoring ones (14.7 dB). The
  likely drivers are conversation genre (long narrative/monologue stretches
  vs rapid multi-party exchange) and channel/recording setting — exactly
  the style-adjacent confounds the model absorbed from AMI+LibriSpeech.

**Track B holds: 1.000 → 0.9996.** The frozen wav2vec 2.0 track transfers
almost perfectly: 99% of unseen readers flagged (mean 0.83), only **3.3%**
of spontaneous conversation flagged (median 0.04), and — unlike Track A —
*every* conversation stays low (per-conversation means 0.005–0.39, no
bimodal split). The SSL-embedding style signal is not an AMI channel
artifact.

This is exactly the pattern the literature predicts: handcrafted prosodic
features entangle style with channel, while self-supervised embeddings
carry the style cue itself.

**The median ensemble does not save you: 0.895.** Where Track A is wrong on
a whole subpopulation (half the SBCSAE conversations), the median of
(prosody≈0.9, wav2vec≈0.05) lands mid-scale — better than Track A alone,
strictly worse than Track B alone. Ensembles smooth *noise*, not
*systematic* member failure. Practical consequences:

- **Cross-domain deployments should trust Track B, not Track A (nor the
  median ensemble)** — Track A's role is explaining decisions, not
  surviving domain shift.
- **The 0.5 threshold trained on one domain is void in another** for Track
  A. Recalibrate on your own traffic (per-side score distributions ship in
  the CSVs), deploy behind an FPR budget (`metrics.threshold_at_fpr`), and
  route borderline scores to human review — never auto-reject on them
  ([ethics.md](ethics.md)).

## What this still does not answer / 仍未回答的问题

- **Read-side channel shift**: the read side is still LibriSpeech-family
  studio audio. A second read corpus with a different channel (e.g. Common
  Voice mic speech) would close the loop — contributions welcome.
- **Same-speaker gold standard**: the cleanest test — the *same person*
  recorded reading and speaking spontaneously on the same mic — needs a
  purpose-collected corpus (consent-first protocol in [ethics.md](ethics.md));
  we publish this protocol so any lab can contribute one.
