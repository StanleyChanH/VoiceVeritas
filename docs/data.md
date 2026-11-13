# Datasets & Licenses / 数据集与许可

VoiceVeritas ships **preparation code, never audio**. All corpora below are
downloaded from their original sources; each section documents the license
you accept by downloading.

VoiceVeritas 只分发**准备代码**，永不分发音频。下列语料均从原始来源下载，各节注明下载即接受的许可协议。

## Benchmark recipe (default) / 默认基准配方

| Role | Dataset | Style / label | Volume | License |
|---|---|---|---|---|
| Read speech | [LibriSpeech](https://www.openslr.org/12) `dev-clean` | read (1) | 20 speakers → 360 units, ~2.8 h | CC BY 4.0 |
| Spontaneous | [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/) (16 sessions) | spontaneous (0) | 12 sessions → 300 chunks, ~2.1 h | CC BY 4.0 |

```bash
uv run vv prepare librispeech --download
uv run vv prepare ami --download
cat data/manifests/librispeech.jsonl data/manifests/ami.jsonl > data/manifests/benchmark.jsonl
```

### Preparation details

- **LibriSpeech**: single sentences (2–15 s) are concatenated per chapter
  with a 0.35 s inter-sentence pause into 8–25 s "answer units", closer to
  real interview-answer length, with natural read pauses intact.
- **AMI**: each ~20-min session (`Mix-Headset` mixdown) is chunked by the
  project's own energy VAD into 8–25 s segments. ⚠️ Each session contains
  several speakers; the session id is used as the speaker unit for
  speaker-disjoint splitting. This prevents room/channel leakage between
  splits (the dominant short-range leakage), but individual human speakers
  can straddle a split — documented as a known approximation.

This benchmark is deliberately *cross-domain* (audiobook read speech vs.
meeting spontaneous speech). High scores here do **not** transfer 1:1 to
your interview traffic — recalibrate on your own data before deployment
(see [model_card.md](model_card.md)).

## Additional datasets (community contributions welcome)

| Dataset | Style | License | Notes |
|---|---|---|---|
| [SBCSAE](http://www.openslr.org/155) (OpenSLR 155) | spontaneous | CC BY-ND 3.0 US | 60 conversations, 6.1 GB; no-derivative clause complicates derived-artifact redistribution |
| [CORAAL](https://oraal.github.io/coraal) | spontaneous interviews | CC BY-NC-SA 4.0 | non-commercial — do not train the default shipped model on it |
| [Common Voice](https://commonvoice.mozilla.org) | read | CC0 | huge read-speech augmentation pool |
| GigaSpeech / WenetSpeech / MagicData-RAMC | mixed / Chinese | various (application) | roadmap: Chinese spontaneous benchmark |

## Adding your own data

Write a JSONL manifest (one object per line):

```json
{"utt_id": "abc123", "path": "/abs/or/rel.wav", "label": 1,
 "duration": 14.2, "speaker_id": "candidate_9", "dataset": "mycorp", "split": "train"}
```

- `label`: `1` = read/recited, `0` = spontaneous
- `speaker_id`: **must** be consistent per person — speaker-disjoint
  splitting depends on it
- `split` is optional (assign `train` everywhere and let the trainer split)

If your data is real candidate audio, collect it **with informed consent**
(see [ethics.md](ethics.md)).
