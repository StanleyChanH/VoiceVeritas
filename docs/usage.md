# Usage Guide / 使用指南

## Installation

```bash
# with uv (recommended)
uv sync --extra api --extra deep

# or plain pip
pip install -e ".[api,deep]"
```

Extras: `deep` (torch/transformers for Track B), `api` (FastAPI service),
`asr` (optional transcript-alignment features), `dev` (lint/test tooling).
For mp3/m4a support install **ffmpeg** and make sure it is on `PATH`.

## Scoring one file

```bash
uv run vv predict runs/20251113_142233_baseline/model.joblib answer.wav
```

```
model            : prosody_lgbm
duration         : 27.9s
P(read)          : 0.812
verdict@0.5      : READ
  [   0.00 -   27.90s]  score=0.812
top contributors :
  speechseg_mean          +0.92
  pause_total_ratio       +0.55
  ...
```

Add `--json` for machine-readable output. The 0.5 verdict line is a
*reference point*, not a recommended policy threshold — see
[ethics.md](ethics.md) on choosing an operating point by false-positive
budget.

## Python API

```python
from voiceveritas.audio import load_audio
from voiceveritas.models import load_model

model = load_model("runs/<run>/model.joblib")
audio = load_audio("answer.mp3")
result = model.predict(audio.samples, audio.sample_rate)
print(result.read_probability, [s.score for s in result.segments])
```

## Preparing benchmark data

See [data.md](data.md) for the full picture:

```bash
uv run vv prepare librispeech --download          # ~340 MB
uv run vv prepare ami --download                  # ~16 sessions, ~400 MB
cat data/manifests/librispeech.jsonl data/manifests/ami.jsonl \
    > data/manifests/benchmark.jsonl
```

## Training

**Track A — interpretable baseline (minutes on CPU):**

```bash
uv run vv train --config configs/baseline.yaml
```

**Track B — frozen wav2vec 2.0 + MLP head (see [benchmark.md](benchmark.md)):**

```bash
uv run vv train-deep --config configs/deep.yaml
```

Both write a self-contained run directory:

```
runs/<timestamp>_<name>/
├── config.yaml            # exact resolved configuration
├── metrics.json           # val/test metrics incl. EER & FPR@TPR95
├── model.joblib           # portable model artifact
└── feature_importance.csv # Track A only
```

Splits are **speaker-disjoint** (a speaker never spans train/val/test), the
single most important guard against over-optimistic read/spontaneous
benchmarks.

## Evaluating on a manifest

```bash
uv run vv eval --run runs/<run> --manifest data/manifests/benchmark.jsonl
```

## Serving the API

```bash
uv run vv serve runs/<run>            # http://127.0.0.1:8000/docs
```

Or Docker:

```bash
docker build -t voiceveritas .
docker run -p 8000:8000 -v "$PWD/runs/<run>:/models" voiceveritas
```

Latency benchmarking:

```bash
uv run python scripts/bench_api.py --url http://127.0.0.1:8000 --n 20
```
