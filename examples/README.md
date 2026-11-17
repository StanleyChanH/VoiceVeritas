# Example clips

Two tiny **synthetic** clips for smoke-testing the pipeline in seconds — no
datasets needed:

| File | What it caricatures | Expected |
|---|---|---|
| `example_read.wav` | read speech: steady F0, metronomic pauses, even intensity | high P(read) |
| `example_spontaneous.wav` | spontaneous speech: wandering F0, irregular pauses | low P(read) |

⚠️ These are **not real speech** — they are deterministic caricatures from
`voiceveritas.utils.synth`, good for checking that install → predict works
end-to-end. Do **not** use them to judge model quality; see
[docs/benchmark.md](../docs/benchmark.md) for real numbers and
[docs/ethics.md](../docs/ethics.md) before any operational use.

```bash
# score them right after installing (model is fetched from GitHub releases)
vv download-model                                  # prints the cached model path
vv predict "$(vv download-model)" examples/example_spontaneous.wav

# regenerate with different seeds/durations
uv run python scripts/make_examples.py --duration 12
```
