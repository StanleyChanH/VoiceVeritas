---
title: VoiceVeritas
emoji: 🎙️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: true
license: apache-2.0
short_description: Read-speech detection for interview anti-cheating (demo)
---

# VoiceVeritas demo

Upload a short English interview answer — get **P(read)**: the probability it
was read/recited from a script rather than spoken spontaneously, plus the
prosodic cues that drove the decision and an SNR reliability warning.

- Code & docs: [github.com/StanleyChanH/VoiceVeritas](https://github.com/StanleyChanH/VoiceVeritas)
- Benchmarks (cross-domain, with caveats): [docs/benchmark.md](https://github.com/StanleyChanH/VoiceVeritas/blob/main/docs/benchmark.md)
- **Read before operational use**: [ethics charter](https://github.com/StanleyChanH/VoiceVeritas/blob/main/docs/ethics.md)

The two built-in example clips are **synthetic caricatures** (steady F0 and
metronomic pauses vs wandering F0 and irregular pauses), not real speech —
they exist so you can see the interface move, not to judge model quality.

**Privacy:** uploads are decoded and scored in memory; nothing is stored or
logged. **Scope:** English speech only; both model tracks are unreliable
below ~10 dB SNR.
