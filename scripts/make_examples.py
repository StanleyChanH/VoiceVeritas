"""Generate the synthetic example clips in examples/ (offline, deterministic).

Usage:
    uv run python scripts/make_examples.py [--out-dir examples] [--duration 12]

The clips are *caricatures* produced by ``voiceveritas.utils.synth`` — steady
F0 and metronomic pauses for ``read``, wandering F0 and irregular pauses for
``spontaneous``. They let anyone exercise the full predict pipeline in
seconds without downloading datasets; they are not real speech and must not
be used for training or benchmarking.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from voiceveritas.utils.synth import SR, synthesize_speechlike

CLIPS: dict[str, tuple[str, int]] = {
    # filename -> (synth style, rng seed) — seeds fixed so clips are reproducible
    "example_read.wav": ("read", 20251116),
    "example_spontaneous.wav": ("spontaneous", 20251117),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="examples", help="output directory")
    parser.add_argument("--duration", type=float, default=12.0, help="clip length in seconds")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (style, seed) in CLIPS.items():
        rng = np.random.default_rng(seed)
        samples = synthesize_speechlike(rng, args.duration, sr=SR, style=style)
        path = out_dir / name
        sf.write(str(path), samples, SR, subtype="PCM_16")
        print(f"wrote {path} ({style}, {args.duration:.0f}s, seed={seed})")


if __name__ == "__main__":
    main()
