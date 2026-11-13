"""Training-time acoustic augmentation.

The original project plan's Phase-4 mitigation for mic/channel diversity,
now backed by our own robustness measurements (docs/robustness.md):
augmenting training audio with white/pink noise at 10-25 dB SNR (plus mild
gain jitter) teaches the model decision boundaries that survive noisy
capture. Validation/test splits are never augmented — degradation must be
measured against reality.
"""

from __future__ import annotations

import hashlib

import numpy as np

from voiceveritas.eval.robustness import add_pink_noise, add_white_noise, apply_gain

DEFAULT_SNR_RANGE_DB = (10.0, 25.0)
DEFAULT_GAIN_RANGE_DB = (-3.0, 3.0)


def augment_for_training(
    samples: np.ndarray,
    sr: int,
    key: str,
    seed: int = 42,
    snr_range_db: tuple[float, float] = DEFAULT_SNR_RANGE_DB,
    gain_range_db: tuple[float, float] = DEFAULT_GAIN_RANGE_DB,
) -> np.ndarray:
    """Deterministic per-utterance augmentation (seeded by utterance key).

    Random choice of white/pink noise at a random SNR within the range,
    plus a mild gain jitter. The same utterance always augments the same
    way for a given seed, keeping runs reproducible and cacheable.
    """
    digest = hashlib.sha1(f"{key}|{seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))

    x = np.asarray(samples, dtype=np.float32)
    snr_db = float(rng.uniform(*snr_range_db))
    if bool(rng.integers(0, 2)):
        x = add_white_noise(x, sr, snr_db=snr_db, rng=rng)
    else:
        x = add_pink_noise(x, sr, snr_db=snr_db, rng=rng)
    return apply_gain(x, sr, float(rng.uniform(*gain_range_db)))
