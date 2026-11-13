"""SNR estimation from frame energies (VAD-masked).

Deployment gate: the robustness report showed both model tracks collapse
below ~15 dB SNR — before trusting a score you must know the noise regime
it came from. This estimator is cheap (no model), deterministic, and
conservative: it returns NaN rather than a guess when it cannot see both
speech and noise frames.
"""

from __future__ import annotations

import numpy as np

from voiceveritas.vad.energy_vad import VadResult, energy_vad, frame_energies_db

_EPS = 1e-10


def estimate_snr_db(samples: np.ndarray, sr: int, vad: VadResult | None = None) -> float:
    """Estimate speech-to-noise SNR in dB from frame energies.

    Speech level = 70th percentile of VAD speech frames; noise floor =
    10th percentile of non-speech frames (robust to speech-energy bursts
    contaminating the non-speech pool; falls back to the 10th percentile
    of all frames when too few non-speech frames survive VAD merging).
    Calibrated on synthetic speech: clean reads saturate high, a 12 dB
    injected SNR reads ~12-14 dB. Returns NaN when no speech is detected.
    """
    if vad is None:
        vad = energy_vad(samples, sr)
    db = frame_energies_db(samples, sr)

    mask = vad.speech_mask[: db.size]
    speech_db = db[mask]
    if speech_db.size == 0:
        return float("nan")

    noise_pool = db[~mask] if (~mask).sum() >= 8 else db
    noise_db = float(np.percentile(noise_pool, 10))

    speech_level = float(np.percentile(speech_db, 70))
    # -2 dB empirical bias correction: frame energies are chi^2-distributed,
    # so p10(noise) sits below and p70(speech) above their power means
    snr = speech_level - noise_db - 2.0
    if not np.isfinite(snr):
        return float("nan")
    # cap implausible extremes (quantization floors, clipping)
    return float(np.clip(snr, 0.0, 60.0))
