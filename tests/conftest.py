"""Shared test fixtures.

All tests run on *synthetic* audio so the suite needs no network access, no
datasets, and no GPUs. ``synthesize_speechlike`` produces harmonic-rich voiced
segments with configurable F0 dynamics and pause structure — enough for the
prosody extractor to have a signal to measure, and enough for VAD to find
speech/pause boundaries deterministically.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

SR = 16000


def sine_wave(freq: float, duration: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    """A pure sine, useful for exact F0 checks."""
    t = np.arange(int(duration * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _voiced_phrase(
    rng: np.random.Generator,
    n: int,
    sr: int,
    f0_start: float,
    f0_drift: float,
    f0_jitter: float,
    harmonics: int,
) -> np.ndarray:
    """One voiced segment: random-walk F0, harmonic stack, smooth envelope."""
    # piecewise-constant f0 every ~50 ms with random walk + frame jitter
    frame = max(1, int(0.05 * sr))
    n_frames = int(np.ceil(n / frame))
    steps = rng.normal(0.0, f0_drift, n_frames)
    f0_track = f0_start + np.cumsum(steps)
    f0_track = np.clip(f0_track, 60.0, 350.0)
    jitter = rng.normal(0.0, f0_jitter, n_frames)
    f0_track = f0_track + jitter
    f0_per_sample = np.repeat(f0_track, frame)[:n]

    phase = 2 * np.pi * np.cumsum(f0_per_sample) / sr
    sig = np.zeros(n, dtype=np.float64)
    for k in range(1, harmonics + 1):
        # slight per-harmonic amplitude variation (formant-ish)
        amp_k = (rng.uniform(0.5, 1.0) / k) * (1.0 if k <= 4 else 0.5)
        sig += amp_k * np.sin(k * phase + rng.uniform(0, 2 * np.pi))

    # smooth onset/offset to avoid clicks
    env = np.ones(n)
    edge = min(n // 4, int(0.02 * sr))
    if edge > 0:
        ramp = np.linspace(0, 1, edge)
        env[:edge] = ramp
        env[-edge:] = ramp[::-1]
    return (sig * env).astype(np.float32)


def synthesize_speechlike(
    rng: np.random.Generator,
    duration: float,
    sr: int = SR,
    style: str = "spontaneous",
    f0_mean: float = 140.0,
) -> np.ndarray:
    """Synthesize speech-like audio with style-dependent prosody.

    Styles (stereotyped on purpose — they caricature the acoustic signature
    each style is known for):

    - ``read``: stable F0, metronomic pauses, steady intensity.
    - ``spontaneous``: wandering F0, irregular long pauses, variable intensity.
    """
    if style == "read":
        params = {
            "f0_drift": 2.0,
            "f0_jitter": 0.05,
            "phrase_mean": 2.4,
            "phrase_std": 0.15,
            "pause_mean": 0.28,
            "pause_std": 0.04,
            "amp_var": 0.03,
            "harmonics": 14,
        }
    elif style == "spontaneous":
        params = {
            "f0_drift": 14.0,
            "f0_jitter": 1.2,
            "phrase_mean": 1.6,
            "phrase_std": 0.9,
            "pause_mean": 0.45,
            "pause_std": 0.45,
            "amp_var": 0.35,
            "harmonics": 12,
        }
    else:
        raise ValueError(f"unknown style {style!r}")

    total = int(duration * sr)
    out = np.zeros(total, dtype=np.float32)
    pos = 0
    f0 = f0_mean
    while pos < total:
        phrase_dur = max(0.2, rng.normal(params["phrase_mean"], params["phrase_std"]))
        phrase_dur = min(phrase_dur, (total - pos) / sr)
        if phrase_dur < 0.05:
            break
        n = int(phrase_dur * sr)
        seg = _voiced_phrase(
            rng,
            n,
            sr,
            f0,
            params["f0_drift"],
            params["f0_jitter"],
            params["harmonics"],
        )
        gain = float(np.clip(rng.normal(1.0, params["amp_var"]), 0.25, 1.6))
        out[pos : pos + n] += seg * gain * 0.25  # keep peaks < 1
        pos += n
        # re-anchor F0 for the next phrase (declination & reset)
        f0 = float(np.clip(f0 + rng.normal(0, params["f0_drift"]), 80, 260))

        pause_dur = max(0.03, rng.normal(params["pause_mean"], params["pause_std"]))
        pos += int(pause_dur * sr)
    return out


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def wav_factory(tmp_path: Path) -> Callable[..., Path]:
    """Factory writing synthetic wavs; returns a write(dur, style) -> Path callable."""
    import soundfile as sf

    counter = {"n": 0}

    def write(
        duration: float = 3.0,
        style: str = "spontaneous",
        sr: int = SR,
        seed: int = 7,
    ) -> Path:
        local_rng = np.random.default_rng(seed)
        samples = synthesize_speechlike(local_rng, duration, sr=sr, style=style)
        counter["n"] += 1
        path = tmp_path / f"synth_{style}_{counter['n']:03d}.wav"
        sf.write(str(path), samples, sr, subtype="PCM_16")
        return path

    return write


# ---------------------------------------------------------------------------
# API test fixtures (shared across test modules)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory):
    """Train a tiny prosody model to serve (shared across API tests)."""
    import soundfile as sf

    from voiceveritas.config import TrainConfig
    from voiceveritas.manifest import Utterance, write_manifest
    from voiceveritas.train import train_baseline

    root = tmp_path_factory.mktemp("apicorpus")
    utts = []
    for spk in range(12):
        style = "read" if spk % 2 == 0 else "spontaneous"
        for i in range(2):
            rng = np.random.default_rng(3000 + spk * 10 + i)
            x = synthesize_speechlike(rng, 6.0, style=style)
            p = root / f"spk{spk:02d}_{i}.wav"
            sf.write(str(p), x, SR, subtype="PCM_16")
            utts.append(
                Utterance(
                    f"api_{spk:02d}_{i}",
                    str(p),
                    1 if style == "read" else 0,
                    6.0,
                    f"spk{spk:02d}",
                    "synthetic",
                )
            )
    manifest = root / "m.jsonl"
    write_manifest(manifest, utts)

    train_baseline(
        TrainConfig(
            manifests=[str(manifest)],
            feature_cache_dir=str(root / "cache"),
            lgbm_params={
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.1,
                "num_leaves": 7,
                "min_child_samples": 2,
                "verbosity": -1,
            },
            num_boost_round=30,
            early_stopping_rounds=10,
            calibrate=False,
            run_dir=str(root / "run"),
        )
    )
    return root / "run" / "model.joblib"


@pytest.fixture(scope="session")
def client(trained_model_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from voiceveritas.api import create_app

    app = create_app(str(trained_model_path), max_upload_mb=5)
    return TestClient(app)
