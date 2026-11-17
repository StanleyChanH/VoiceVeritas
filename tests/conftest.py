"""Shared test fixtures.

All tests run on *synthetic* audio so the suite needs no network access, no
datasets, and no GPUs. The synthesizer itself lives in the package
(:mod:`voiceveritas.utils.synth`) so scripts and demos can reuse it; it is
re-exported here because test modules import it via ``tests.conftest``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from voiceveritas.utils.synth import SR, sine_wave, synthesize_speechlike

__all__ = ["SR", "sine_wave", "synthesize_speechlike"]


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
