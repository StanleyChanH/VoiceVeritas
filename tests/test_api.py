"""API service tests (TestClient; no network)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tests.conftest import SR, synthesize_speechlike  # noqa: E402
from voiceveritas.config import TrainConfig  # noqa: E402
from voiceveritas.train import train_baseline  # noqa: E402


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a tiny prosody model to serve (shared across API tests)."""
    root = tmp_path_factory.mktemp("apicorpus")
    from voiceveritas.manifest import Utterance, write_manifest

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


@pytest.fixture(scope="module")
def client(trained_model_path: Path) -> TestClient:
    from voiceveritas.api import create_app

    app = create_app(str(trained_model_path), max_upload_mb=5)
    return TestClient(app)


def _wav_bytes(style: str = "spontaneous", duration: float = 5.0, seed: int = 99) -> bytes:
    x = synthesize_speechlike(np.random.default_rng(seed), duration, style=style)
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV")
    return buf.getvalue()


class TestApi:
    def test_health(self, client: TestClient):
        r = client.get("/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model"]["name"] == "prosody_lgbm"

    def test_info_documents_ethics(self, client: TestClient):
        r = client.get("/v1/info")
        assert r.status_code == 200
        assert "human in the loop" in r.json()["interpretation"]["warning"]

    def test_predict_valid_wav(self, client: TestClient):
        r = client.post(
            "/v1/predict",
            files={"audio": ("answer.wav", _wav_bytes(), "audio/wav")},
            data={"reference_text": "optional"},
        )
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["read_probability"] <= 1.0
        assert body["label"] in {"read", "spontaneous"}
        assert len(body["segmented_scores"]) >= 1
        assert body["processing_ms"] > 0
        assert body["model"]["name"] == "prosody_lgbm"

    def test_predict_rejects_bad_format(self, client: TestClient):
        r = client.post(
            "/v1/predict",
            files={"audio": ("file.txt", b"not audio", "text/plain")},
        )
        assert r.status_code == 400

    def test_predict_rejects_undecodable(self, client: TestClient):
        r = client.post(
            "/v1/predict",
            files={"audio": ("fake.wav", b"\x00" * 64, "audio/wav")},
        )
        assert r.status_code == 400

    def test_predict_rejects_oversize(self, client: TestClient):
        # max_upload_mb=5 in the fixture app
        r = client.post(
            "/v1/predict",
            files={"audio": ("big.wav", b"\x00" * (6 << 20), "audio/wav")},
        )
        assert r.status_code == 413
