"""API service tests (TestClient; no network)."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from tests.conftest import SR, synthesize_speechlike


def _wav_bytes(style: str = "spontaneous", duration: float = 5.0, seed: int = 99) -> bytes:
    x = synthesize_speechlike(np.random.default_rng(seed), duration, style=style)
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV")
    return buf.getvalue()


class TestApi:
    def test_health(self, client):
        r = client.get("/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model"]["name"] == "prosody_lgbm"

    def test_info_documents_ethics(self, client):
        r = client.get("/v1/info")
        assert r.status_code == 200
        assert "human in the loop" in r.json()["interpretation"]["warning"]

    def test_predict_valid_wav(self, client):
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

    def test_predict_rejects_bad_format(self, client):
        r = client.post(
            "/v1/predict",
            files={"audio": ("file.txt", b"not audio", "text/plain")},
        )
        assert r.status_code == 400

    def test_predict_rejects_undecodable(self, client):
        r = client.post(
            "/v1/predict",
            files={"audio": ("fake.wav", b"\x00" * 64, "audio/wav")},
        )
        assert r.status_code == 400

    def test_predict_rejects_oversize(self, client):
        # max_upload_mb=5 in the fixture app
        r = client.post(
            "/v1/predict",
            files={"audio": ("big.wav", b"\x00" * (6 << 20), "audio/wav")},
        )
        assert r.status_code == 413
