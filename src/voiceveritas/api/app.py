"""FastAPI application for read-speech scoring.

Endpoints
---------
- ``GET  /v1/health``  — liveness + model identity
- ``GET  /v1/info``    — model card summary, thresholds, limits
- ``POST /v1/predict`` — multipart audio upload -> P(read) + segment scores

The reference implementation keeps **no copy** of uploaded audio: files are
decoded in-memory and discarded with the request (see docs/ethics.md).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from voiceveritas import __version__
from voiceveritas.audio import AudioLoadError, load_audio
from voiceveritas.models import load_model
from voiceveritas.models.base import ReadSpeechModel

logger = logging.getLogger(__name__)

DEFAULT_MAX_UPLOAD_MB = 50
DEFAULT_MAX_DURATION_S = 600.0
ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


class SegmentScoreOut(BaseModel):
    start_s: float
    end_s: float
    score: float


class ModelInfo(BaseModel):
    name: str
    package_version: str = __version__


class ContributorOut(BaseModel):
    feature: str
    contribution: float


class PredictResponse(BaseModel):
    read_probability: float = Field(..., ge=0.0, le=1.0)
    label: str  # "read" | "spontaneous" at the 0.5 reference threshold
    confidence: float = Field(..., ge=0.0, le=1.0)
    duration_s: float
    segmented_scores: list[SegmentScoreOut]
    processing_ms: float
    model: ModelInfo
    top_contributors: list[ContributorOut] | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    model: ModelInfo


def create_app(
    model_path: str,
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB,
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
) -> FastAPI:
    """Build the API app around a serialized model artifact."""
    model: ReadSpeechModel = load_model(model_path)

    app = FastAPI(
        title="VoiceVeritas API",
        description=(
            "Read-speech detection for interview audio. Returns P(read): the "
            "probability that an answer was read/recited rather than spoken "
            "spontaneously. **Decision support only — never an automated "
            "verdict** (see docs/ethics.md)."
        ),
        version=__version__,
    )

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(model=ModelInfo(name=model.name))

    @app.get("/v1/info")
    def info() -> dict[str, Any]:
        return {
            "model": model.name,
            "package_version": __version__,
            "limits": {
                "max_upload_mb": max_upload_mb,
                "max_duration_s": max_duration_s,
                "allowed_formats": sorted(ALLOWED_SUFFIXES),
            },
            "interpretation": {
                "read_probability": "P(answer is read/recited); 0.5 is the "
                "reference threshold, not a recommended policy threshold",
                "warning": "deploy with a threshold chosen by FPR budget "
                "(see metrics.fpr_at_tpr95); keep a human in the loop",
            },
        }

    @app.post(
        "/v1/predict",
        response_model=PredictResponse,
        responses={400: {"description": "bad audio"}, 413: {"description": "too large"}},
    )
    async def predict(
        audio: UploadFile = File(..., description="interview answer audio"),
        reference_text: str | None = Form(
            default=None,
            description="optional; reserved for ASR-alignment features, "
            "unused by audio-only models",
        ),
    ) -> PredictResponse:
        del reference_text  # audio-only for now; hook for future alignment
        data = await audio.read()
        if len(data) > max_upload_mb << 20:
            raise HTTPException(413, f"audio larger than {max_upload_mb} MB")
        suffix = "." + (audio.filename or "audio.wav").rsplit(".", 1)[-1].lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(400, f"unsupported format {suffix!r}")

        started = time.perf_counter()
        try:
            loaded = load_audio(bytes(data))
        except AudioLoadError as exc:
            raise HTTPException(400, f"could not decode audio: {exc}") from exc
        if loaded.duration > max_duration_s:
            raise HTTPException(400, f"audio longer than {max_duration_s:.0f}s")

        result = model.predict(loaded.samples, loaded.sample_rate)
        processing_ms = (time.perf_counter() - started) * 1000.0

        contributors = None
        if result.top_contributors:
            contributors = [
                ContributorOut(feature=f, contribution=round(c, 4))
                for f, c in result.top_contributors
            ]

        return PredictResponse(
            read_probability=round(result.read_probability, 4),
            label="read" if result.read_probability >= 0.5 else "spontaneous",
            confidence=round(abs(result.read_probability - 0.5) * 2.0, 4),
            duration_s=round(result.duration_s, 2),
            segmented_scores=[
                SegmentScoreOut(start_s=round(s.start_s, 2), end_s=round(s.end_s, 2), score=round(s.score, 4))
                for s in result.segments
            ],
            processing_ms=round(processing_ms, 1),
            model=ModelInfo(name=result.model_name),
            top_contributors=contributors,
        )

    return app
