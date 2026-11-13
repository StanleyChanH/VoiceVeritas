"""Ensemble of read-speech models (median aggregation).

Median (not mean) aggregation: one hallucinating member — e.g. a deep
model under white-noise inversion — cannot drag the ensemble past the
robust members. Members are embedded inline in the artifact, so the file
is portable (no external paths).

Windowing: members may use different window defaults, so the ensemble
plans its own windows and scores each member per window via
``score_samples`` — segment scores are medians across members.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from voiceveritas import __version__
from voiceveritas.models.base import (
    PredictionResult,
    ReadSpeechModel,
    SegmentScore,
    plan_windows,
)

logger = logging.getLogger(__name__)

MODEL_TYPE = "ensemble"


class EnsembleModel(ReadSpeechModel):
    """Median-of-members read-speech detector."""

    name = MODEL_TYPE

    def __init__(self, members: list[ReadSpeechModel], window_s: float = 30.0, hop_s: float = 15.0):
        if len(members) < 2:
            raise ValueError("an ensemble needs at least two members")
        self.members = members
        self.window_s = window_s
        self.hop_s = hop_s
        self.name = f"ensemble({len(members)})"

    def score_samples(self, samples: np.ndarray, sr: int, start_s: float, end_s: float) -> float:
        scores = [m.score_samples(samples, sr, start_s, end_s) for m in self.members]
        return float(np.median(scores))

    def predict(  # type: ignore[override]  # window defaults come from the bundle
        self,
        samples: np.ndarray,
        sr: int,
        window_s: float | None = None,
        hop_s: float | None = None,
    ) -> PredictionResult:
        w = window_s or self.window_s
        h = hop_s or self.hop_s
        duration = len(samples) / sr
        windows = plan_windows(duration, w, h)

        member_results = [m.predict(samples, sr, window_s=w, hop_s=h) for m in self.members]
        segments = []
        for i, (s, e) in enumerate(windows):
            scores = [r.segments[i].score for r in member_results if i < len(r.segments)]
            segments.append(SegmentScore(start_s=s, end_s=e, score=float(np.median(scores))))
        agg_scores = [r.read_probability for r in member_results]
        agg = float(np.median(agg_scores))

        from voiceveritas.vad import estimate_snr_db

        return PredictionResult(
            read_probability=agg,
            segments=segments,
            duration_s=duration,
            model_name=self.name,
            estimated_snr_db=estimate_snr_db(samples, sr),
        )

    # ------------------------------------------------------------------
    # persistence: member payloads embedded inline
    # ------------------------------------------------------------------
    def save(self, path: str | Path, train_stats: dict[str, Any] | None = None) -> Path:  # type: ignore[override]
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": MODEL_TYPE,
            "created": datetime.now(timezone.utc).isoformat(),
            "package_version": __version__,
            "window_s": self.window_s,
            "hop_s": self.hop_s,
            "members": [_member_payload(m) for m in self.members],
            "train_stats": train_stats or {},
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> EnsembleModel:
        import joblib

        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("type") != MODEL_TYPE:
            raise ValueError(f"{path} is not an {MODEL_TYPE} artifact")
        members = [_load_member(p) for p in payload["members"]]
        return cls(
            members,
            window_s=payload.get("window_s", 30.0),
            hop_s=payload.get("hop_s", 15.0),
        )


def _member_payload(model: ReadSpeechModel) -> dict[str, Any]:
    """Serialize a member to its joblib payload dict (in memory)."""
    import io

    import joblib

    buf = io.BytesIO()
    joblib.dump(model, buf)
    return {"nested": buf.getvalue()}


def _load_member(payload: dict[str, Any]) -> ReadSpeechModel:
    import io

    import joblib

    if "nested" in payload:  # inline joblib blob (our format)
        return joblib.load(io.BytesIO(payload["nested"]))
    raise ValueError("ensemble member payload not recognized")
