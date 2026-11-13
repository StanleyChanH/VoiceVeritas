"""Model zoo: shared abstractions and concrete read-speech models."""

from voiceveritas.models.base import (
    PredictionResult,
    ReadSpeechModel,
    SegmentScore,
    plan_windows,
)
from voiceveritas.models.registry import load_model

__all__ = [
    "PredictionResult",
    "ReadSpeechModel",
    "SegmentScore",
    "load_model",
    "plan_windows",
]
