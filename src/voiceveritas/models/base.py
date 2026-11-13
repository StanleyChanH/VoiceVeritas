"""Shared model abstractions: windowing, segment scoring, prediction payloads.

Long answers are scored window-by-window and aggregated with the median —
the recipe validated in the scripted/spontaneous-classification literature
(arXiv:2412.11896) — so that partial reading ("starts honest, then reads")
shows up as elevated segment scores rather than being averaged away.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SegmentScore:
    """Score for one analysis window."""

    start_s: float
    end_s: float
    score: float  # P(read) for this window


@dataclass
class PredictionResult:
    """Complete prediction payload for one utterance."""

    read_probability: float  # aggregated P(read) in [0, 1]
    segments: list[SegmentScore] = field(default_factory=list)
    aggregation: str = "median"
    duration_s: float = 0.0
    model_name: str = ""
    feature_values: dict[str, float] | None = None
    top_contributors: list[tuple[str, float]] | None = None  # (name, contribution)


def plan_windows(
    duration_s: float, window_s: float = 30.0, hop_s: float = 15.0
) -> list[tuple[float, float]]:
    """Analysis windows covering [0, duration].

    - shorter audio → a single window covering everything;
    - longer audio → sliding windows; the last window is shifted left to end
      exactly at the audio end so the tail is not dropped.
    """
    if duration_s <= 0:
        return []
    if duration_s <= window_s:
        return [(0.0, duration_s)]
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start + window_s <= duration_s:
        windows.append((start, start + window_s))
        start += hop_s
    if windows and windows[-1][1] < duration_s - 1e-6:
        windows.append((duration_s - window_s, duration_s))
    return windows


class ReadSpeechModel(ABC):
    """Base class for read-speech detectors."""

    name: str = "base"

    @abstractmethod
    def score_samples(self, samples: np.ndarray, sr: int, start_s: float, end_s: float) -> float:
        """Score one already-extracted window (P(read))."""

    def predict(
        self,
        samples: np.ndarray,
        sr: int,
        window_s: float = 30.0,
        hop_s: float = 15.0,
    ) -> PredictionResult:
        """Windowed prediction with median aggregation."""
        duration = len(samples) / sr
        windows = plan_windows(duration, window_s, hop_s)
        segments = [
            SegmentScore(start_s=s, end_s=e, score=self.score_samples(samples, sr, s, e))
            for s, e in windows
        ]
        scores = np.array([seg.score for seg in segments], dtype=np.float64)
        agg = float(np.median(scores)) if scores.size else 0.0
        return PredictionResult(
            read_probability=agg,
            segments=segments,
            duration_s=duration,
            model_name=self.name,
        )

    def predict_file(self, path, window_s: float = 30.0, hop_s: float = 15.0) -> PredictionResult:
        """Convenience: load audio from path then predict."""
        from voiceveritas.audio import load_audio

        loaded = load_audio(path)
        return self.predict(loaded.samples, loaded.sample_rate, window_s=window_s, hop_s=hop_s)
