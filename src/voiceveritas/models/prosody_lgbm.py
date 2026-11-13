"""Track A: prosody features -> LightGBM (isotonic-calibrated).

The serialized artifact is a plain dict via joblib:

    {
      "type": "prosody_lgbm",
      "created": iso-timestamp,
      "feature_names": [...],        # exact order at train time
      "package_version": "0.1.0",
      "model": lgbm-or-calibrated-pipeline,
      "train_stats": {...},          # dataset stats snapshot
    }

Loading verifies the feature contract so a stale model fails loudly instead
of silently mis-scoring.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from voiceveritas import __version__
from voiceveritas.features import FEATURE_NAMES, extract_prosody
from voiceveritas.models.base import PredictionResult, ReadSpeechModel

logger = logging.getLogger(__name__)

MODEL_TYPE = "prosody_lgbm"


class ProsodyLGBM(ReadSpeechModel):
    """Calibrated LightGBM over the Track-A prosodic feature vector."""

    name = MODEL_TYPE  # class-level, so the registry can dispatch on it

    def __init__(
        self, model: Any, feature_names: list[str], metadata: dict[str, Any] | None = None
    ):
        if list(feature_names) != list(FEATURE_NAMES):
            raise ValueError(
                "feature contract mismatch: model was trained with different "
                f"features (n={len(feature_names)}) than this package provides "
                f"(n={len(FEATURE_NAMES)}). Retrain the model."
            )
        self.model = model
        self.feature_names = list(feature_names)
        self.metadata = metadata or {}

    def _features_for_window(
        self, samples: np.ndarray, sr: int, start_s: float, end_s: float
    ) -> np.ndarray:
        i0, i1 = round(start_s * sr), min(round(end_s * sr), len(samples))
        result = extract_prosody(samples[i0:i1], sr)
        return result.values

    def score_samples(self, samples: np.ndarray, sr: int, start_s: float, end_s: float) -> float:
        x = self._features_for_window(samples, sr, start_s, end_s).reshape(1, -1)
        proba = float(self.model.predict_proba(x)[0, 1])
        return float(np.clip(proba, 0.0, 1.0))

    def predict(
        self,
        samples: np.ndarray,
        sr: int,
        window_s: float = 30.0,
        hop_s: float = 15.0,
        with_explanation: bool = True,
    ) -> PredictionResult:
        result = super().predict(samples, sr, window_s=window_s, hop_s=hop_s)
        if with_explanation and result.segments:
            # explain the *longest* window (representative of the utterance)
            seg = max(result.segments, key=lambda s: s.end_s - s.start_s)
            x = self._features_for_window(samples, sr, seg.start_s, seg.end_s).reshape(1, -1)
            result.feature_values = dict(zip(self.feature_names, x.ravel().tolist(), strict=True))
            result.top_contributors = self.explain(x)
        return result

    def _raw_lgbm(self):  # type: ignore[no-untyped-def]
        """Unwrap the underlying LightGBM (for TreeSHAP contributions)."""
        m = self.model
        if hasattr(m, "calibrated_classifiers_"):  # sklearn CalibratedClassifierCV
            return m.calibrated_classifiers_[0].estimator
        return m

    def explain(self, x: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """TreeSHAP contributions (logit scale) of the top features pushing toward 'read'."""
        try:
            contrib = self._raw_lgbm().predict(x, pred_contrib=True)  # (n, n_features + bias)
            shap = contrib[0, :-1]
            order = np.argsort(-np.abs(shap))[:top_k]
            return [(FEATURE_NAMES[i], float(shap[i])) for i in order]
        except Exception:
            logger.debug("explanation failed", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path, train_stats: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": MODEL_TYPE,
            "created": datetime.now(timezone.utc).isoformat(),
            "package_version": __version__,
            "feature_names": self.feature_names,
            "model": self.model,
            "train_stats": train_stats or {},
            "metadata": self.metadata,
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> ProsodyLGBM:
        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("type") != MODEL_TYPE:
            raise ValueError(f"{path} is not a {MODEL_TYPE} artifact")
        model_version = payload.get("package_version", "?")
        if model_version != __version__:
            logger.warning(
                "model trained with package %s but running %s — feature "
                "contract is verified, but behavior may differ",
                model_version,
                __version__,
            )
        return cls(
            model=payload["model"],
            feature_names=payload["feature_names"],
            metadata=payload.get("metadata", {}),
        )
