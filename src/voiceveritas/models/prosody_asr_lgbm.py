"""Track A+ASR: prosody + lexical/alignment features -> LightGBM.

Extends the interpretable baseline with the ASR feature block
(fillers, pause-syntax misalignment, formality proxies, optional
perplexity/WER). At prediction time each analysis window is transcribed
on the fly — requires the ``[asr]`` extra (faster-whisper).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from voiceveritas import __version__
from voiceveritas.asr.features import ASR_FEATURE_NAMES, extract_asr_features
from voiceveritas.asr.transcribe import WordTimes, transcribe_words
from voiceveritas.features import FEATURE_NAMES as PROSODY_NAMES
from voiceveritas.features import extract_prosody
from voiceveritas.models.prosody_lgbm import ProsodyLGBM

logger = logging.getLogger(__name__)

MODEL_TYPE = "prosody_asr_lgbm"
COMBINED_NAMES: tuple[str, ...] = tuple(PROSODY_NAMES) + tuple(ASR_FEATURE_NAMES)


class ProsodyASRLGBM(ProsodyLGBM):
    """Prosody + ASR features through the same calibrated-LightGBM contract."""

    name = MODEL_TYPE

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        metadata: dict[str, Any] | None = None,
        compute_ppl: bool = False,
    ):
        if list(feature_names) != list(COMBINED_NAMES):
            raise ValueError(
                "feature contract mismatch: expected prosody+ASR combined "
                f"features (n={len(COMBINED_NAMES)}), got n={len(feature_names)}. "
                "This artifact belongs to a different feature set."
            )
        # NOTE: bypass ProsodyLGBM's own contract check by constructing the
        # parent chain manually — we keep its calibration/persistence logic.
        self.model = model
        self.feature_names = list(feature_names)
        self.metadata = metadata or {}
        self.compute_ppl = compute_ppl

    def _features_for_window(
        self, samples: np.ndarray, sr: int, start_s: float, end_s: float
    ) -> np.ndarray:
        i0, i1 = round(start_s * sr), min(round(end_s * sr), len(samples))
        window = samples[i0:i1]
        prosody = extract_prosody(window, sr).values
        words: list[WordTimes] = transcribe_words(window, sr)
        asr = extract_asr_features(words, len(window) / sr, compute_ppl=self.compute_ppl).values
        return np.concatenate([prosody, asr])

    def explain(self, x: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """TreeSHAP over combined names (reuse parent logic via names)."""
        try:
            contrib = self._raw_lgbm().predict(x, pred_contrib=True)
            shap = contrib[0, :-1]
            order = np.argsort(-np.abs(shap))[:top_k]
            return [(COMBINED_NAMES[i], float(shap[i])) for i in order]
        except Exception:
            logger.debug("explanation failed", exc_info=True)
            return []

    @classmethod
    def load(cls, path: str | Path) -> ProsodyASRLGBM:
        import joblib

        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("type") != MODEL_TYPE:
            raise ValueError(f"{path} is not a {MODEL_TYPE} artifact")
        return cls(
            model=payload["model"],
            feature_names=payload["feature_names"],
            metadata=payload.get("metadata", {}),
            compute_ppl=payload.get("compute_ppl", False),
        )

    def save(self, path: str | Path, train_stats: dict[str, Any] | None = None) -> Path:  # type: ignore[override]
        import joblib

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
            "compute_ppl": self.compute_ppl,
        }
        joblib.dump(payload, path)
        return path
