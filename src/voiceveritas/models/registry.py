"""Model registry: load any supported model artifact by type tag."""

from __future__ import annotations

from typing import Any

from voiceveritas.models.base import ReadSpeechModel
from voiceveritas.models.prosody_lgbm import ProsodyLGBM

_LOADERS: dict[str, Any] = {
    ProsodyLGBM.name: ProsodyLGBM.load,
}


def _register_deep() -> None:
    """Register optional-extras models lazily (torch/transformers/[asr])."""
    try:
        from voiceveritas.models.wav2vec_head import Wav2vecHead

        _LOADERS[Wav2vecHead.name] = Wav2vecHead.load
    except ImportError:  # pragma: no cover - depends on extras installed
        pass
    try:
        from voiceveritas.models.prosody_asr_lgbm import ProsodyASRLGBM

        _LOADERS[ProsodyASRLGBM.name] = ProsodyASRLGBM.load
    except ImportError:  # pragma: no cover - depends on extras installed
        pass
    try:
        from voiceveritas.models.ensemble import EnsembleModel

        _LOADERS[EnsembleModel.name] = EnsembleModel.load
    except ImportError:  # pragma: no cover - depends on extras installed
        pass


_register_deep()


def load_model(path) -> ReadSpeechModel:  # type: ignore[no-untyped-def]
    """Load a serialized model artifact, dispatching on its ``type`` tag."""
    import joblib

    payload = joblib.load(path)
    model_type = payload.get("type") if isinstance(payload, dict) else None
    if model_type not in _LOADERS:
        raise ValueError(f"unknown model type {model_type!r}; supported: {sorted(_LOADERS)}")
    return _LOADERS[model_type](path)
