"""Model registry: load any supported model artifact by type tag."""

from __future__ import annotations

from typing import Any

from voiceveritas.models.base import ReadSpeechModel
from voiceveritas.models.prosody_lgbm import ProsodyLGBM

_LOADERS: dict[str, Any] = {
    ProsodyLGBM.name: ProsodyLGBM.load,
}


def _register_deep() -> None:
    """Register Track-B lazily: torch/transformers are optional extras."""
    try:
        from voiceveritas.models.wav2vec_head import Wav2vecHead

        _LOADERS[Wav2vecHead.name] = Wav2vecHead.load
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
