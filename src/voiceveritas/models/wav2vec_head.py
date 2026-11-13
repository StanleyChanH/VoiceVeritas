"""Track B: frozen SSL backbone (wav2vec 2.0) embeddings + MLP head.

Follows the recipe validated in the literature (arXiv:2412.11896): *frozen*
pretrained speech-transformer embeddings with a light classifier reach SOTA
on scripted/spontaneous classification — fine-tuning is unnecessary. On
CPU-only machines this is the only practical deep approach: the backbone
runs once per utterance (embeddings are cacheable), and the head trains in
seconds.

Embedding: time-mean + time-std pooling over the last hidden state
(2 x 768 = 1536 dims per window).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from voiceveritas import __version__
from voiceveritas.models.base import ReadSpeechModel

logger = logging.getLogger(__name__)

MODEL_TYPE = "wav2vec_head"
DEFAULT_BACKBONE = "facebook/wav2vec2-base"
EMB_VERSION = "wav2vec2-base-v1"

# module-level memo of loaded backbones (they are big; loading per call
# would dominate latency)
_BACKBONE_CACHE: dict[str, Any] = {}


def embed_window(
    samples: np.ndarray,
    sr: int,
    backbone_name: str = DEFAULT_BACKBONE,
) -> np.ndarray:
    """Pool a frozen backbone's last hidden state into a 2*D vector."""
    import torch

    if backbone_name not in _BACKBONE_CACHE:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(backbone_name)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _BACKBONE_CACHE[backbone_name] = model
    model = _BACKBONE_CACHE[backbone_name]

    expected_sr = getattr(model.config, "sampling_rate", 16000)
    x = np.asarray(samples, dtype=np.float32)
    if sr != expected_sr:
        import librosa

        x = librosa.resample(x, orig_sr=sr, target_sr=expected_sr)

    with torch.inference_mode():
        out = model(torch.from_numpy(x).unsqueeze(0))
        hidden = out.last_hidden_state[0].numpy()  # (T, D)
    return np.concatenate([hidden.mean(axis=0), hidden.std(axis=0)])


class Wav2vecHead(ReadSpeechModel):
    """MLP head over frozen wav2vec 2.0 embeddings."""

    name = MODEL_TYPE

    def __init__(
        self,
        state_dict: dict[str, Any],
        hidden_dim: int,
        embed_dim: int,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        backbone_name: str = DEFAULT_BACKBONE,
        metadata: dict[str, Any] | None = None,
    ):
        import torch
        import torch.nn as nn

        class _Head(nn.Module):
            def __init__(self, in_dim: int, hidden: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden, 1),
                )

            def forward(self, x):  # type: ignore[no-untyped-def]
                return self.net(x).squeeze(-1)

        self.torch = torch
        self.net = _Head(embed_dim, hidden_dim)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float32)
        self.scaler_scale = np.asarray(scaler_scale, dtype=np.float32)
        self.backbone_name = backbone_name
        self.metadata = metadata or {}

    def _score_embedding(self, emb: np.ndarray) -> float:
        import torch

        z = (emb.astype(np.float32) - self.scaler_mean) / self.scaler_scale
        with torch.inference_mode():
            logit = float(self.net(torch.from_numpy(z).unsqueeze(0)))
        return float(1.0 / (1.0 + np.exp(-logit)))  # sigmoid

    def score_samples(self, samples: np.ndarray, sr: int, start_s: float, end_s: float) -> float:
        i0, i1 = round(start_s * sr), min(round(end_s * sr), len(samples))
        emb = embed_window(samples[i0:i1], sr, self.backbone_name)
        return self._score_embedding(emb)

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
            "backbone": self.backbone_name,
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()},
            "train_stats": train_stats or {},
            "metadata": self.metadata,
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> Wav2vecHead:
        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("type") != MODEL_TYPE:
            raise ValueError(f"{path} is not a {MODEL_TYPE} artifact")
        return cls(
            state_dict=payload["state_dict"],
            hidden_dim=payload["hidden_dim"],
            embed_dim=payload["embed_dim"],
            scaler_mean=payload["scaler_mean"],
            scaler_scale=payload["scaler_scale"],
            backbone_name=payload.get("backbone", DEFAULT_BACKBONE),
            metadata=payload.get("metadata", {}),
        )
