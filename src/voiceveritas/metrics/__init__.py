"""Evaluation metrics for binary read-speech detection."""

from voiceveritas.metrics.binary import (
    compute_binary_metrics,
    eer_from_roc,
    threshold_at_fpr,
)

__all__ = ["compute_binary_metrics", "eer_from_roc", "threshold_at_fpr"]
