"""Binary classification metrics.

Beyond the usual AUC family we report **EER** and **FPR@TPR95** — in this
application a false positive means accusing an honest, fluent candidate of
cheating, which is the error that matters most to control (see docs/ethics.md).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def eer_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Equal error rate: FPR at the operating point where FPR == FNR (1 - TPR)."""
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(max(fpr[i], fnr[i]))


def fpr_at_tpr(fpr: np.ndarray, tpr: np.ndarray, target_tpr: float = 0.95) -> float:
    """Smallest FPR achieved at (at least) the target TPR."""
    eligible = tpr >= target_tpr
    if not eligible.any():
        return 1.0
    return float(fpr[eligible].min())


def threshold_at_fpr(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    max_fpr: float = 0.05,
) -> float:
    """Score threshold whose FPR is (just under) ``max_fpr``.

    Used to deploy at a fixed false-alarm budget: flag as "read" only what
    scores above this threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    eligible = np.flatnonzero(fpr <= max_fpr)
    if eligible.size == 0:
        return float("inf")
    best = eligible[np.argmax(tpr[eligible])]
    return float(thresholds[best])


def compute_binary_metrics(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
    label: str = "",
) -> dict[str, Any]:
    """Full metric report for read-speech scores (higher score = more 'read').

    Args:
        y_true: binary ground truth (1 = read).
        scores: predicted probabilities / scores in [0, 1].
        threshold: decision threshold for confusion-based metrics.
        label: optional subset label, echoed into the result.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)

    if y_true.size == 0:
        raise ValueError("y_true is empty")
    if y_true.shape != scores.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs scores {scores.shape}")

    both_classes = np.unique(y_true).size == 2
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    out: dict[str, Any] = {
        "label": label,
        "n": int(y_true.size),
        "n_read": int(y_true.sum()),
        "n_spontaneous": int((y_true == 0).sum()),
        "threshold": threshold,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy": float((preds == y_true).mean()),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
        "f1": _f1(tp, fp, fn),
        "brier": float(brier_score_loss(y_true, np.clip(scores, 0.0, 1.0))),
    }

    if both_classes:
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        out.update(
            {
                "auc_roc": float(roc_auc_score(y_true, scores)),
                "auc_pr": float(average_precision_score(y_true, scores)),
                "eer": eer_from_roc(fpr, tpr),
                "fpr_at_tpr95": fpr_at_tpr(fpr, tpr, 0.95),
                "thr_at_fpr05": float(thresholds[int(np.nanargmin(np.abs(fpr - 0.05)))]),
            }
        )
    else:
        out.update({"auc_roc": None, "auc_pr": None, "eer": None, "fpr_at_tpr95": None})
    return out


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0
