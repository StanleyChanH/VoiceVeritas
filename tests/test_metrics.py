"""Tests for binary metrics."""

from __future__ import annotations

import numpy as np
import pytest

from voiceveritas.metrics import compute_binary_metrics, eer_from_roc, threshold_at_fpr


def test_perfect_scores():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_binary_metrics(y, s)
    assert m["auc_roc"] == 1.0
    assert m["auc_pr"] == 1.0
    assert m["eer"] == pytest.approx(0.0, abs=0.05)
    assert m["accuracy"] == 1.0
    assert m["confusion"]["tp"] == 2 and m["confusion"]["tn"] == 2


def test_chance_scores():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.5, 0.5, 0.5, 0.5])
    m = compute_binary_metrics(y, s)
    assert m["auc_roc"] == pytest.approx(0.5)
    assert m["fpr_at_tpr95"] == 1.0  # TPR never reaches 0.95


def test_single_class_degrades_gracefully():
    y = np.array([1, 1])
    s = np.array([0.7, 0.9])
    m = compute_binary_metrics(y, s)
    assert m["auc_roc"] is None
    assert "accuracy" in m


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="mismatch"):
        compute_binary_metrics(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


def test_eer_symmetric():
    fpr = np.array([0.0, 0.1, 0.2, 1.0])
    tpr = np.array([0.0, 0.8, 0.9, 1.0])
    eer = eer_from_roc(fpr, tpr)
    assert 0.0 <= eer <= 1.0


def test_threshold_at_fpr():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    scores = np.clip(0.3 * y + rng.normal(0.4, 0.15, 500), 0, 1)
    thr = threshold_at_fpr(y, scores, max_fpr=0.10)
    preds = scores >= thr
    fpr = preds[y == 0].mean()
    assert fpr <= 0.10 + 0.02  # small slack for discreteness


def test_fpr_at_tpr95_reported():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    scores = np.clip(0.4 * y + rng.normal(0.35, 0.12, 2000), 0, 1)
    m = compute_binary_metrics(y, scores)
    assert 0.0 <= m["fpr_at_tpr95"] <= 1.0
    assert 0.0 <= m["eer"] <= 1.0
