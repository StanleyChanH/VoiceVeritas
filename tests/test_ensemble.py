"""Tests for the median ensemble model."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import SR, synthesize_speechlike
from voiceveritas.models import load_model
from voiceveritas.models.base import ReadSpeechModel
from voiceveritas.models.ensemble import EnsembleModel


class _Fixed(ReadSpeechModel):
    """Returns a constant score — lets us test aggregation arithmetic."""

    def __init__(self, name: str, score: float):
        self.name = name
        self.score = score

    def score_samples(self, samples, sr, start_s, end_s):  # type: ignore[no-untyped-def]
        return self.score


class TestEnsembleAggregation:
    def test_median_resists_outlier(self, rng):
        x = synthesize_speechlike(rng, 8.0)
        # members: 0.2, 0.3, 0.9 — median 0.3, mean would be 0.47
        ens = EnsembleModel([_Fixed("a", 0.2), _Fixed("b", 0.3), _Fixed("c", 0.9)])
        result = ens.predict(x, SR)
        assert result.read_probability == pytest.approx(0.3)
        assert result.model_name == "ensemble(3)"
        for seg in result.segments:
            assert seg.score == pytest.approx(0.3)

    def test_needs_two_members(self):
        with pytest.raises(ValueError, match="two members"):
            EnsembleModel([_Fixed("a", 0.5)])

    def test_snr_attached(self, rng):
        ens = EnsembleModel([_Fixed("a", 0.5), _Fixed("b", 0.5)])
        result = ens.predict(synthesize_speechlike(rng, 6.0), SR)
        assert result.estimated_snr_db == result.estimated_snr_db  # finite


class TestEnsemblePersistence:
    def test_roundtrip_with_real_models(self, trained_model_path, tmp_path):
        """Bundle the shared trained model (twice) — save/load/predict."""
        m1 = load_model(trained_model_path)
        m2 = load_model(trained_model_path)
        ens = EnsembleModel([m1, m2])
        out = tmp_path / "ens.joblib"
        ens.save(out)

        loaded = load_model(out)
        assert isinstance(loaded, EnsembleModel)
        assert len(loaded.members) == 2

        rng = np.random.default_rng(3)
        x = synthesize_speechlike(rng, 6.0, style="spontaneous")
        r_member = m1.predict(x, SR)
        r_ens = loaded.predict(x, SR)
        # identical members -> ensemble equals member
        assert r_ens.read_probability == pytest.approx(r_member.read_probability, abs=1e-6)

    def test_rejects_foreign_artifact(self, tmp_path):
        import joblib

        joblib.dump({"type": "bogus"}, tmp_path / "bad.joblib")
        with pytest.raises(ValueError, match="ensemble"):
            EnsembleModel.load(tmp_path / "bad.joblib")
