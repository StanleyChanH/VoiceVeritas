"""Correctness tests for the prosodic feature extractor.

Two layers:
1. *Physical correctness* — a 220 Hz tone must yield f0 ≈ 220 Hz.
2. *Construct validity* — the caricatured synthetic styles differ in exactly
   the feature families the literature says are discriminative (F0 stability,
   pause variability, rhythm regularity).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import SR, sine_wave, synthesize_speechlike
from voiceveritas.features import FEATURE_NAMES, extract_prosody
from voiceveritas.vad import energy_vad


class TestContract:
    def test_names_unique_and_fixed_order(self):
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
        assert FEATURE_NAMES[0] == "f0_voiced_ratio"

    def test_no_duration_shortcut(self):
        # duration leaks dataset identity (see docs/benchmark.md) — guard
        # against reintroducing it as a feature
        assert "duration_log" not in FEATURE_NAMES

    def test_vector_shape_matches_names(self, rng):
        x = synthesize_speechlike(rng, 5.0, style="spontaneous")
        result = extract_prosody(x, SR)
        assert result.values.shape == (len(FEATURE_NAMES),)
        assert set(result.to_dict()) == set(FEATURE_NAMES)

    def test_no_nan_for_speechlike(self, rng):
        x = synthesize_speechlike(rng, 12.0, style="read")
        result = extract_prosody(x, SR)
        # everything except shimmer/jitter (praat-dependent) should be finite
        allow_nan = {"jitter_local", "shimmer_local", "hnr_db"}
        d = result.to_dict()
        bad = [k for k, v in d.items() if not np.isfinite(v) and k not in allow_nan]
        assert not bad, f"unexpected NaNs in {bad}"

    def test_accepts_precomputed_vad(self, rng):
        x = synthesize_speechlike(rng, 6.0)
        vad = energy_vad(x, SR)
        a = extract_prosody(x, SR, vad=vad)
        b = extract_prosody(x, SR)
        np.testing.assert_allclose(a.values, b.values, equal_nan=True)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError, match="mono"):
            extract_prosody(np.zeros((2, 100)), SR)
        with pytest.raises(ValueError, match="non-empty"):
            extract_prosody(np.zeros(0), SR)


class TestPhysicalCorrectness:
    def test_f0_of_pure_tone(self):
        result = extract_prosody(sine_wave(220.0, 3.0, amp=0.4), SR)
        d = result.to_dict()
        assert d["f0_mean"] == pytest.approx(220.0, abs=5.0)
        assert d["f0_median"] == pytest.approx(220.0, abs=5.0)

    def test_constant_tone_is_all_speech(self):
        result = extract_prosody(sine_wave(200.0, 2.0), SR)
        assert result.to_dict()["speech_ratio"] == pytest.approx(1.0)


@pytest.fixture(scope="module")
def style_features():
    read = extract_prosody(synthesize_speechlike(np.random.default_rng(1), 30.0, style="read"), SR)
    spon = extract_prosody(
        synthesize_speechlike(np.random.default_rng(2), 30.0, style="spontaneous"), SR
    )
    return read.to_dict(), spon.to_dict()


class TestConstructValidity:
    """The synthetic styles must be separable by the intended feature families."""

    def test_read_has_stabler_f0(self, style_features):
        read, spon = style_features
        assert read["f0_std"] < spon["f0_std"]
        assert read["f0_range_semitones"] < spon["f0_range_semitones"]

    def test_read_has_more_regular_pauses(self, style_features):
        read, spon = style_features
        # pause_cv is intentionally not asserted here: the VAD's min-pause
        # merging distorts the ratio for very short, narrow pause
        # distributions (the surviving tail inflates CV).
        assert read["pause_std"] < spon["pause_std"]

    def test_read_has_more_regular_rhythm(self, style_features):
        read, spon = style_features
        assert read["npvi_speech"] < spon["npvi_speech"]

    def test_spontaneous_pauses_longer_on_average(self, style_features):
        read, spon = style_features
        assert read["pause_mean"] < spon["pause_mean"]
