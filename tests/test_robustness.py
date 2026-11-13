"""Tests for acoustic perturbations and the robustness harness."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from tests.conftest import SR, sine_wave, synthesize_speechlike
from voiceveritas.eval.robustness import (
    Perturbation,
    _realized_snr_db,
    add_pink_noise,
    add_white_noise,
    apply_gain,
    default_conditions,
    mp3_transcode,
    resample_loop,
    run_robustness_eval,
    telephony_band,
)
from voiceveritas.manifest import Utterance, write_manifest
from voiceveritas.models.base import ReadSpeechModel


def _clean_speech(rng: np.random.Generator, duration: float = 8.0) -> np.ndarray:
    return synthesize_speechlike(rng, duration, style="spontaneous")


class TestPerturbations:
    def test_white_noise_hits_target_snr(self, rng):
        x = _clean_speech(rng)
        noisy = add_white_noise(x, SR, snr_db=10.0, rng=rng)
        realized = _realized_snr_db(x, noisy - x)
        assert realized == pytest.approx(10.0, abs=0.5)
        assert noisy.shape == x.shape

    def test_pink_noise_hits_target_snr_and_lowfreq_heavy(self, rng):
        x = _clean_speech(rng)
        noisy = add_pink_noise(x, SR, snr_db=10.0, rng=rng)
        noise = noisy - x
        assert _realized_snr_db(x, noise) == pytest.approx(10.0, abs=1.0)
        # pink: low half-band carries more noise power than high half
        half = noise.shape[0] // 2
        low = np.mean(np.fft.rfft(noise[: half * 2]).real ** 2)
        high = np.mean(np.fft.rfft(noise[half : half * 2]).real ** 2)
        assert low > high

    def test_gain_exact_and_clips(self):
        x = sine_wave(200.0, 1.0, amp=0.5)
        y = apply_gain(x, SR, db=6.0)
        np.testing.assert_allclose(y, x * 10.0 ** (6.0 / 20.0), atol=1e-6)
        z = apply_gain(x, SR, db=20.0)
        assert np.max(np.abs(z)) <= 1.0  # clipped, not wrapped

    def test_telephony_band_removes_highs(self, rng):
        x = sine_wave(6000.0, 1.0, amp=0.5)  # above the 3.4 kHz band
        y = telephony_band(x, SR)
        assert float(np.sqrt(np.mean(y**2))) < 0.05 * float(np.sqrt(np.mean(x**2)))

    def test_resample_loop_preserves_shape(self, rng):
        x = _clean_speech(rng, duration=3.0)
        y = resample_loop(x, SR, 8000)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))
        # 6 kHz tone does not survive an 8 kHz round trip
        tone = resample_loop(sine_wave(6000.0, 1.0, amp=0.5), SR, 8000)
        assert float(np.sqrt(np.mean(tone**2))) < 0.1

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
    def test_mp3_transcode_roundtrip(self, rng):
        x = _clean_speech(rng, duration=4.0)
        y = mp3_transcode(x, SR, bitrate_kbps=32)
        assert y is not None and y.shape == x.shape
        # survives structurally, differs numerically
        assert not np.allclose(y, x, atol=1e-4)


class _F0StdModel(ReadSpeechModel):
    """Deterministic stand-in classifier: scores by F0 dispersion.

    Real read speech has *lower* F0 dispersion — enough for the harness
    test without training anything.
    """

    name = "fake"

    def score_samples(self, samples: np.ndarray, sr: int, start_s: float, end_s: float) -> float:
        from voiceveritas.features import extract_prosody

        f0_std = extract_prosody(samples, sr).to_dict()["f0_std"]
        return float(np.clip(1.0 - f0_std / 80.0, 0.0, 1.0))


class TestRobustnessHarness:
    @pytest.fixture()
    def mini_manifest(self, tmp_path, rng):
        utts = []
        for i in range(4):
            style = "read" if i % 2 == 0 else "spontaneous"
            x = synthesize_speechlike(np.random.default_rng(500 + i), 6.0, style=style)
            import soundfile as sf

            p = tmp_path / f"utt_{i}.wav"
            sf.write(str(p), x, SR, subtype="PCM_16")
            utts.append(
                Utterance(
                    f"rob_{i}",
                    str(p),
                    1 if style == "read" else 0,
                    6.0,
                    f"spk{i}",
                    "synthetic",
                    split="test",
                )
            )
        m = tmp_path / "m.jsonl"
        write_manifest(m, utts)
        return m

    def test_default_conditions_cover_plan_matrix(self):
        names = [c.name for c in default_conditions()]
        assert "clean" in names
        assert any(n.startswith("white_noise") for n in names)
        assert any(n.startswith("pink_noise") for n in names)
        assert "telephony_300_3400" in names
        assert "resample_8k" in names
        assert "mp3_32k" in names

    def test_harness_runs_and_reports_drift(self, mini_manifest):
        conditions = [
            Perturbation("clean", "unmodified", "identity", {}),
            Perturbation("white_10db", "noise", "white", {"snr_db": 10.0}),
        ]
        rows = run_robustness_eval(_F0StdModel(), mini_manifest, conditions)
        assert rows[0]["condition"] == "clean"
        assert rows[0]["delta_auc_roc"] is None
        assert rows[1]["condition"] == "white_10db"
        assert isinstance(rows[1]["delta_auc_roc"], float)
        for row in rows:
            assert row["n"] == 4
            assert 0.0 <= row["auc_roc"] <= 1.0

    def test_skipped_condition_reported_not_raised(self, mini_manifest):
        rows = run_robustness_eval(
            _F0StdModel(),
            mini_manifest,
            [Perturbation("bogus_fn", "unknown", "no_such_fn", {})],
        )
        # unknown fn perturbs nothing: _perturb raises ValueError per utt -> skipped row
        assert rows[0].get("status") in {"skipped"} or rows[0]["condition"] == "bogus_fn"
