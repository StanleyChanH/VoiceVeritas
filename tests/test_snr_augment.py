"""Tests for SNR estimation, training augmentation, and SNR-gated warnings."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import SR, sine_wave, synthesize_speechlike
from voiceveritas.eval.robustness import add_white_noise
from voiceveritas.train.augment import augment_for_training
from voiceveritas.vad import estimate_snr_db


class TestSnrEstimator:
    def test_recovers_known_snr(self, rng):
        x = synthesize_speechlike(rng, 20.0, style="spontaneous")
        noisy = add_white_noise(x, SR, snr_db=12.0, rng=rng)
        est = estimate_snr_db(noisy, SR)
        assert est == pytest.approx(12.0, abs=3.0)

    def test_clean_speech_high_snr(self, rng):
        x = synthesize_speechlike(rng, 15.0, style="read")
        est = estimate_snr_db(x, SR)
        assert est > 20.0

    def test_pure_silence_nan(self):
        assert np.isnan(estimate_snr_db(np.zeros(int(3 * SR), dtype=np.float32), SR))

    def test_constant_tone_flags_low_snr(self):
        # no noise frames observable: the fallback drives SNR low, which
        # correctly routes non-speech input to human review
        est = estimate_snr_db(sine_wave(200.0, 2.0), SR)
        assert est == est  # finite
        assert est < 15.0


class TestAugmentation:
    def test_deterministic_per_key(self, rng):
        x = synthesize_speechlike(rng, 10.0)
        a1 = augment_for_training(x, SR, key="utt_1", seed=42)
        a2 = augment_for_training(x, SR, key="utt_1", seed=42)
        np.testing.assert_array_equal(a1, a2)

    def test_different_keys_differ(self, rng):
        x = synthesize_speechlike(rng, 10.0)
        a1 = augment_for_training(x, SR, key="utt_1", seed=42)
        a2 = augment_for_training(x, SR, key="utt_2", seed=42)
        assert not np.allclose(a1, a2)

    def test_output_finite_and_shape(self, rng):
        x = synthesize_speechlike(rng, 8.0)
        y = augment_for_training(x, SR, key="k", seed=1)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))

    def test_stays_in_snr_range(self, rng):
        from voiceveritas.eval.robustness import _realized_snr_db

        for i in range(6):
            x = synthesize_speechlike(np.random.default_rng(100 + i), 12.0)
            y = augment_for_training(x, SR, key=f"utt_{i}", seed=42)
            snr = _realized_snr_db(x, y - x)
            assert 8.0 <= snr <= 28.0, f"augmented SNR {snr:.1f} outside 10-25 dB window"


class TestAugmentedDataset:
    def test_augmented_cache_namespace_isolated(self, tmp_path, rng):
        import soundfile as sf

        from voiceveritas.manifest import Utterance, write_manifest
        from voiceveritas.train.dataset import build_feature_dataset

        root = tmp_path / "corp"
        root.mkdir()
        utts = []
        for i in range(4):
            x = synthesize_speechlike(np.random.default_rng(200 + i), 6.0)
            p = root / f"u{i}.wav"
            sf.write(str(p), x, SR, subtype="PCM_16")
            utts.append(
                Utterance(
                    f"aug_{i}",
                    str(p),
                    i % 2,
                    6.0,
                    f"spk{i}",
                    "synthetic",
                    split="train" if i < 3 else "test",
                )
            )
        m = root / "m.jsonl"
        write_manifest(m, utts)

        clean = build_feature_dataset([str(m)], cache_dir=tmp_path / "cache")
        aug = build_feature_dataset(
            [str(m)], cache_dir=tmp_path / "cache", augment=True, augment_seed=42
        )
        # train rows differ (augmented), test rows identical
        assert not np.allclose(clean.X[:3], aug.X[:3])
        np.testing.assert_allclose(clean.X[3], aug.X[3], equal_nan=True)
        # and clean cache file untouched by the augmented namespace
        assert (tmp_path / "cache" / "prosody-v1.npz").exists()
        assert (tmp_path / "cache" / "prosody-v1-aug.npz").exists()


class TestSnrInPredictions:
    def test_predict_result_carries_snr(self, rng):
        from voiceveritas.models.base import ReadSpeechModel

        class _Const(ReadSpeechModel):
            name = "const"

            def score_samples(self, samples, sr, start_s, end_s):  # type: ignore[no-untyped-def]
                return 0.5

        model = _Const()
        clean = model.predict(synthesize_speechlike(np.random.default_rng(1), 10.0), SR)
        assert clean.estimated_snr_db == clean.estimated_snr_db  # finite
        assert clean.estimated_snr_db > 15.0

        noisy_audio = add_white_noise(
            synthesize_speechlike(np.random.default_rng(2), 10.0),
            SR,
            8.0,
            np.random.default_rng(3),
        )
        noisy = model.predict(noisy_audio, SR)
        assert noisy.estimated_snr_db < 15.0

    def test_api_warns_on_low_snr(self, client, rng):  # client fixture: tests/conftest.py
        import io

        import soundfile as sf

        x = synthesize_speechlike(np.random.default_rng(5), 6.0)
        noisy = add_white_noise(x, SR, 8.0, np.random.default_rng(6))
        buf = io.BytesIO()
        sf.write(buf, noisy, SR, format="WAV")
        r = client.post(
            "/v1/predict", files={"audio": ("low_snr.wav", buf.getvalue(), "audio/wav")}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["estimated_snr_db"] is not None and body["estimated_snr_db"] < 15.0
        assert body["warning"] and "human review" in body["warning"]
