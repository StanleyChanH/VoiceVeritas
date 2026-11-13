"""Tests for audio loading and IO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.conftest import SR, sine_wave
from voiceveritas.audio import AudioLoadError, load_audio, save_wav


class TestLoadAudio:
    def test_roundtrip_wav(self, tmp_path: Path):
        y = sine_wave(220.0, 1.5)
        p = save_wav(y, tmp_path / "a.wav")
        loaded = load_audio(p)
        assert loaded.sample_rate == SR
        assert loaded.duration == pytest.approx(1.5, abs=0.01)
        assert loaded.samples.dtype == np.float32
        # PCM_16 quantization: allow small error
        np.testing.assert_allclose(loaded.samples, y, atol=2 / 32768 + 1e-4)

    def test_resamples_to_target_sr(self, tmp_path: Path):
        sr8k = 8000
        y = sine_wave(200.0, 2.0, sr=sr8k)
        p = tmp_path / "b.wav"
        sf.write(str(p), y, sr8k)
        loaded = load_audio(p, sample_rate=16000)
        assert loaded.sample_rate == 16000
        assert loaded.duration == pytest.approx(2.0, abs=0.05)
        assert len(loaded.samples) == int(2.0 * 16000)

    def test_stereo_downmix(self, tmp_path: Path):
        left = sine_wave(200.0, 1.0)
        right = sine_wave(200.0, 1.0, amp=0.1)
        p = tmp_path / "stereo.wav"
        sf.write(str(p), np.stack([left, right], axis=1), SR)
        loaded = load_audio(p)
        assert loaded.samples.ndim == 1
        np.testing.assert_allclose(loaded.samples, (left + right) / 2, atol=1e-3)

    def test_load_from_bytes(self, tmp_path: Path):
        import io

        y = sine_wave(300.0, 0.5)
        buf = io.BytesIO()
        sf.write(buf, y, SR, format="WAV")
        loaded = load_audio(buf.getvalue())
        assert loaded.duration == pytest.approx(0.5, abs=0.01)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(AudioLoadError, match="does not exist"):
            load_audio(tmp_path / "nope.wav")

    def test_empty_audio_raises(self, tmp_path: Path):
        p = tmp_path / "empty.wav"
        sf.write(str(p), np.zeros(0, dtype=np.float32), SR)
        with pytest.raises(AudioLoadError, match="zero samples"):
            load_audio(p)


class TestSaveWav:
    def test_rejects_multichannel(self, tmp_path: Path):
        with pytest.raises(ValueError, match="mono"):
            save_wav(np.zeros((2, 100), dtype=np.float32), tmp_path / "x.wav")

    def test_creates_parent_dirs(self, tmp_path: Path):
        p = tmp_path / "deep" / "nested" / "x.wav"
        save_wav(np.zeros(100, dtype=np.float32), p)
        assert p.exists()
