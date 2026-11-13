"""Tests for the energy VAD."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import SR, sine_wave, synthesize_speechlike
from voiceveritas.vad import energy_vad


class TestEnergyVad:
    def test_pure_silence_no_speech(self):
        vad = energy_vad(np.zeros(int(3.0 * SR), dtype=np.float32), SR)
        assert vad.speech_ratio == 0.0
        assert vad.segments == []

    def test_constant_tone_all_speech(self):
        vad = energy_vad(sine_wave(220.0, 2.0), SR)
        assert vad.speech_ratio == 1.0
        assert len(vad.segments) == 1
        assert vad.segments[0][0] == 0.0

    def test_finds_pauses_in_speechlike(self, rng):
        x = synthesize_speechlike(rng, 20.0, style="spontaneous")
        vad = energy_vad(x, SR)
        assert 0.3 < vad.speech_ratio < 0.97
        assert len(vad.segments) >= 3, "expected multiple speech runs separated by pauses"

    def test_segment_boundary_consistency(self, rng):
        x = synthesize_speechlike(rng, 15.0, style="read")
        vad = energy_vad(x, SR)
        for s, e in vad.segments:
            assert 0.0 <= s < e <= 15.0 + 1e-6
        # segments sorted and non-overlapping
        for (_, e1), (s2, _) in zip(vad.segments[:-1], vad.segments[1:], strict=False):
            assert e1 < s2

    def test_short_bursts_dropped(self, rng):
        # speech + a 20 ms blip: blip must not survive min_speech_s
        x = synthesize_speechlike(rng, 5.0, style="read")
        blip = sine_wave(300.0, 0.02, amp=0.6)
        x[-len(blip) :] += blip
        vad = energy_vad(x, SR)
        for s, e in vad.segments:
            assert e - s >= 0.05 - 1e-6

    def test_rejects_non_mono(self):
        with pytest.raises(ValueError, match="mono"):
            energy_vad(np.zeros((2, 100)), SR)
