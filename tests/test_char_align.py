"""Tests for char-level forced alignment (pure functions, no models)."""

from __future__ import annotations

import numpy as np
import pytest

from voiceveritas.asr.char_align import (
    CharSpan,
    align_hyp_indices,
    align_transcript,
    frames_to_words,
    midword_silence_frames,
    _normalize,
)
from voiceveritas.asr.features import ALIGN_FEATURE_NAMES, extract_align_features
from voiceveritas.asr.transcribe import WordTimes


def span(ch: str, start: float, end: float) -> CharSpan:
    return CharSpan(ch, start, end)


class TestNormalize:
    def test_uppercase_and_strip(self):
        assert _normalize("Hello, World! It's 42-B.") == "HELLO WORLD IT'S B"

    def test_empty(self):
        assert _normalize("123 ?!") == ""


class TestDPAlign:
    def test_perfect_match(self):
        match = align_hyp_indices(list("AB"), list("AB"))
        assert match == [0, 1]

    def test_substitution_binds(self):
        # one decode error still binds the char to the nearest frame
        match = align_hyp_indices(list("CAT"), list("KAT"))
        assert match[1] == 1 and match[2] == 2

    def test_insertion_in_hyp(self):
        match = align_hyp_indices(list("AB"), list("AXXB"))
        assert match[0] == 0 and match[1] == 3

    def test_deletion_in_hyp(self):
        match = align_hyp_indices(list("ABC"), list("AC"))
        assert match[0] == 0 and match[1] == -1 and match[2] == 1

    def test_empty_ref(self):
        assert align_hyp_indices([], list("ABC")) == []


class TestFramesToWords:
    def test_groups_on_separator(self):
        spans = [
            span("H", 0.00, 0.02),
            span("E", 0.02, 0.04),
            span("|", 0.04, 0.06),
            span("A", 0.06, 0.08),
            span("T", 0.08, 0.10),
        ]
        words = frames_to_words(spans)
        assert [w.word for w in words] == ["HE", "AT"]
        assert words[0].start == pytest.approx(0.00)
        assert words[1].end == pytest.approx(0.10)

    def test_midword_silences_detected(self):
        # 60 ms hole inside a word = 3 skipped frames -> one mid-word silence
        spans = [
            span("C", 0.00, 0.02),
            span("O", 0.02, 0.04),
            span("M", 0.10, 0.12),
            span("P", 0.12, 0.14),
        ]
        gaps = midword_silence_frames(spans)
        assert len(gaps) == 1
        assert gaps[0] == pytest.approx(0.06)

    def test_cross_separator_gap_not_midword(self):
        spans = [span("A", 0.0, 0.02), span("|", 0.10, 0.12), span("B", 0.14, 0.16)]
        assert midword_silence_frames(spans) == []


class TestAlignTranscript:
    def test_end_to_end_binding(self):
        # CTC path: "THE QUICK BROWN FOX" with one decode error (BR0WN)
        spans = []
        t = 0.0
        for ch, dur in [
            ("T", 1),
            ("H", 1),
            ("E", 1),
            ("|", 1),
            ("Q", 1),
            ("U", 1),
            ("I", 1),
            ("C", 1),
            ("K", 1),
            ("|", 1),
            ("B", 1),
            ("R", 1),
            ("0", 1),
            ("W", 1),
            ("N", 1),
            ("|", 1),
            ("F", 1),
            ("O", 1),
            ("X", 1),
        ]:
            spans.append(span(ch, t, t + 0.02 * dur))
            t += 0.02 * dur
        words = align_transcript(spans, "the quick brown fox")
        assert [w.word for w in words] == ["THE", "QUICK", "BROWN", "FOX"]
        # boundaries land on CTC frame grid
        assert words[0].start == pytest.approx(0.0)
        assert words[1].start == pytest.approx(0.08)

    def test_sparse_hyp_drops_most_words(self):
        # with a single-char hyp path, substitution lets at most one word
        # bind (degenerate binding — exactly what happens on hard audio)
        spans = [span("X", 0.0, 0.02)]
        words = align_transcript(spans, "completely different text")
        assert len(words) <= 1

    def test_empty_transcript(self):
        assert align_transcript([span("A", 0, 0.02)], "") == []


def _aligned(words: list[tuple[str, float, float]]) -> list[WordTimes]:
    return [WordTimes(w, s, e, 1.0) for w, s, e in words]


class TestAlignFeatures:
    def test_names_contract(self):
        assert len(ALIGN_FEATURE_NAMES) == 5
        assert "align_aligned_word_frac" not in ALIGN_FEATURE_NAMES  # the leak guard

    def test_too_few_words_all_nan(self):
        result = extract_align_features(_aligned([("A", 0, 0.1)]), 10.0)
        assert not np.isfinite(result.values).any()

    def test_final_lengthening_direction(self):
        # read-ish: pre-pause words stretched (0.3s), mid-run short (0.1s)
        read = _aligned(
            [
                ("THE", 0.0, 0.1),
                ("QUICK", 0.12, 0.22),
                ("FOX", 0.24, 0.54),
                ("JUMPS", 0.80, 1.10),
                ("OVER", 1.12, 1.22),
                ("DOG", 1.24, 1.54),
            ]
        )
        # spontaneous-ish: pre-pause words not stretched
        spon = _aligned(
            [
                ("THE", 0.0, 0.1),
                ("QUICK", 0.12, 0.22),
                ("FOX", 0.24, 0.34),
                ("JUMPS", 0.80, 0.90),
                ("OVER", 0.92, 1.02),
                ("DOG", 1.04, 1.14),
            ]
        )
        d_read = extract_align_features(read, 5.0).to_dict()
        d_spon = extract_align_features(spon, 5.0).to_dict()
        assert (
            d_read["align_word_final_lengthening_ratio"]
            > d_spon["align_word_final_lengthening_ratio"]
        )

    def test_midword_silence_rate(self):
        words = _aligned(
            [
                ("THE", 0.0, 0.1),
                ("QUICK", 0.12, 0.22),
                ("FOX", 0.24, 0.54),
                ("JUMPS", 0.80, 1.10),
                ("OVER", 1.12, 1.22),
                ("DOG", 1.24, 1.54),
            ]
        )
        with_mw = extract_align_features(words, 2.0, midword_silences=[0.06, 0.08]).to_dict()
        assert with_mw["align_midword_silence_rate_per_min"] == pytest.approx(60.0)
