"""Tests for ASR lexical/alignment features (pure functions, no ASR model)."""

from __future__ import annotations

import numpy as np
import pytest

from voiceveritas.asr import ASR_FEATURE_NAMES, WordTimes, extract_asr_features
from voiceveritas.asr.features import _gpt2_perplexity  # noqa: F401 - import guard
from voiceveritas.asr.lexical import (
    count_contractions,
    count_core_fillers,
    count_discourse_markers,
    count_immediate_repetitions,
    interword_gaps,
    pause_after_function_ratio,
    wer_vs_reference,
    word_edit_distance,
    word_rate_stats,
)


def w(word: str, start: float, end: float, prob: float = 0.9) -> WordTimes:
    return WordTimes(word=word, start=start, end=end, prob=prob)


def readish_words() -> list[WordTimes]:
    """Metronomic, formal, no fillers; pauses after content words."""
    text = "The comprehensive framework demonstrates considerable organizational efficiency"
    words = text.split()
    out, t = [], 0.0
    for i, word in enumerate(words):
        out.append(w(word, t, t + 0.35))
        t += 0.35 + (0.45 if i % 2 else 0.05)  # regular clause-ish pauses
    return out


def spontaneousish_words() -> list[WordTimes]:
    """Hesitant, fillers, repetitions; pauses after function words."""
    seq = [
        ("i", 0.02),
        ("was", 0.15),
        ("um", 0.1),
        ("thinking", 0.4),
        ("about", 0.2),
        ("the", 0.15),
        ("the", 0.15),
        ("thing", 0.3),
        ("you", 0.1),
        ("know", 0.1),
        ("it's", 0.2),
        ("like", 0.15),
        ("hard", 0.25),
    ]
    out, t = [], 0.0
    for word, dur in seq:
        out.append(w(word, t, t + dur, prob=0.75))
        t += dur + 0.28  # planning pauses (mid-phrase)
    return out


class TestLexicalHelpers:
    def test_core_filler_counting(self):
        assert count_core_fillers(["um", "I", "um", "was", "uh"]) == 3
        assert count_core_fillers(["under", "umbrella"]) == 0

    def test_discourse_markers_incl_bigrams(self):
        assert count_discourse_markers(["you", "know", "like"]) == 2
        assert count_discourse_markers(["well", "basically"]) == 2

    def test_contraction_detection(self):
        assert count_contractions(["don't", "it's", "do", "not"]) == 2
        assert count_contractions(["cannot", "shall"]) == 0

    def test_immediate_repetitions(self):
        assert count_immediate_repetitions(["the", "the", "thing"]) == 1
        assert count_immediate_repetitions(["the", "a", "the"]) == 0

    def test_word_edit_distance(self):
        assert word_edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0
        assert word_edit_distance(["a", "b"], ["a", "x"]) == 1
        assert word_edit_distance([], ["a", "b"]) == 2

    def test_wer_vs_reference(self):
        assert wer_vs_reference("hello world", ["hello", "world"]) == 0.0
        assert wer_vs_reference("hello world", ["hello"]) == 0.5
        assert wer_vs_reference("", ["hello"]) is None

    def test_interword_gaps(self):
        gaps = interword_gaps([w("a", 0.0, 0.2), w("b", 0.5, 0.7), w("c", 0.9, 1.0)])
        assert [round(g, 2) for g, _, _ in gaps] == [0.30, 0.20]

    def test_pause_after_function_direction(self):
        read = pause_after_function_ratio(
            [w("framework", 0, 0.3), w("demonstrates", 0.8, 1.1)], min_gap_s=0.15
        )
        spon = pause_after_function_ratio([w("i", 0, 0.1), w("think", 0.4, 0.7)], min_gap_s=0.15)
        assert read == 0.0 and spon == 1.0

    def test_word_rate_stats(self):
        words = [w("a", 0.0, 0.3), w("b", 0.4, 0.7), w("c", 3.0, 3.3)]  # 2 runs
        mean, std = word_rate_stats(words)
        assert mean is not None and std is not None
        assert mean == pytest.approx((2 / 0.7 + 1 / 0.3) / 2, rel=0.2)


class TestAsrFeatures:
    def test_names_contract(self):
        assert len(ASR_FEATURE_NAMES) == len(set(ASR_FEATURE_NAMES))
        assert ASR_FEATURE_NAMES[-2:] == ("ppl_gpt2", "wer_vs_reference")

    def test_empty_words_all_nan(self):
        result = extract_asr_features([], 10.0)
        assert result.values.shape == (len(ASR_FEATURE_NAMES),)
        assert not np.isfinite(result.values).any()

    def test_directional_validity(self):
        """The caricatured styles must differ on the intended families."""
        read = extract_asr_features(readish_words(), 15.0).to_dict()
        spon = extract_asr_features(spontaneousish_words(), 15.0).to_dict()

        assert spon["filler_rate_per_min"] > read["filler_rate_per_min"]
        assert spon["discourse_marker_rate_per_min"] > read["discourse_marker_rate_per_min"]
        assert spon["contraction_ratio"] > read["contraction_ratio"]
        assert spon["repetition_rate_per_min"] > read["repetition_rate_per_min"]
        assert spon["pause_after_function_ratio"] > read["pause_after_function_ratio"]
        assert read["mean_word_len_chars"] > spon["mean_word_len_chars"]
        assert read["asr_conf_mean"] > spon["asr_conf_mean"]
        # optional signals stay NaN unless enabled
        assert np.isnan(read["ppl_gpt2"]) and np.isnan(read["wer_vs_reference"])

    def test_wer_populated_with_reference(self):
        words = [w("hello", 0, 0.4), w("world", 0.5, 0.9)]
        d = extract_asr_features(words, 2.0, reference_text="hello world").to_dict()
        assert d["wer_vs_reference"] == 0.0

    def test_ppl_disabled_when_flag_false(self):
        # compute_ppl=True would load GPT-2; guard that the default path
        # never touches transformers
        result = extract_asr_features(readish_words(), 10.0, compute_ppl=False)
        assert np.isnan(result.to_dict()["ppl_gpt2"])
