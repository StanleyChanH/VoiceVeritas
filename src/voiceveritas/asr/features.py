"""ASR-derived lexical/alignment features.

Pure functions of (word timestamps, duration) — see ``lexical.py`` for the
linguistic grounding. Deliberately **no duration-correlated features**
(e.g. word counts): the benchmark's shortcut postmortem (docs/benchmark.md)
showed collection artifacts leak dataset identity.

``ppl_gpt2`` (written-text perplexity — the project plan's "文本困惑度")
and ``wer_vs_reference`` are computed only when explicitly enabled / a
reference text is provided; otherwise they are NaN and LightGBM treats
them as missing.
"""

from __future__ import annotations

import contextlib
import itertools
import math

import numpy as np

from voiceveritas.asr.lexical import (
    count_contractions,
    count_core_fillers,
    count_discourse_markers,
    count_immediate_repetitions,
    interword_gaps,
    np_mean,
    np_std,
    pause_after_function_ratio,
    wer_vs_reference,
    word_rate_stats,
)
from voiceveritas.asr.transcribe import WordTimes
from voiceveritas.features import FeatureResult

_EPS = 1e-10

ASR_FEATURE_NAMES: tuple[str, ...] = (
    # hesitation / disfluency
    "filler_rate_per_min",
    "discourse_marker_rate_per_min",
    "repetition_rate_per_min",
    "contraction_ratio",
    # rate & pausing
    "word_rate_mean",
    "word_rate_std",
    "long_pause_rate_per_min",
    "pause_after_function_ratio",
    # lexical formality (writtenness proxies)
    "mean_word_len_chars",
    "type_token_ratio",
    "long_word_ratio",
    # decoder confidence (read speech transcribes cleaner)
    "asr_conf_mean",
    "asr_conf_std",
    # optional signals (NaN unless enabled)
    "ppl_gpt2",
    "wer_vs_reference",
)

# ----------------------------------------------------------------------
# alignment-grade features (char-level CTC alignment; see char_align.py)
#
# ⚠️ DEPLOYMENT TRAP (documented in docs/robustness.md): alignment quality
# is a *channel* property. On mixed-source corpora, spontaneous audio may
# align far worse than read audio — the resulting NaN-missingness pattern
# leaks corpus identity exactly like duration/asr_conf did. Enable only on
# same-channel data where both classes align comparably.
# ----------------------------------------------------------------------
ALIGN_FEATURE_NAMES: tuple[str, ...] = (
    "align_word_final_lengthening_ratio",
    "align_final_lengthening_cv",
    "align_gap_cv",
    "align_midword_silence_rate_per_min",
    "align_char_rate_cv",
)
# NOTE: alignment *coverage* is deliberately NOT a feature — it is the
# channel-quality leak itself. It is returned as diagnostics only.

_PPL_CACHE: dict[str, object] = {}


def _gpt2_perplexity(tokens_joined: str, model_name: str = "gpt2") -> float:
    """GPT-2 perplexity of the transcript (lower = more written/predictable)."""
    if model_name not in _PPL_CACHE:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        _PPL_CACHE[model_name] = (tok, model)
    tok, model = _PPL_CACHE[model_name]  # type: ignore[misc]

    import torch

    enc = tok(tokens_joined, return_tensors="pt", truncation=True, max_length=512)
    if enc.input_ids.shape[1] < 4:
        return math.nan
    with torch.inference_mode():
        logits = model(enc.input_ids).logits
    log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    targets = enc.input_ids[:, 1:]
    nll = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)
    return float(torch.exp(nll.mean()))


def extract_asr_features(
    words: list[WordTimes],
    duration_s: float,
    reference_text: str | None = None,
    compute_ppl: bool = False,
    ppl_model_name: str = "gpt2",
) -> FeatureResult:
    """Compute the fixed-order ASR feature vector from a word list."""
    tokens = [w.word.lower().strip(".,!?\"'") for w in words]
    tokens = [t for t in tokens if t]
    minutes = max(duration_s / 60.0, _EPS)

    n_words = len(tokens)
    if n_words == 0:
        nan = float("nan")
        return FeatureResult(
            values=np.array([nan] * len(ASR_FEATURE_NAMES), dtype=np.float64),
            names=ASR_FEATURE_NAMES,
        )

    rate_mean, rate_std = word_rate_stats(words)
    gaps = [g for g, _a, _b in interword_gaps(words)]
    long_pauses = sum(1 for g in gaps if g > 0.5)

    pafr = pause_after_function_ratio(words)
    char_lens = [len(t) for t in tokens]
    feats: dict[str, float] = {
        "filler_rate_per_min": count_core_fillers(tokens) / minutes,
        "discourse_marker_rate_per_min": count_discourse_markers(tokens) / minutes,
        "repetition_rate_per_min": count_immediate_repetitions(tokens) / minutes,
        "contraction_ratio": count_contractions(tokens) / n_words,
        "word_rate_mean": rate_mean if rate_mean is not None else float("nan"),
        "word_rate_std": rate_std if rate_std is not None else float("nan"),
        "long_pause_rate_per_min": long_pauses / minutes,
        "pause_after_function_ratio": float("nan") if pafr is None else pafr,
        "mean_word_len_chars": float(np_mean(char_lens)),
        "type_token_ratio": len(set(tokens)) / n_words,
        "long_word_ratio": sum(1 for c in char_lens if c >= 7) / n_words,
        "asr_conf_mean": float(np_mean([w.prob for w in words])),
        "asr_conf_std": float(np_std([w.prob for w in words])),
        "ppl_gpt2": float("nan"),
        "wer_vs_reference": float("nan"),
    }

    if compute_ppl:
        with contextlib.suppress(Exception):
            feats["ppl_gpt2"] = _gpt2_perplexity(" ".join(tokens), ppl_model_name)
    if reference_text:
        wer = wer_vs_reference(reference_text, tokens)
        feats["wer_vs_reference"] = float("nan") if wer is None else float(wer)

    vector = np.array([feats[name] for name in ASR_FEATURE_NAMES], dtype=np.float64)
    return FeatureResult(values=vector, names=ASR_FEATURE_NAMES)


def extract_align_features(
    aligned_words: list[WordTimes],
    duration_s: float,
    n_transcript_words: int | None = None,
    midword_silences: list[float] | None = None,
) -> FeatureResult:
    """Fine-timing features from char-aligned word timestamps.

    Linguistic grounding: phrase-**final lengthening** (words before a
    pause stretch) is how speakers mark prosodic boundaries — read speech
    lengthens *regularly* at clause ends (low CV), spontaneous speech
    lengthens irregularly wherever planning happens. Gap CV exploits the
    20 ms-grid precision that word-level timestamps cannot offer.
    Mid-word silences (articulation breaks) are visible only to char-level
    alignment.

    ``n_transcript_words`` is accepted for interface symmetry; alignment
    *coverage* is a diagnostic for callers, never a feature (see the
    deployment-trap note above).
    """
    del n_transcript_words  # diagnostics live with the caller
    minutes = max(duration_s / 60.0, _EPS)
    n = len(aligned_words)

    if n < 4:
        nan = float("nan")
        return FeatureResult(
            values=np.array([nan] * len(ALIGN_FEATURE_NAMES), dtype=np.float64),
            names=ALIGN_FEATURE_NAMES,
        )

    durs = [w.end - w.start for w in aligned_words]
    gaps = [b.start - a.end for a, b in itertools.pairwise(aligned_words)]

    pre_pause = [d for d, g in zip(durs[:-1], gaps, strict=False) if g >= 0.15]
    mid_run = [d for d, g in zip(durs[:-1], gaps, strict=False) if g < 0.05]
    len_ratio = (
        float(np_mean(pre_pause) / np_mean(mid_run)) if pre_pause and mid_run else float("nan")
    )
    len_cv = float(np_std(pre_pause) / np_mean(pre_pause)) if len(pre_pause) > 1 else float("nan")
    gap_cv = (
        float(np_std(gaps) / np_mean(gaps)) if len(gaps) > 1 and np_mean(gaps) > 0 else float("nan")
    )

    mw_rate = len(midword_silences or []) / minutes

    # chars/sec within speech runs (runs split at gaps >= 0.6s)
    runs: list[tuple[int, float]] = []
    run_chars, run_start = 0, aligned_words[0].start
    last_end = aligned_words[0].end
    for w, g in zip(aligned_words[1:], gaps, strict=False):
        if g >= 0.6:
            runs.append((run_chars, last_end - run_start))
            run_chars, run_start = 0, w.start
        else:
            run_chars += len(w.word)
        last_end = w.end
    runs.append((run_chars, last_end - run_start))
    rates = [c / t for c, t in runs if t > 0.2 and c > 2]
    char_rate_cv = float(np_std(rates) / np_mean(rates)) if len(rates) > 1 else float("nan")

    feats = {
        "align_word_final_lengthening_ratio": len_ratio,
        "align_final_lengthening_cv": len_cv,
        "align_gap_cv": gap_cv,
        "align_midword_silence_rate_per_min": mw_rate,
        "align_char_rate_cv": char_rate_cv,
    }
    vector = np.array([feats[name] for name in ALIGN_FEATURE_NAMES], dtype=np.float64)
    return FeatureResult(values=vector, names=ALIGN_FEATURE_NAMES)
