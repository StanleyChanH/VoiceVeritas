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
