"""Character-level forced alignment via CTC (transformers-only, CPU-friendly).

This is the same mechanism WhisperX uses internally — a wav2vec2 CTC model
whose greedy output path *is* a per-frame character transcript — without
the whisperx dependency chain: torch + transformers ([deep] extra) suffice.

Why char-level matters here: word timestamps from ASR decoding (faster-
whisper) carry 50-100 ms jitter, which blurs inter-word gaps; the CTC
frame path resolves word boundaries to a 20 ms grid, and — uniquely —
reveals *mid-word* silences (articulation breaks inside a word), a
hesitation marker invisible to word-level timestamps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from voiceveritas.asr.transcribe import WordTimes

logger = logging.getLogger(__name__)

ALIGN_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-english"  # whisperx's align model
FAST_ALIGN_MODEL = "facebook/wav2vec2-base-960h"  # lighter fallback for clean audio
FRAME_MS = 20.0  # wav2vec2 output frame stride at 16 kHz

_CTC_CACHE: dict[str, tuple[object, object]] = {}


def _get_ctc(model_name: str):  # type: ignore[no-untyped-def]
    if model_name not in _CTC_CACHE:
        from transformers import AutoModelForCTC, Wav2Vec2Processor

        _CTC_CACHE[model_name] = (
            Wav2Vec2Processor.from_pretrained(model_name),
            AutoModelForCTC.from_pretrained(model_name),
        )
    return _CTC_CACHE[model_name]


@dataclass(frozen=True)
class CharSpan:
    """One CTC frame-run: a decoded character with its time span."""

    char: str
    start_s: float
    end_s: float


def frame_chars(samples: np.ndarray, sr: int, model_name: str = ALIGN_MODEL) -> list[CharSpan]:
    """CTC greedy path as time-stamped character runs (collapses repeats)."""
    import torch

    proc, model = _get_ctc(model_name)
    model.eval()  # type: ignore[union-attr]
    x = np.asarray(samples, dtype=np.float32)
    if sr != 16000:
        import librosa

        x = librosa.resample(x, orig_sr=sr, target_sr=16000)

    inputs = proc(x, sampling_rate=16000, return_tensors="pt")  # type: ignore[misc]
    with torch.inference_mode():
        logits = model(inputs.input_values).logits  # type: ignore[union-attr]
    ids = logits.argmax(dim=-1)[0].tolist()  # type: ignore[union-attr]

    # raw token chars — decode() would scrub the '|' word separator to a space
    vocab = proc.tokenizer.convert_ids_to_tokens(list(range(proc.tokenizer.vocab_size)))  # type: ignore[union-attr]

    # collapse repeats, drop blanks (CTC blank = <pad>, id 0); uppercase the
    # CTC path so lowercase-vocab models compare cleanly with transcripts
    spans: list[CharSpan] = []
    prev = -1
    for frame, token in enumerate(ids):
        if token == prev or token == 0:
            prev = token
            continue
        prev = token
        ch = vocab[token]
        if not ch or ch.startswith("<"):
            continue
        spans.append(
            CharSpan(ch.upper(), frame * FRAME_MS / 1000.0, (frame + 1) * FRAME_MS / 1000.0)
        )
    return spans


def frames_to_words(spans: list[CharSpan], sep_char: str = "|") -> list[WordTimes]:
    """Group character runs into words; the CTC word separator is '|'."""
    words: list[WordTimes] = []
    current_chars: list[CharSpan] = []

    def flush() -> None:
        if not current_chars:
            return
        text = "".join(c.char for c in current_chars)
        start = current_chars[0].start_s
        end = current_chars[-1].end_s
        # aligned words carry no decoder probability; 1.0 marks deterministic
        words.append(WordTimes(text, start, end, 1.0))
        current_chars.clear()

    for span in spans:
        if span.char == sep_char:
            flush()
        else:
            current_chars.append(span)
    flush()
    return words


def midword_silence_frames(spans: list[CharSpan], sep_char: str = "|") -> list[float]:
    """Blank-frame gaps *inside* word spans (frames the CTC path skipped).

    Returns gap durations in seconds; each is an articulation break within
    a word — e.g. "com-(pause)-puter" — a disfluency marker that word-level
    timestamps cannot see.
    """
    gaps: list[float] = []
    prev: CharSpan | None = None
    for span in spans:
        if prev is not None and span.char != sep_char and prev.char != sep_char:
            gap_ms = (span.start_s - prev.end_s) * 1000.0
            if gap_ms >= 2.0 * FRAME_MS:  # ≥2 skipped frames inside a word
                gaps.append(gap_ms / 1000.0)
        prev = span
    return gaps


def char_aligned_words(
    samples: np.ndarray, sr: int, model_name: str = ALIGN_MODEL
) -> list[WordTimes]:
    """Transcribe + align in one pass: precise (20 ms grid) word timestamps."""
    spans = frame_chars(samples, sr, model_name)
    return frames_to_words(spans)


# ----------------------------------------------------------------------
# forced alignment: known transcript -> CTC frame path (whisperx-style)
# ----------------------------------------------------------------------
def _normalize(text: str) -> str:
    """Uppercase letters + apostrophes + spaces, everything else dropped."""
    import re

    return re.sub(r"[^A-Z' ]", "", text.upper()).strip()


def align_hyp_indices(ref_chars: list[str], hyp_chars: list[str]) -> list[int]:
    """Edit-distance backtrace: hyp index matched to each ref char (-1 = none).

    Pure function (no models) — unit-testable. Substitution cost is lower
    than insertion+deletion so approximate CTC matches still bind.
    """
    n, m = len(ref_chars), len(hyp_chars)
    if n == 0:
        return []
    # DP over prefixes; cost sub=0 (match) / 1 (mismatch), ins/del = 1
    n + m + 1
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + (0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1)
            dp[i][j] = min(sub, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    # backtrace
    match = [-1] * n
    i, j = n, m
    while i > 0:
        if j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
            0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
        ):
            match[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        elif j > 0:
            j -= 1
        else:  # pragma: no cover - defensive
            break
    return match


def align_transcript(
    spans: list[CharSpan],
    transcript: str,
    sep_char: str = "|",
) -> list[WordTimes]:
    """Force a known transcript onto the CTC character path.

    The WhisperX recipe with zero extra dependencies: the robust ASR
    (faster-whisper) supplies the words; the CTC frame path supplies
    20 ms-precision timing; edit-distance backtrace binds them, absorbing
    CTC decode errors on hard audio (crosstalk, accents, noise).
    """
    norm = _normalize(transcript)
    if not norm:
        return []
    ref_words = norm.split()
    # hyp: character runs, separators replaced by space for word-agnostic char stream
    hyp_chars: list[str] = []
    hyp_span_idx: list[int] = []
    for idx, span in enumerate(spans):
        ch = " " if span.char == sep_char else span.char
        hyp_chars.append(ch)
        hyp_span_idx.append(idx)

    ref_stream = " ".join(ref_words)
    match = align_hyp_indices(list(ref_stream), hyp_chars)

    words: list[WordTimes] = []
    pos = 0  # char position in ref_stream
    for word in ref_words:
        end_pos = pos + len(word)
        span_ids = [
            hyp_span_idx[match[p]]
            for p in range(pos, end_pos)
            if match[p] >= 0 and hyp_chars[match[p]] != " "
        ]
        pos = end_pos + 1  # skip the space
        if not span_ids:
            continue  # word entirely unmatched (severe CTC failure)
        first, last = spans[min(span_ids)], spans[max(span_ids)]
        words.append(WordTimes(word, first.start_s, last.end_s, 1.0))
    return words
