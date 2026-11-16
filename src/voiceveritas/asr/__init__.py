"""ASR transcription and lexical/alignment features (optional ``[asr]`` extra).

Architecture note: transcription (heavy, model-backed) is strictly separated
from feature computation (pure functions over word timestamps), so the
feature layer is unit-testable without any ASR model, and transcripts are
cached once to disk for repeated experiments.
"""

from voiceveritas.asr.char_align import CharSpan, align_transcript, char_aligned_words, frame_chars
from voiceveritas.asr.features import (
    ALIGN_FEATURE_NAMES,
    ASR_FEATURE_NAMES,
    extract_align_features,
    extract_asr_features,
)
from voiceveritas.asr.transcribe import WordTimes, transcribe_words

__all__ = [
    "ALIGN_FEATURE_NAMES",
    "ASR_FEATURE_NAMES",
    "CharSpan",
    "WordTimes",
    "align_transcript",
    "char_aligned_words",
    "extract_align_features",
    "extract_asr_features",
    "frame_chars",
    "transcribe_words",
]
