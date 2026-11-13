"""ASR transcription and lexical/alignment features (optional ``[asr]`` extra).

Architecture note: transcription (heavy, model-backed) is strictly separated
from feature computation (pure functions over word timestamps), so the
feature layer is unit-testable without any ASR model, and transcripts are
cached once to disk for repeated experiments.
"""

from voiceveritas.asr.features import ASR_FEATURE_NAMES, extract_asr_features
from voiceveritas.asr.transcribe import WordTimes, transcribe_words

__all__ = [
    "ASR_FEATURE_NAMES",
    "WordTimes",
    "extract_asr_features",
    "transcribe_words",
]
