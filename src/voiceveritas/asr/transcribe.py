"""faster-whisper transcription with word-level timestamps.

The model is loaded lazily and memoized per process. Word lists are cached
as JSON keyed by utterance identity so an experiment sweep transcribes the
corpus exactly once.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ASR_MODEL = "tiny.en"
ASR_CACHE_VERSION = "asr-tiny.en-v1"


@dataclass(frozen=True)
class WordTimes:
    """One transcribed word with its time span and probability."""

    word: str
    start: float
    end: float
    prob: float


_MODEL_CACHE: dict[str, object] = {}


def _get_model(model_size: str):  # type: ignore[no-untyped-def]
    if model_size not in _MODEL_CACHE:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError(
                "faster-whisper is required for ASR features — install with: "
                'uv sync --extra asr   (or: pip install ".[asr]")'
            ) from exc
        _MODEL_CACHE[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _MODEL_CACHE[model_size]


def transcribe_words(
    samples: np.ndarray,
    sr: int,
    model_size: str = ASR_MODEL,
    language: str = "en",
) -> list[WordTimes]:
    """Transcribe audio to a word list with timestamps (16 kHz float32 in)."""
    model = _get_model(model_size)
    x = np.asarray(samples, dtype=np.float32)
    if sr != 16000:
        import librosa

        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    segments, _info = model.transcribe(
        x, language=language, word_timestamps=True, vad_filter=False, beam_size=1
    )
    words: list[WordTimes] = []
    for segment in segments:
        for w in segment.words or []:
            words.append(
                WordTimes(w.word.strip(), float(w.start), float(w.end), float(w.probability))
            )
    return words


# ----------------------------------------------------------------------
# transcript cache
# ----------------------------------------------------------------------
def word_cache_key(utt_id: str, path: str) -> str:
    digest = hashlib.sha1(f"{utt_id}|{path}|{ASR_CACHE_VERSION}".encode()).hexdigest()[:20]
    return digest


def load_word_cache(cache_dir: str | Path) -> dict[str, list[WordTimes]]:
    cache_path = Path(cache_dir) / f"{ASR_CACHE_VERSION}.json"
    if not cache_path.exists():
        return {}
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    return {k: [WordTimes(**w) for w in v] for k, v in raw.items()}


def save_word_cache(cache_dir: str | Path, cache: dict[str, list[WordTimes]]) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ASR_CACHE_VERSION}.json"
    cache_path.write_text(
        json.dumps({k: [asdict(w) for w in v] for k, v in cache.items()}),
        encoding="utf-8",
    )
    return cache_path
