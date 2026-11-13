"""Audio loading utilities.

All downstream components (VAD, prosody features, neural backbones) consume
mono float32 waveforms resampled to a single target rate (16 kHz by default,
matching the conventions of wav2vec 2.0 and the interview-voice ecosystem).

Supported inputs: ``.wav`` / ``.flac`` (via soundfile) and ``.mp3`` / ``.m4a`` /
``.ogg`` (via librosa's ffmpeg/audioread fallback). Loading from raw bytes is
supported for the soundfile-backed formats (i.e. what the API receives as
uploads).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16000

AudioSource = str | Path | bytes | bytearray


class AudioLoadError(RuntimeError):
    """Raised when audio cannot be decoded or is empty."""


@dataclass(frozen=True)
class LoadedAudio:
    """A decoded mono waveform plus metadata."""

    samples: np.ndarray  # float32, shape (n,), range ~[-1, 1]
    sample_rate: int
    source: str  # description of origin, for logs/errors

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.samples) / self.sample_rate


def _decode_native(source: AudioSource) -> tuple[np.ndarray, int, str]:
    """Decode audio at its native sample rate. Returns (multi-channel, sr, desc)."""
    if isinstance(source, (bytes, bytearray)):
        data = sf.read(io.BytesIO(bytes(source)), dtype="float32", always_2d=True)
        frames, sr = data
        return frames.T, int(sr), "<uploaded bytes>"

    path = Path(source)
    desc = str(path)
    if not path.exists():
        raise AudioLoadError(f"audio file does not exist: {desc}")

    try:
        frames, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return frames.T, int(sr), desc
    except Exception as exc:
        logger.debug("soundfile failed on %s (%s); falling back to librosa", desc, exc)

    # ffmpeg-backed fallback (mp3, m4a, ogg, ...)
    try:
        import librosa

        y, sr = librosa.load(str(path), sr=None, mono=False)
    except Exception as exc:
        raise AudioLoadError(
            f"could not decode {desc!r}. For mp3/m4a/ogg support, install ffmpeg "
            f"and make sure it is on PATH. Underlying error: {exc}"
        ) from exc

    if y.ndim == 1:
        y = y[np.newaxis, :]
    return np.ascontiguousarray(y, dtype=np.float32), int(sr), desc


def load_audio(
    source: AudioSource,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    mono: bool = True,
) -> LoadedAudio:
    """Load audio from a path or bytes, resampled and mono-mixed.

    Args:
        source: file path or raw encoded bytes (wav/flac only for bytes).
        sample_rate: target sample rate (default 16 kHz).
        mono: downmix to a single channel by averaging.

    Returns:
        LoadedAudio with float32 samples at ``sample_rate``.

    Raises:
        AudioLoadError: if decoding fails or produces no samples.
    """
    multichannel, native_sr, desc = _decode_native(source)

    if multichannel.ndim != 2:
        raise AudioLoadError(f"unexpected audio shape {multichannel.shape} from {desc}")

    if mono and multichannel.shape[0] > 1:
        y = multichannel.mean(axis=0)
    else:
        y = multichannel if not mono else multichannel[0]
    y = np.ascontiguousarray(y, dtype=np.float32)

    if y.size == 0:
        raise AudioLoadError(f"decoded audio contains zero samples: {desc}")

    if native_sr != sample_rate:
        import librosa

        y = librosa.resample(y, orig_sr=native_sr, target_sr=sample_rate, res_type="soxr_hq")
        y = np.ascontiguousarray(y, dtype=np.float32)

    return LoadedAudio(samples=y, sample_rate=sample_rate, source=desc)


def save_wav(samples: np.ndarray, path: str | Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Path:
    """Write a mono float32 waveform to a 16-bit PCM wav file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(samples, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError(f"save_wav expects mono 1-D samples, got shape {y.shape}")
    sf.write(str(path), y, sample_rate, subtype="PCM_16")
    return path
