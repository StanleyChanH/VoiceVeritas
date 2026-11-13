"""Shared helpers: grouping LibriSpeech sentences into answer-length units."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf

INTER_SENTENCE_SILENCE_S = 0.35  # natural read pause between sentences
MIN_UNIT_S = 8.0
MAX_UNIT_S = 25.0


def group_flac_files_by_chapter(flac_files: list[Path]) -> dict[tuple[str, str], list[Path]]:
    """Group LibriSpeech FLACs by (speaker, chapter), keeping file order.

    LibriSpeech filenames are ``<spk>-<chapter>-<utt>.flac``; sorting by the
    utt number preserves reading order within a chapter.
    """
    groups: dict[tuple[str, str], list[Path]] = {}
    for f in flac_files:
        parts = f.stem.split("-")
        if len(parts) != 3:
            continue
        spk, chapter, _utt = parts
        groups.setdefault((spk, chapter), []).append(f)
    for files in groups.values():
        files.sort(key=lambda p: int(p.stem.split("-")[2]))
    return groups


def build_answer_units(
    files: list[Path],
    out_dir: Path,
    speaker_id: str,
    sr: int = 16000,
    rng: random.Random | None = None,
    max_units: int | None = None,
    target_range_s: tuple[float, float] = (12.0, 25.0),
) -> list[tuple[Path, float]]:
    """Concatenate consecutive sentences into answer-length wavs.

    Each unit closes once it passes a *randomized* target length (uniform in
    ``target_range_s``, seeded) — matching the randomized chunking used for
    the spontaneous side of the benchmark so utterance length cannot act as
    a dataset fingerprint (see docs/benchmark.md).

    Returns list of (wav_path, duration_s). Units from one chapter share a
    speaker, so speaker-disjoint splitting stays valid.
    """
    rng = rng or random.Random(0)
    out_dir.mkdir(parents=True, exist_ok=True)
    pad = np.zeros(int(INTER_SENTENCE_SILENCE_S * sr), dtype=np.float32)
    lo, hi = target_range_s
    units: list[tuple[Path, float]] = []
    buffer: list[np.ndarray] = []
    buffer_s = 0.0
    unit_idx = 0
    target = rng.uniform(lo, hi)

    def flush() -> None:
        nonlocal buffer, buffer_s, unit_idx, target
        if not buffer:
            return
        audio = np.concatenate(buffer)
        duration = len(audio) / sr
        if duration >= MIN_UNIT_S * 0.8:  # tolerate slightly short tails
            out_path = out_dir / f"{speaker_id}_u{unit_idx:03d}.wav"
            sf.write(str(out_path), audio, sr, subtype="PCM_16")
            units.append((out_path, duration))
        buffer, buffer_s, unit_idx = [], 0.0, unit_idx + 1
        target = rng.uniform(lo, hi)

    for f in files:
        data, file_sr = sf.read(str(f), dtype="float32")
        if file_sr != sr:
            import librosa

            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if buffer:
            buffer.append(pad)  # natural read pause between sentences
            buffer_s += INTER_SENTENCE_SILENCE_S
        buffer.append(data)
        buffer_s += len(data) / sr + INTER_SENTENCE_SILENCE_S
        if buffer_s >= target:
            flush()
        if max_units is not None and unit_idx >= max_units:
            break
    flush()
    return units
