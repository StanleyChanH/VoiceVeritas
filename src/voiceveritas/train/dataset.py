"""Feature dataset construction with a disk cache.

Extracting prosodic features is CPU-bound (Praat pitch tracking, spectral
analysis). To keep training iterations snappy we cache feature vectors on
disk keyed by (utt_id, feature-version, duration, sr) — a stale cache entry
can never silently mismatch a changed extractor.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voiceveritas import __version__
from voiceveritas.audio import load_audio
from voiceveritas.features import FeatureResult, extract_prosody
from voiceveritas.manifest import Utterance, load_manifest

logger = logging.getLogger(__name__)

FEATURE_VERSION = "prosody-v1"


def _default_transcriber(samples: np.ndarray, sr: int) -> list:
    """Real transcription (faster-whisper); wrapped for lazy import + typing."""
    from voiceveritas.asr.transcribe import transcribe_words

    return transcribe_words(samples, sr)


@dataclass
class FeatureDataset:
    """Feature matrix plus provenance columns, split-aware."""

    X: np.ndarray  # (n, len(FEATURE_NAMES)) float64, NaN allowed
    y: np.ndarray  # (n,) int
    utt_ids: list[str]
    speakers: list[str]
    datasets: list[str]
    splits: list[str]
    durations: list[float]

    def by_split(self, split: str) -> FeatureDataset:
        idx = [i for i, s in enumerate(self.splits) if s == split]
        return FeatureDataset(
            X=self.X[idx],
            y=self.y[idx],
            utt_ids=[self.utt_ids[i] for i in idx],
            speakers=[self.speakers[i] for i in idx],
            datasets=[self.datasets[i] for i in idx],
            splits=[self.splits[i] for i in idx],
            durations=[self.durations[i] for i in idx],
        )


def _cache_key(utt: Utterance, feature_version: str) -> str:
    digest = hashlib.sha1(
        f"{utt.utt_id}|{utt.path}|{utt.duration:.3f}|{feature_version}|{__version__}".encode()
    ).hexdigest()[:20]
    return digest


def _middle_trim(samples: np.ndarray, sr: int, max_duration_s: float) -> np.ndarray:
    n_max = int(max_duration_s * sr)
    if len(samples) <= n_max:
        return samples
    start = (len(samples) - n_max) // 2
    return samples[start : start + n_max]


def build_feature_dataset(
    manifest_paths: Sequence[str | Path],
    cache_dir: str | Path | None = None,
    max_duration_s: float = 30.0,
    min_duration_s: float = 2.0,
    sample_rate: int = 16000,
    feature_fn: Callable[..., FeatureResult] = extract_prosody,
    with_asr: bool = False,
    transcriber: Callable[..., list] | None = None,
    compute_ppl: bool = False,
) -> FeatureDataset:
    """Load manifests and extract (or fetch cached) features per utterance.

    With ``with_asr=True`` the prosody vector is concatenated with the ASR
    lexical/alignment vector (transcripts cached under ``cache_dir``).
    ``transcriber`` is injectable for tests.
    """
    utterances: list[Utterance] = []
    for p in manifest_paths:
        utterances.extend(load_manifest(p))

    cache: dict[str, np.ndarray] = {}
    cache_path = None
    # combined prosody+ASR vectors get their own cache namespace: sharing
    # one with prosody-only vectors would silently mix dimensions
    version = "prosody-asr-v1" if with_asr else FEATURE_VERSION
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{version}.npz"
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            cache = {k: cached[k] for k in cached.files}
            logger.info("loaded %d cached feature vectors from %s", len(cache), cache_path)

    word_cache: dict[str, list] = {}
    if with_asr and cache_dir is not None:
        from voiceveritas.asr.transcribe import load_word_cache

        word_cache = dict(load_word_cache(cache_dir))

    rows: list[np.ndarray] = []
    ys: list[int] = []
    utt_ids: list[str] = []
    speakers: list[str] = []
    datasets: list[str] = []
    splits: list[str] = []
    durations: list[float] = []

    n_skipped = 0
    for utt in utterances:
        key = _cache_key(utt, FEATURE_VERSION)
        vec = cache.get(key)
        if vec is None:
            try:
                loaded = load_audio(utt.path, sample_rate=sample_rate)
            except Exception as exc:
                logger.warning("skipping %s (%s): %s", utt.utt_id, utt.path, exc)
                n_skipped += 1
                continue
            trimmed = _middle_trim(loaded.samples, loaded.sample_rate, max_duration_s)
            trim_duration = len(trimmed) / loaded.sample_rate
            if trim_duration < min_duration_s:
                n_skipped += 1
                continue
            vec = feature_fn(trimmed, loaded.sample_rate).values
            if with_asr:
                from voiceveritas.asr.features import extract_asr_features
                from voiceveritas.asr.transcribe import word_cache_key

                wkey = word_cache_key(utt.utt_id, utt.path)
                words = word_cache.get(wkey)
                if words is None:
                    t_fn = transcriber or _default_transcriber
                    words = t_fn(trimmed, loaded.sample_rate)
                    word_cache[wkey] = words
                asr_vec = extract_asr_features(words, trim_duration, compute_ppl=compute_ppl).values
                vec = np.concatenate([vec, asr_vec])
            cache[key] = vec
        rows.append(np.asarray(vec, dtype=np.float64))
        ys.append(utt.label)
        utt_ids.append(utt.utt_id)
        speakers.append(utt.speaker_id)
        datasets.append(utt.dataset)
        splits.append(utt.split)
        durations.append(utt.duration)

    if n_skipped:
        logger.warning(
            "skipped %d/%d utterances (unloadable or too short)", n_skipped, len(utterances)
        )
    if not rows:
        raise ValueError("no usable utterances found in manifests")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **cache)  # type: ignore[arg-type]
        logger.info("cached %d feature vectors at %s", len(cache), cache_path)

    if with_asr and cache_dir is not None and word_cache:
        from voiceveritas.asr.transcribe import save_word_cache

        save_word_cache(cache_dir, word_cache)  # type: ignore[arg-type]

    return FeatureDataset(
        X=np.vstack(rows),
        y=np.asarray(ys, dtype=int),
        utt_ids=utt_ids,
        speakers=speakers,
        datasets=datasets,
        splits=splits,
        durations=durations,
    )
