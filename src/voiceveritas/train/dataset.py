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
) -> FeatureDataset:
    """Load manifests and extract (or fetch cached) features per utterance."""
    utterances: list[Utterance] = []
    for p in manifest_paths:
        utterances.extend(load_manifest(p))

    cache: dict[str, np.ndarray] = {}
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{FEATURE_VERSION}.npz"
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            cache = {k: cached[k] for k in cached.files}
            logger.info("loaded %d cached feature vectors from %s", len(cache), cache_path)

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
            if len(trimmed) / loaded.sample_rate < min_duration_s:
                n_skipped += 1
                continue
            vec = feature_fn(trimmed, loaded.sample_rate).values
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

    return FeatureDataset(
        X=np.vstack(rows),
        y=np.asarray(ys, dtype=int),
        utt_ids=utt_ids,
        speakers=speakers,
        datasets=datasets,
        splits=splits,
        durations=durations,
    )
