"""Manifest JSONL IO and summary statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voiceveritas.manifest.schema import Utterance


def write_manifest(path: str | Path, utterances: list[Utterance]) -> Path:
    """Write utterances to a JSONL manifest (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with path.open("w", encoding="utf-8") as fh:
        for utt in utterances:
            if utt.utt_id in seen:
                raise ValueError(f"duplicate utt_id in manifest: {utt.utt_id!r}")
            seen.add(utt.utt_id)
            fh.write(utt.to_json() + "\n")
    return path


def load_manifest(path: str | Path, check_exists: bool = False) -> list[Utterance]:
    """Read a JSONL manifest.

    Args:
        path: manifest file path.
        check_exists: verify that each audio file exists (relative paths are
            resolved against the manifest's directory).
    """
    path = Path(path)
    utterances: list[Utterance] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                utt = Utterance.from_json(line)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: invalid manifest entry: {exc}") from exc
            if check_exists:
                audio_path = Path(utt.path)
                if not audio_path.is_absolute():
                    audio_path = path.parent / audio_path
                if not audio_path.exists():
                    raise FileNotFoundError(f"{path}:{lineno}: missing audio file {utt.path}")
            utterances.append(utt)
    return utterances


def manifest_stats(utterances: list[Utterance]) -> dict[str, Any]:
    """Summary statistics for a collection of utterances."""
    if not utterances:
        return {
            "num_utterances": 0,
            "num_speakers": 0,
            "total_hours": 0.0,
            "by_split": {},
            "by_label": {},
        }

    by_split: dict[str, dict[str, float]] = {}
    by_label: dict[str, dict[str, float]] = {}
    speakers: set[str] = set()
    total_seconds = 0.0

    for utt in utterances:
        speakers.add(utt.speaker_id)
        total_seconds += utt.duration
        for bucket, key in ((by_split, utt.split), (by_label, utt.label_name)):
            entry = bucket.setdefault(key, {"count": 0, "seconds": 0.0})
            entry["count"] += 1
            entry["seconds"] += utt.duration

    return {
        "num_utterances": len(utterances),
        "num_speakers": len(speakers),
        "total_hours": total_seconds / 3600,
        "by_split": by_split,
        "by_label": by_label,
    }
