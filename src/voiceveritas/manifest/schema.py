"""Manifest data model.

A manifest is a JSONL file; each line describes one training/evaluation
utterance. Keeping manifests as plain JSONL makes them diffable,
versionable, and consumable by any tooling.

Label convention (binary, fixed project-wide):
    0 = spontaneous speech (negative)
    1 = read / recited speech (positive, the thing we detect)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any

LABEL_SPONTANEOUS = 0
LABEL_READ = 1

LABEL_NAMES = {LABEL_SPONTANEOUS: "spontaneous", LABEL_READ: "read"}

_SPLIT_TRAIN = "train"
_SPLIT_VAL = "val"
_SPLIT_TEST = "test"
VALID_SPLITS = frozenset({_SPLIT_TRAIN, _SPLIT_VAL, _SPLIT_TEST})


@dataclass
class Utterance:
    """One audio segment with its style label and provenance metadata."""

    utt_id: str
    path: str  # absolute path, or relative to the manifest file's directory
    label: int  # 0 = spontaneous, 1 = read
    duration: float  # seconds
    speaker_id: str
    dataset: str  # e.g. "librispeech", "sbcsae"
    split: str = _SPLIT_TRAIN
    text: str | None = None  # transcript when available
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in LABEL_NAMES:
            raise ValueError(
                f"label must be {LABEL_SPONTANEOUS} (spontaneous) or {LABEL_READ} (read), "
                f"got {self.label!r}"
            )
        if self.split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {self.split!r}")
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if not self.utt_id:
            raise ValueError("utt_id must be non-empty")

    @property
    def label_name(self) -> str:
        return LABEL_NAMES[self.label]

    def to_json(self) -> str:
        """Serialize to one JSONL line."""
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Utterance:
        """Deserialize from a dict, tolerating unknown fields (kept in extra)."""
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, value in record.items():
            if key in known:
                kwargs[key] = value
            else:
                extra[key] = value
        utt = cls(**kwargs)
        merged = {**extra, **utt.extra}
        utt.extra = merged
        return utt

    @classmethod
    def from_json(cls, line: str) -> Utterance:
        return cls.from_dict(json.loads(line))
