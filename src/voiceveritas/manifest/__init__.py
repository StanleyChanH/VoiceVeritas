"""Dataset manifests: schema, IO, and speaker-disjoint splitting."""

from voiceveritas.manifest.io import load_manifest, manifest_stats, write_manifest
from voiceveritas.manifest.schema import LABEL_NAMES, LABEL_READ, LABEL_SPONTANEOUS, Utterance
from voiceveritas.manifest.split import speaker_disjoint_split

__all__ = [
    "LABEL_NAMES",
    "LABEL_READ",
    "LABEL_SPONTANEOUS",
    "Utterance",
    "load_manifest",
    "manifest_stats",
    "speaker_disjoint_split",
    "write_manifest",
]
