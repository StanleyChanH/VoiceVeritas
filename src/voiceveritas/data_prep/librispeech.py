"""Prepare LibriSpeech (READ speech, label=1) into answer-length units.

LibriSpeech ships single *sentences* (2-15 s) as FLAC inside
``<split>/<speaker>/<chapter>/``. Interview answers are longer monologues, so
we concatenate consecutive sentences of a chapter (with a natural inter-
sentence pause) into 8-25 s units — the window our prosodic features
(pause distribution, rhythm) are designed for.
"""

from __future__ import annotations

import logging
import random
import tarfile
from pathlib import Path

from voiceveritas.data_prep.chunking import build_answer_units, group_flac_files_by_chapter
from voiceveritas.manifest import Utterance, write_manifest

logger = logging.getLogger(__name__)

LIBRISPEECH_URLS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
}

LABEL_READ = 1


def extract_tar(tar_path: str | Path, dest_dir: str | Path) -> Path:
    """Extract a .tar.gz archive (idempotent)."""
    dest_dir = Path(dest_dir)
    marker = dest_dir / ".extracted"
    if marker.exists():
        logger.info("archive already extracted: %s", dest_dir)
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest_dir, filter="data")
    marker.touch()
    return dest_dir


def prepare_librispeech(
    raw_dir: str | Path,
    out_dir: str | Path,
    manifest_path: str | Path,
    max_speakers: int = 20,
    max_units_per_speaker: int = 20,
    seed: int = 42,
) -> list[Utterance]:
    """Build answer-length read-speech units + manifest from extracted LibriSpeech.

    Args:
        raw_dir: directory containing the extracted ``LibriSpeech`` tree.
        out_dir: where unit wavs are written.
        manifest_path: output JSONL manifest.
        max_speakers: cap on distinct speakers (random sample, seeded).
        max_units_per_speaker: cap per speaker for class/hours balance.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flacs = sorted(raw_dir.rglob("*.flac"))
    if not flacs:
        raise FileNotFoundError(f"no .flac files under {raw_dir} — extract the tar first")
    groups = group_flac_files_by_chapter(flacs)

    # one entry per speaker: chapters concatenated in order
    by_speaker: dict[str, list[Path]] = {}
    for (spk, _chapter), files in sorted(groups.items()):
        by_speaker.setdefault(spk, []).extend(files)

    rng = random.Random(seed)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    speakers = speakers[:max_speakers]

    utterances: list[Utterance] = []
    for spk in speakers:
        files = by_speaker[spk]
        # keep chapter/utterance order: units read as natural running speech
        units = build_answer_units(
            files,
            out_dir=out_dir,
            speaker_id=spk,
            rng=rng,
            max_units=max_units_per_speaker,
        )
        for path, duration in units:
            utterances.append(
                Utterance(
                    utt_id=f"ls_{spk}_{path.stem.split('_')[-1]}",
                    path=str(path),
                    label=LABEL_READ,
                    duration=duration,
                    speaker_id=f"ls_{spk}",
                    dataset="librispeech",
                )
            )
        logger.info("librispeech speaker %s -> %d units", spk, len(units))

    write_manifest(manifest_path, utterances)
    logger.info(
        "librispeech manifest: %d units, %.2f h -> %s",
        len(utterances),
        sum(u.duration for u in utterances) / 3600,
        manifest_path,
    )
    return utterances
