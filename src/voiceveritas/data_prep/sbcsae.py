"""Prepare SBCSAE (SPONTANEOUS conversation speech, label=0) — eval-only.

The Santa Barbara Corpus of Spoken American English (OpenSLR 155) is 60
naturally-occurring ~20-minute conversations: the project's *held-out
domain-shift corpus*. Models are trained on LibriSpeech+AMI and never see
SBCSAE; scoring SBCSAE against unseen-speaker read speech answers the
question every reviewer asks — did the classifier learn *style* or just the
AMI meeting channel? (see docs/in_domain.md)

Two deliberate policy choices:

- **Eval-only (split=test)**: SBCSAE is never mixed into training manifests,
  so it stays a clean generalization probe. Its CC BY-ND license also makes
  derived training artifacts awkward to redistribute — as an untouched test
  set it is only ever referenced, not remixed.
- **Conversation = speaker unit**: like AMI sessions, each recording becomes
  one ``speaker_id``. Recordings contain several human speakers; the
  approximation (documented in docs/data.md) prevents channel/recording
  leakage rather than exact speaker identity.
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path

from voiceveritas.audio import load_audio, save_wav
from voiceveritas.data_prep.ami import chunk_session
from voiceveritas.manifest import Utterance, write_manifest

logger = logging.getLogger(__name__)

# Mirrors in tried order (OpenSLR SLR155, 6.1 GB tar.gz).
SBCSAE_URLS: list[str] = [
    "https://openslr.elda.org/resources/155/SBCSAE.tar.gz",
    "https://openslr.trmal.net/resources/155/SBCSAE.tar.gz",
    "https://openslr.magicdatatech.com/resources/155/SBCSAE.tar.gz",
    "https://www.openslr.org/resources/155/SBCSAE.tar.gz",
]

LABEL_SPONTANEOUS = 0

_CONV_RE = re.compile(r"(?:sbc)?0*(\d+)", re.IGNORECASE)


def conversation_id(path: Path) -> str:
    """Normalize a corpus filename (SBC001.wav / SBC0009.mp3) → ``sbc001``."""
    m = _CONV_RE.search(path.stem)
    return f"sbc{int(m.group(1)):03d}" if m else path.stem.lower()


def find_conversations(raw_dir: str | Path) -> dict[str, Path]:
    """Map conversation id → audio path for every SBC* file under ``raw_dir``."""
    raw_dir = Path(raw_dir)
    found: dict[str, Path] = {}
    for pattern in ("SBC*.wav", "SBC*.mp3", "sbc*.wav", "sbc*.mp3"):
        for p in sorted(raw_dir.rglob(pattern)):
            found.setdefault(conversation_id(p), p)
    return found


def prepare_sbcsae(
    raw_dir: str | Path,
    out_dir: str | Path,
    manifest_path: str | Path,
    conversations: list[str] | None = None,
    max_chunks_per_conversation: int = 25,
    sample_rate: int = 16000,
    seed: int = 42,
) -> list[Utterance]:
    """Chunk SBCSAE conversation audio (already downloaded) into a manifest.

    Args:
        raw_dir: directory holding the extracted corpus (SBC*.wav anywhere
            below it; the OpenSLR tarball layout is handled by rglob).
        out_dir: output directory for chunk wavs.
        manifest_path: output JSONL manifest.
        conversations: conversation ids (e.g. ``["sbc001", "sbc009"]``) to
            use; default: all found.
        max_chunks_per_conversation: cap per conversation for hours balance.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    available = find_conversations(raw_dir)
    if not available:
        raise FileNotFoundError(
            f"no SBC*.wav audio under {raw_dir} — extract the OpenSLR 155 tarball first"
        )
    selected = sorted(conversations) if conversations else sorted(available)
    rng = random.Random(seed)

    utterances: list[Utterance] = []
    for conv in selected:
        src = available.get(conv)
        if src is None:
            logger.warning("conversation %s not found under %s — skipping", conv, raw_dir)
            continue

        loaded = load_audio(src, sample_rate=sample_rate)
        chunks = chunk_session(loaded.samples, loaded.sample_rate, rng=rng)
        rng.shuffle(chunks)
        chunks = chunks[:max_chunks_per_conversation]

        for idx, (start, end) in enumerate(sorted(chunks)):
            i0, i1 = int(start * sample_rate), int(end * sample_rate)
            seg = loaded.samples[i0:i1]
            path = out_dir / f"{conv}_c{idx:03d}.wav"
            save_wav(seg, path, sample_rate)
            utterances.append(
                Utterance(
                    utt_id=f"sbcsae_{conv}_c{idx:03d}",
                    path=str(path),
                    label=LABEL_SPONTANEOUS,
                    duration=(i1 - i0) / sample_rate,
                    speaker_id=f"sbcsae_{conv}",
                    dataset="sbcsae",
                    split="test",  # eval-only by design (see module docstring)
                )
            )
        logger.info("sbcsae conversation %s -> %d chunks", conv, len(chunks))

    if not utterances:
        raise FileNotFoundError(f"none of the requested conversations found under {raw_dir}")
    write_manifest(manifest_path, utterances)
    logger.info(
        "sbcsae manifest: %d chunks, %.2f h (eval-only, split=test) -> %s",
        len(utterances),
        sum(u.duration for u in utterances) / 3600,
        manifest_path,
    )
    return utterances


def download_sbcsae(raw_dir: str | Path, retries: int = 2) -> Path:
    """Download the corpus tarball from the first reachable OpenSLR mirror."""
    from voiceveritas.data_prep.download import download_file

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in SBCSAE_URLS:
        try:
            return download_file(url, raw_dir / "SBCSAE.tar.gz", retries=retries)
        except Exception as exc:  # next mirror
            logger.warning("mirror failed %s: %s", url, exc)
            last_error = exc
    raise RuntimeError(f"all SBCSAE mirrors failed: {last_error}")
