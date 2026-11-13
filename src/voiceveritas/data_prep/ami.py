"""Prepare AMI Meeting Corpus (SPONTANEOUS speech, label=0).

Downloads per-session ``Mix-Headset`` wavs from the Edinburgh mirror and
chunks them into utterance-like segments with our own energy VAD —
dogfooding the detector's own pause analysis on real meeting audio.

Caveat (documented in docs/data.md): sessions contain multiple speakers; we
treat each *session* as the speaker unit for speaker-disjoint splitting.
This still prevents room/channel/recording leakage across splits, which is
the dominant short-range leakage risk.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np

from voiceveritas.audio import load_audio, save_wav
from voiceveritas.manifest import Utterance, write_manifest
from voiceveritas.vad import energy_vad

logger = logging.getLogger(__name__)

AMI_BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
LABEL_SPONTANEOUS = 0

# A balanced default selection: 16 sessions across the three AMI series.
AMI_SESSIONS: list[str] = [
    "ES2002a",
    "ES2002b",
    "ES2002c",
    "ES2002d",
    "ES2003a",
    "ES2003b",
    "ES2003c",
    "ES2003d",
    "IS1000a",
    "IS1000b",
    "IS1000c",
    "IS1000d",
    "TS3003a",
    "TS3003b",
    "TS3003c",
    "TS3003d",
]


def ami_session_url(session: str) -> str:
    return f"{AMI_BASE}/{session}/audio/{session}.Mix-Headset.wav"


def chunk_session(
    samples: np.ndarray,
    sr: int,
    min_chunk_s: float = 8.0,
    max_chunk_s: float = 25.0,
    rng: random.Random | None = None,
) -> list[tuple[float, float]]:
    """Cut VAD speech regions into utterance-like [start, end] chunks.

    Each chunk gets a *randomized* target length (uniform in
    [min+4, max], seeded) instead of a constant cap: meeting speech has long
    continuous runs, so any fixed cap would put every chunk at exactly that
    length — a dataset fingerprint the classifier would happily exploit
    (see docs/benchmark.md).
    """
    rng = rng or random.Random(0)
    vad = energy_vad(samples, sr)
    lo = min_chunk_s + 4.0
    hi = max_chunk_s
    if hi <= lo:
        lo, hi = min_chunk_s, max_chunk_s

    chunks: list[tuple[float, float]] = []
    cursor = 0.0
    total = len(samples) / sr
    speech = np.zeros(int(total * 1000) + 2, dtype=bool)  # 1 ms grid
    for start, end in vad.segments:
        speech[int(start * 1000) : int(end * 1000)] = True

    while cursor < total:
        # skip leading silence
        i = int(cursor * 1000)
        while i < len(speech) and not speech[i]:
            i += 1
        if i >= len(speech):
            break
        start_s = i / 1000.0
        target = rng.uniform(lo, hi)
        end_s = min(start_s + target, total)
        chunks.append((start_s, end_s))
        cursor = end_s + 0.05  # small inter-chunk offset

    # drop chunks that ended up too short (audio tail)
    return [(s, e) for s, e in chunks if e - s >= min_chunk_s * 0.6]


def prepare_ami(
    raw_dir: str | Path,
    out_dir: str | Path,
    manifest_path: str | Path,
    sessions: list[str] | None = None,
    max_chunks_per_session: int = 25,
    sample_rate: int = 16000,
    seed: int = 42,
) -> list[Utterance]:
    """Chunk AMI session wavs (already downloaded) into a manifest.

    Args:
        raw_dir: directory holding ``<session>.Mix-Headset.wav`` files.
        out_dir: output directory for chunk wavs.
        manifest_path: output JSONL manifest.
        sessions: session IDs to use (default: AMI_SESSIONS).
        max_chunks_per_session: cap per session for hours balance.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = sessions or AMI_SESSIONS
    rng = random.Random(seed)

    utterances: list[Utterance] = []
    for session in sessions:
        candidates = [
            raw_dir / f"{session}.Mix-Headset.wav",
            raw_dir / session / "audio" / f"{session}.Mix-Headset.wav",
        ]
        src = next((c for c in candidates if c.exists()), None)
        if src is None:
            logger.warning("session %s not found under %s — skipping", session, raw_dir)
            continue

        loaded = load_audio(src, sample_rate=sample_rate)
        chunks = chunk_session(loaded.samples, loaded.sample_rate, rng=rng)
        rng.shuffle(chunks)
        chunks = chunks[:max_chunks_per_session]

        for idx, (start, end) in enumerate(sorted(chunks)):
            i0, i1 = int(start * sample_rate), int(end * sample_rate)
            seg = loaded.samples[i0:i1]
            path = out_dir / f"{session}_c{idx:03d}.wav"
            save_wav(seg, path, sample_rate)
            utterances.append(
                Utterance(
                    utt_id=f"ami_{session}_c{idx:03d}",
                    path=str(path),
                    label=LABEL_SPONTANEOUS,
                    duration=(i1 - i0) / sample_rate,
                    speaker_id=f"ami_{session}",
                    dataset="ami",
                )
            )
        logger.info("ami session %s -> %d chunks", session, len(chunks))

    if not utterances:
        raise FileNotFoundError(f"no AMI sessions found under {raw_dir}")
    write_manifest(manifest_path, utterances)
    logger.info(
        "ami manifest: %d chunks, %.2f h -> %s",
        len(utterances),
        sum(u.duration for u in utterances) / 3600,
        manifest_path,
    )
    return utterances


def download_ami_sessions(
    raw_dir: str | Path,
    sessions: list[str],
    retries: int = 2,
) -> list[Path]:
    """Download session Mix-Headset wavs; returns successfully downloaded paths."""
    from voiceveritas.data_prep.download import download_file

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for session in sessions:
        url = ami_session_url(session)
        try:
            dest = raw_dir / f"{session}.Mix-Headset.wav"
            downloaded.append(download_file(url, dest, retries=retries))
        except Exception as exc:
            logger.warning("could not download %s: %s", session, exc)
    return downloaded
