"""Build the held-out domain-shift evaluation corpus.

Pairs two sources the shipped models have NEVER seen in training:

- **SBCSAE** (OpenSLR 155) spontaneous conversations — label 0, unseen corpus
- **LibriSpeech test-clean** read speech — label 1, unseen speakers
  (verified disjoint from the dev-clean speakers used in training)

Everything lands in one eval-only manifest (all rows ``split=test``) so a
plain ``vv eval --split test`` scores it. See docs/in_domain.md for the
protocol rationale and the interpretation of results.

Usage:
    uv run python scripts/build_in_domain.py \
        --sbcsae-raw data/raw/sbcsae \
        --libri-raw data/raw/librispeech/test-clean \
        --out-dir data/processed/in_domain \
        --manifest data/manifests/in_domain.jsonl

Both corpora must be downloaded and extracted first
(``vv prepare sbcsae --download``; LibriSpeech test-clean tarball extracted
under ``data/raw/librispeech``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from voiceveritas.data_prep.librispeech import prepare_librispeech
from voiceveritas.data_prep.sbcsae import prepare_sbcsae
from voiceveritas.manifest import Utterance, load_manifest, manifest_stats, write_manifest

# Deterministic spread across the 60-conversation corpus (every 5th).
DEFAULT_CONVERSATIONS = [f"sbc{i:03d}" for i in range(1, 60, 5)]


def build(
    sbcsae_raw: Path,
    libri_raw: Path,
    out_dir: Path,
    manifest: Path,
    benchmark_manifest: Path,
    conversations: list[str],
    max_chunks: int,
    libri_speakers: int,
    libri_units: int,
    seed: int,
) -> list[Utterance]:
    spont = prepare_sbcsae(
        sbcsae_raw,
        out_dir=out_dir / "sbcsae",
        manifest_path=out_dir / "sbcsae.jsonl",
        conversations=conversations,
        max_chunks_per_conversation=max_chunks,
        seed=seed,
    )
    read = prepare_librispeech(
        libri_raw,
        out_dir=out_dir / "libritest",
        manifest_path=out_dir / "libritest.jsonl",
        max_speakers=libri_speakers,
        max_units_per_speaker=libri_units,
        seed=seed,
    )

    # Guard: the read side must be NEW speakers, not dev-clean retreads.
    seen = {u.speaker_id for u in load_manifest(benchmark_manifest)}
    overlap = {u.speaker_id for u in read} & seen
    if overlap:
        raise RuntimeError(
            f"speaker overlap with the training benchmark detected: {sorted(overlap)} — "
            "the domain-shift eval is void"
        )

    # Everything is evaluation data: normalize splits to test.
    entries = [*spont, *read]
    for i, u in enumerate(entries):
        entries[i] = Utterance(
            utt_id=u.utt_id,
            path=u.path,
            label=u.label,
            duration=u.duration,
            speaker_id=u.speaker_id,
            dataset=u.dataset,
            split="test",
        )
    write_manifest(manifest, entries)
    print(manifest_stats(entries))
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sbcsae-raw", type=Path, default=Path("data/raw/sbcsae"))
    ap.add_argument("--libri-raw", type=Path, default=Path("data/raw/librispeech/test-clean"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/in_domain"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifests/in_domain.jsonl"))
    ap.add_argument(
        "--benchmark-manifest",
        type=Path,
        default=Path("data/manifests/benchmark.jsonl"),
        help="training benchmark, used to verify speaker disjointness",
    )
    ap.add_argument(
        "--conversations",
        nargs="+",
        default=DEFAULT_CONVERSATIONS,
        help=f"SBCSAE conversation ids (default: every 5th of 60 -> {DEFAULT_CONVERSATIONS})",
    )
    ap.add_argument("--max-chunks", type=int, default=10, help="chunks per conversation")
    ap.add_argument("--libri-speakers", type=int, default=6)
    ap.add_argument("--libri-units", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    build(
        sbcsae_raw=args.sbcsae_raw,
        libri_raw=args.libri_raw,
        out_dir=args.out_dir,
        manifest=args.manifest,
        benchmark_manifest=args.benchmark_manifest,
        conversations=args.conversations,
        max_chunks=args.max_chunks,
        libri_speakers=args.libri_speakers,
        libri_units=args.libri_units,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
