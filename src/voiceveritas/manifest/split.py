"""Speaker-disjoint dataset splitting.

Speakers must not appear in more than one split: a model that memorizes a
speaker's voice (or recording conditions) will leak test information and
report inflated accuracy. We therefore assign *speakers*, not utterances.

The split is stratified by label so that each split keeps roughly the same
read/spontaneous balance as the full collection.
"""

from __future__ import annotations

import random
from collections import defaultdict

from voiceveritas.manifest.schema import Utterance


def speaker_disjoint_split(
    utterances: list[Utterance],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> list[Utterance]:
    """Assign ``split`` in-place (a copy is not made) via speaker-level greedy packing.

    For each label, speakers are shuffled (deterministically given ``seed``)
    and greedily packed into train/val/test until each split reaches its
    target share of that label's utterances. Remaining speakers go to test.

    Args:
        utterances: list with ``label`` and ``speaker_id`` populated.
        train_ratio: fraction of utterances for training (by label).
        val_ratio: fraction for validation; test gets the remainder.
        seed: RNG seed for reproducibility.

    Returns:
        The same list, with ``split`` fields updated.
    """
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError(f"ratios out of range: train={train_ratio}, val={val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    rng = random.Random(seed)

    # label -> speaker -> [utt indices]; a speaker is assumed single-label
    per_label_speakers: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for idx, utt in enumerate(utterances):
        per_label_speakers[utt.label][utt.speaker_id].append(idx)

    for _label, speakers in per_label_speakers.items():
        total = sum(len(v) for v in speakers.values())
        targets = {
            "train": total * train_ratio,
            "val": total * val_ratio,
        }
        order = sorted(speakers.keys())
        rng.shuffle(order)

        speaker_split: dict[str, str] = {}
        counts = {"train": 0, "val": 0}
        for phase in ("train", "val"):
            for speaker in order:
                if speaker in speaker_split:
                    continue
                if counts[phase] >= targets[phase]:
                    break
                speaker_split[speaker] = phase
                counts[phase] += len(speakers[speaker])
        for speaker in order:
            if speaker not in speaker_split:
                speaker_split[speaker] = "test"

        for speaker, phase in speaker_split.items():
            for idx in speakers[speaker]:
                utterances[idx].split = phase

    return utterances
