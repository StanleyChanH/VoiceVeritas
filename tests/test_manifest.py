"""Tests for manifest schema, IO, and speaker-disjoint splitting."""

from __future__ import annotations

import json

import pytest

from voiceveritas.manifest import (
    LABEL_READ,
    LABEL_SPONTANEOUS,
    Utterance,
    load_manifest,
    manifest_stats,
    speaker_disjoint_split,
    write_manifest,
)


def make_utts(n_speakers: int = 6, per_speaker: int = 5, seed_offset: int = 0) -> list[Utterance]:
    utts = []
    for s in range(n_speakers):
        label = LABEL_READ if s % 2 == 0 else LABEL_SPONTANEOUS
        for i in range(per_speaker):
            utts.append(
                Utterance(
                    utt_id=f"utt{seed_offset}_{s}_{i}",
                    path=f"/data/audio/utt{s}_{i}.wav",
                    label=label,
                    duration=2.5 + (i % 3),
                    speaker_id=f"spk{s}",
                    dataset="synthetic",
                    text=f"hello {s} {i}",
                )
            )
    return utts


class TestSchema:
    def test_invalid_label_rejected(self):
        with pytest.raises(ValueError, match="label"):
            Utterance("u", "/x.wav", label=7, duration=1.0, speaker_id="s", dataset="d")

    def test_invalid_split_rejected(self):
        with pytest.raises(ValueError, match="split"):
            Utterance(
                "u", "/x.wav", label=0, duration=1.0, speaker_id="s", dataset="d", split="oops"
            )

    def test_nonpositive_duration_rejected(self):
        with pytest.raises(ValueError, match="duration"):
            Utterance("u", "/x.wav", label=0, duration=0.0, speaker_id="s", dataset="d")

    def test_label_names(self):
        u = Utterance("u", "/x.wav", label=1, duration=1.0, speaker_id="s", dataset="d")
        assert u.label_name == "read"


class TestManifestIO:
    def test_roundtrip(self, tmp_path):
        utts = make_utts()
        path = write_manifest(tmp_path / "m.jsonl", utts)
        loaded = load_manifest(path)
        assert len(loaded) == len(utts)
        by_id = {u.utt_id: u for u in loaded}
        for u in utts:
            assert by_id[u.utt_id] == u

    def test_unknown_fields_preserved(self, tmp_path):
        u = Utterance("u", "/x.wav", label=0, duration=1.0, speaker_id="s", dataset="d")
        write_manifest(tmp_path / "m.jsonl", [u])
        lines = (tmp_path / "m.jsonl").read_text().strip().splitlines()
        rec = json.loads(lines[0])
        rec["new_field"] = 123
        (tmp_path / "m2.jsonl").write_text(json.dumps(rec) + "\n")
        loaded = load_manifest(tmp_path / "m2.jsonl")
        assert loaded[0].extra["new_field"] == 123

    def test_duplicate_id_rejected(self, tmp_path):
        u = Utterance("dup", "/x.wav", label=0, duration=1.0, speaker_id="s", dataset="d")
        with pytest.raises(ValueError, match="duplicate"):
            write_manifest(tmp_path / "m.jsonl", [u, u])

    def test_check_exists_relative(self, tmp_path):
        (tmp_path / "audio").mkdir()
        (tmp_path / "audio" / "a.wav").write_bytes(b"")
        u = Utterance("u", "audio/a.wav", label=0, duration=1.0, speaker_id="s", dataset="d")
        write_manifest(tmp_path / "m.jsonl", [u])
        assert len(load_manifest(tmp_path / "m.jsonl", check_exists=True)) == 1

        u2 = Utterance("u", "audio/missing.wav", label=0, duration=1.0, speaker_id="s", dataset="d")
        write_manifest(tmp_path / "m2.jsonl", [u2])
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "m2.jsonl", check_exists=True)

    def test_stats(self):
        utts = make_utts(n_speakers=4, per_speaker=4)
        speaker_disjoint_split(utts, seed=1)
        stats = manifest_stats(utts)
        assert stats["num_utterances"] == 16
        assert stats["num_speakers"] == 4
        assert stats["by_label"]["read"]["count"] == 8
        assert stats["by_label"]["spontaneous"]["count"] == 8
        total_secs = sum(u.duration for u in utts)
        assert stats["total_hours"] == pytest.approx(total_secs / 3600, rel=1e-3)


class TestSpeakerDisjointSplit:
    def test_no_speaker_leakage(self):
        utts = make_utts(n_speakers=12, per_speaker=4)
        speaker_disjoint_split(utts, seed=3)
        speakers_by_split: dict[str, set[str]] = {}
        for u in utts:
            speakers_by_split.setdefault(u.split, set()).add(u.speaker_id)
        all_spk = [s for spk in speakers_by_split.values() for s in spk]
        assert len(all_spk) == len(set(all_spk)), "a speaker appears in two splits"

    def test_all_splits_present(self):
        utts = make_utts(n_speakers=30, per_speaker=3)
        speaker_disjoint_split(utts, seed=5)
        counts = {"train": 0, "val": 0, "test": 0}
        for u in utts:
            counts[u.split] += 1
        assert counts["train"] > 0 and counts["val"] > 0 and counts["test"] > 0

    def test_ratios_approx(self):
        utts = make_utts(n_speakers=40, per_speaker=5)
        speaker_disjoint_split(utts, seed=11)
        n = len(utts)
        counts = {"train": 0, "val": 0, "test": 0}
        for u in utts:
            counts[u.split] += 1
        # greedy packing is lumpy (speaker granularity); allow generous tolerance
        assert 0.5 * n <= counts["train"] <= 0.85 * n
        assert 0.02 * n <= counts["val"] <= 0.35 * n
        assert counts["test"] >= 0.05 * n

    def test_deterministic(self):
        a = make_utts()
        b = make_utts()
        speaker_disjoint_split(a, seed=99)
        speaker_disjoint_split(b, seed=99)
        assert [u.split for u in a] == [u.split for u in b]

    def test_label_stratification(self):
        utts = make_utts(n_speakers=40, per_speaker=5)
        speaker_disjoint_split(utts, seed=13)
        for split in ("train", "val", "test"):
            labels = [u.label for u in utts if u.split == split]
            if labels:
                # each split keeps a mix of both labels (speakers alternate labels)
                assert len(set(labels)) == 2, f"split {split} lost a class"

    def test_invalid_ratios(self):
        utts = make_utts(n_speakers=2, per_speaker=1)
        with pytest.raises(ValueError):
            speaker_disjoint_split(utts, train_ratio=0.9, val_ratio=0.5)
