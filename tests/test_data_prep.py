"""Tests for dataset preparation (synthetic files, no network)."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.conftest import SR, synthesize_speechlike
from voiceveritas.data_prep.ami import ami_session_url, chunk_session
from voiceveritas.data_prep.chunking import build_answer_units, group_flac_files_by_chapter
from voiceveritas.manifest import load_manifest


@pytest.fixture()
def fake_librispeech_tree(tmp_path: Path) -> Path:
    root = tmp_path / "LibriSpeech" / "dev-clean"
    for spk in ("spkA", "spkB"):
        for chapter in ("0001", "0002"):
            d = root / spk / chapter
            d.mkdir(parents=True)
            for utt in range(1, 5):
                t = np.arange(int(3.0 * SR)) / SR
                x = (0.3 * np.sin(2 * np.pi * 180 * t + utt)).astype(np.float32)
                sf.write(str(d / f"{spk}-{chapter}-{utt:04d}.flac"), x, SR)
    return root


class TestLibriSpeechPrep:
    def test_grouping_keeps_chapter_order(self, fake_librispeech_tree: Path):
        flacs = sorted(fake_librispeech_tree.rglob("*.flac"))
        groups = group_flac_files_by_chapter(flacs)
        assert set(groups) == {
            ("spkA", "0001"),
            ("spkA", "0002"),
            ("spkB", "0001"),
            ("spkB", "0002"),
        }
        order = [p.name for p in groups[("spkA", "0001")]]
        assert order == sorted(order, key=lambda n: int(n.rsplit("-", 1)[1].split(".")[0]))

    def test_build_units_respect_length(self, fake_librispeech_tree: Path, tmp_path: Path):
        flacs = sorted(fake_librispeech_tree.rglob("*.flac"))
        groups = group_flac_files_by_chapter(flacs)
        all_files = [f for files in groups.values() for f in files]
        units = build_answer_units(all_files, tmp_path / "out", "spkA")
        assert len(units) >= 1
        for path, dur in units:
            assert 5.0 <= dur <= 26.0
            assert path.exists()

    def test_prepare_librispeech_manifest(self, fake_librispeech_tree: Path, tmp_path: Path):
        from voiceveritas.data_prep.librispeech import prepare_librispeech

        manifest = tmp_path / "ls.jsonl"
        utts = prepare_librispeech(
            fake_librispeech_tree.parent,  # rglob finds the tree anyway
            out_dir=tmp_path / "units",
            manifest_path=manifest,
            max_speakers=2,
            max_units_per_speaker=4,
        )
        assert utts, "no units produced"
        assert all(u.label == 1 for u in utts)
        assert all(u.dataset == "librispeech" for u in utts)
        reloaded = load_manifest(manifest)
        assert {u.utt_id for u in reloaded} == {u.utt_id for u in utts}


class TestAmiChunking:
    def test_chunks_cover_speech(self, rng):
        x = synthesize_speechlike(rng, 60.0, style="spontaneous")
        chunks = chunk_session(x, SR)
        assert len(chunks) >= 2
        for s, e in chunks:
            # chunks close at natural boundaries past 25s, hard-capped at 37.5s
            assert 7.5 <= e - s <= 37.5 + 1e-6
        lengths = {round(e - s, 1) for s, e in chunks}
        assert len(lengths) > 1, "all chunks hit the cap — a dataset fingerprint"
        # chunks sorted and non-overlapping
        for (_, e1), (s2, _) in itertools.pairwise(chunks):
            assert e1 <= s2

    def test_empty_audio_no_chunks(self):
        chunks = chunk_session(np.zeros(int(10 * SR), dtype=np.float32), SR)
        assert chunks == []

    def test_session_url_format(self):
        url = ami_session_url("ES2002a")
        assert url == (
            "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/"
            "ES2002a/audio/ES2002a.Mix-Headset.wav"
        )


class TestSbcsaePrep:
    @pytest.fixture()
    def fake_sbcsae_tree(self, tmp_path: Path) -> Path:
        """Two fake conversations with VAD-detectable speechlike audio."""
        root = tmp_path / "SBCSAE" / "wav"
        root.mkdir(parents=True)
        for conv, seed in (("SBC001", 11), ("SBC009", 12)):
            rng_local = np.random.default_rng(seed)
            x = synthesize_speechlike(rng_local, 40.0, style="spontaneous")
            sf.write(str(root / f"{conv}.wav"), x, SR, subtype="PCM_16")
        return root.parent.parent  # rglob reaches the wavs from the top

    def test_prepare_sbcsae_manifest(self, fake_sbcsae_tree: Path, tmp_path: Path):
        from voiceveritas.data_prep.sbcsae import prepare_sbcsae

        manifest = tmp_path / "sbcsae.jsonl"
        utts = prepare_sbcsae(
            fake_sbcsae_tree,
            out_dir=tmp_path / "chunks",
            manifest_path=manifest,
            max_chunks_per_conversation=3,
        )
        assert utts, "no chunks produced"
        assert all(u.label == 0 for u in utts)
        assert all(u.dataset == "sbcsae" for u in utts)
        # eval-only corpus: every entry is the test split
        assert all(u.split == "test" for u in utts)
        assert {u.speaker_id for u in utts} == {"sbcsae_sbc001", "sbcsae_sbc009"}
        reloaded = load_manifest(manifest)
        assert {u.utt_id for u in reloaded} == {u.utt_id for u in utts}

    def test_conversation_subset(self, fake_sbcsae_tree: Path, tmp_path: Path):
        from voiceveritas.data_prep.sbcsae import prepare_sbcsae

        utts = prepare_sbcsae(
            fake_sbcsae_tree,
            out_dir=tmp_path / "chunks",
            manifest_path=tmp_path / "m.jsonl",
            conversations=["sbc001"],
            max_chunks_per_conversation=3,
        )
        assert utts and {u.speaker_id for u in utts} == {"sbcsae_sbc001"}

    def test_conversation_id_normalizes(self):
        from voiceveritas.data_prep.sbcsae import conversation_id

        assert conversation_id(Path("SBC001.wav")) == "sbc001"
        assert conversation_id(Path("SBC009.wav")) == "sbc009"
        assert conversation_id(Path("sbc060.mp3")) == "sbc060"

    def test_missing_audio_raises(self, tmp_path: Path):
        from voiceveritas.data_prep.sbcsae import prepare_sbcsae

        with pytest.raises(FileNotFoundError, match="extract"):
            prepare_sbcsae(tmp_path, out_dir=tmp_path / "o", manifest_path=tmp_path / "m.jsonl")
