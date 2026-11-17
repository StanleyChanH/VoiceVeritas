"""Tests for `vv eval`: split filtering and per-utterance prediction dumps."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from voiceveritas.cli import main
from voiceveritas.manifest import Utterance, write_manifest
from voiceveritas.utils.synth import SR, synthesize_speechlike


def _make_manifest(path: Path, entries: list[tuple[str, str, int]]) -> Path:
    """Write a manifest with one synthetic wav per (utt_id, split, label)."""
    utts = []
    for i, (utt_id, split, label) in enumerate(entries):
        style = "read" if label else "spontaneous"
        rng = np.random.default_rng(100 + i)
        wav = path.parent / f"{utt_id}.wav"
        sf.write(str(wav), synthesize_speechlike(rng, 4.0, style=style), SR, subtype="PCM_16")
        utts.append(
            Utterance(utt_id, str(wav), label, 4.0, f"spk_{utt_id[-1]}", "synthetic", split)
        )
    write_manifest(path, utts)
    return path


def test_eval_dump_predictions_csv(tmp_path: Path, trained_model_path: Path, capsys) -> None:
    manifest = _make_manifest(
        tmp_path / "m.jsonl",
        [
            ("read_a", "test", 1),
            ("read_b", "train", 1),  # must be filtered out by --split test
            ("spon_c", "test", 0),
        ],
    )
    csv_path = tmp_path / "preds.csv"

    rc = main(
        [
            "eval",
            "--run",
            str(trained_model_path),
            "--manifest",
            str(manifest),
            "--split",
            "test",
            "--dump-predictions",
            str(csv_path),
        ]
    )
    assert rc == 0

    # stdout is the metric report JSON for the filtered split
    report = json.loads(capsys.readouterr().out)
    assert report["n"] == 2
    assert report["n_read"] == 1 and report["n_spontaneous"] == 1

    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["utt_id"] for r in rows] == ["read_a", "spon_c"]
    assert "model" in rows[0]
    for r in rows:
        p = float(r["read_probability"])
        assert 0.0 <= p <= 1.0
        assert r["predicted_label"] in {"0", "1"}
        assert int(r["predicted_label"]) == int(p >= 0.5)
    # the train-split utterance never leaked into the artifact
    assert all(r["utt_id"] != "read_b" for r in rows)


def test_eval_without_dump_unchanged(tmp_path: Path, trained_model_path: Path, capsys) -> None:
    manifest = _make_manifest(tmp_path / "m.jsonl", [("x1", "test", 0), ("x2", "test", 1)])
    rc = main(["eval", "--run", str(trained_model_path), "--manifest", str(manifest)])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["n"] == 2
    assert not list(tmp_path.glob("*.csv"))
