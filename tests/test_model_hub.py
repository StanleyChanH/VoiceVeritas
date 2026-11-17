"""Tests for ``vv download-model`` / model_hub (offline — downloads are faked)."""

from __future__ import annotations

from pathlib import Path

import pytest

from voiceveritas import model_hub
from voiceveritas.cli import main


def test_asset_url_format():
    # NOTE: the asset name embeds the full tag *with* the leading "v" —
    # verified against the real asset names on the releases page.
    assert model_hub.asset_url("prosody", "v0.1.1") == (
        "https://github.com/StanleyChanH/VoiceVeritas/releases/download/"
        "v0.1.1/voiceveritas-prosody-lgbm-v0.1.1.joblib"
    )


def test_asset_url_accepts_bare_version():
    assert model_hub.asset_url("prosody", "0.1.1") == model_hub.asset_url("prosody", "v0.1.1")


def test_asset_url_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        model_hub.asset_url("bogus")


def test_cache_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEVERITAS_CACHE_DIR", str(tmp_path))
    assert model_hub.cache_dir() == tmp_path


def test_download_model_uses_cache_and_url(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEVERITAS_CACHE_DIR", str(tmp_path))
    seen: dict[str, str] = {}

    def fake_download(url, dest, **kwargs):
        seen["url"] = url
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"joblib")
        return dest

    monkeypatch.setattr(model_hub, "download_file", fake_download)
    path = model_hub.download_model("prosody", "v0.1.1")
    assert seen["url"].endswith("v0.1.1/voiceveritas-prosody-lgbm-v0.1.1.joblib")
    assert path == tmp_path / "voiceveritas-prosody-lgbm-v0.1.1.joblib"
    assert path.read_bytes() == b"joblib"


def test_download_model_force_refetches(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEVERITAS_CACHE_DIR", str(tmp_path))
    calls: list[str] = []

    def fake_download(url, dest, **kwargs):
        calls.append(url)
        dest = Path(dest)
        dest.write_bytes(b"joblib")
        return dest

    monkeypatch.setattr(model_hub, "download_file", fake_download)
    model_hub.download_model("prosody", "v0.1.1")
    model_hub.download_model("prosody", "v0.1.1", force=True)
    assert len(calls) == 2


def test_cli_download_model_invokes_hub(tmp_path, monkeypatch, capsys):
    out = tmp_path / "m.joblib"

    def fake_download_model(**kwargs):
        out.write_bytes(b"x")
        return out

    monkeypatch.setattr(model_hub, "download_model", fake_download_model)
    rc = main(["download-model", "--out", str(out)])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == str(out)  # stdout = bare path, for $( ) capture
    assert "try it with" in captured.err
