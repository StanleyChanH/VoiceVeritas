"""Fetch pretrained model artifacts from GitHub releases.

``vv download-model`` wraps this so users never copy URLs by hand::

    uv run vv download-model                 # -> cache path, reused by predict
    uv run vv download-model --out m.joblib  # explicit destination

Artifacts are listed on the releases page; ``download_file`` (stdlib-only)
handles retries and skips an already-cached file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from voiceveritas.data_prep.download import download_file

REPO = "StanleyChanH/VoiceVeritas"

# Latest release that actually carries a pretrained artifact. Keep in sync
# with the releases page (later releases may attach new ones).
DEFAULT_VERSION = "v0.1.1"

# model key -> release-asset name template ({tag} = full release tag, e.g. "v0.1.1")
_MODELS: dict[str, str] = {
    "prosody": "voiceveritas-prosody-lgbm-{tag}.joblib",
}


def available_models() -> list[str]:
    """Model keys that can be downloaded (see the releases page for details)."""
    return sorted(_MODELS)


def asset_url(model: str = "prosody", version: str = DEFAULT_VERSION) -> str:
    """Stable download URL for a release asset."""
    try:
        template = _MODELS[model]
    except KeyError:
        raise ValueError(
            f"unknown model {model!r}; available: {', '.join(available_models())}"
        ) from None
    tag = version if version.startswith("v") else f"v{version}"
    return f"https://github.com/{REPO}/releases/download/{tag}/{template.format(tag=tag)}"


def cache_dir() -> Path:
    """Platform-aware cache directory (override with ``VOICEVERITAS_CACHE_DIR``)."""
    override = os.environ.get("VOICEVERITAS_CACHE_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "VoiceVeritas" / "models"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "voiceveritas" / "models"


def download_model(
    model: str = "prosody",
    version: str = DEFAULT_VERSION,
    out: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download (or reuse a cached) pretrained artifact; returns its path."""
    url = asset_url(model, version)
    dest = Path(out) if out else cache_dir() / url.rsplit("/", 1)[1]
    if force and dest.exists():
        dest.unlink()
    return download_file(url, dest)
