"""Robust file downloads (stdlib-only, resumable-ish, with progress)."""

from __future__ import annotations

import logging
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_UA = "VoiceVeritas/0.1 (+https://github.com/StanleyChanH/VoiceVeritas)"


def download_file(
    url: str,
    dest: str | Path,
    retries: int = 3,
    timeout: int = 120,
    chunk: int = 1 << 20,
) -> Path:
    """Download ``url`` to ``dest`` with retries and a simple progress line.

    Skips the download when the file already exists (delete to re-fetch).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("already downloaded: %s", dest)
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as fh:
                total = int(resp.headers.get("Content-Length", 0) or 0)
                done = 0
                start = time.time()
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    fh.write(block)
                    done += len(block)
                    _progress(dest.name, done, total, start)
            sys.stdout.write("\n")
            sys.stdout.flush()
            shutil.move(str(tmp), str(dest))
            return dest
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            logger.warning("download attempt %d/%d failed: %s", attempt, retries, exc)
            time.sleep(2 * attempt)
    raise RuntimeError(f"failed to download {url}: {last_error}")


def _progress(name: str, done: int, total: int, start: float) -> None:
    if not sys.stdout.isatty():
        return
    rate = done / max(time.time() - start, 1e-6) / (1 << 20)
    if total:
        pct = 100 * done / total
        bar = "#" * int(pct // 4)
        sys.stdout.write(f"\r{name}: [{bar:<25s}] {pct:5.1f}%  {rate:5.1f} MiB/s")
    else:
        sys.stdout.write(f"\r{name}: {done / (1 << 20):7.1f} MiB  {rate:5.1f} MiB/s")
    sys.stdout.flush()
