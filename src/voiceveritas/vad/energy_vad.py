"""Energy-based voice activity detection with an adaptive threshold.

Why hand-rolled instead of a neural VAD? The pause/silence statistics that
discriminate read from spontaneous speech *are* features — we want a
transparent, dependency-free, deterministic VAD whose thresholds can be
audited and tested on synthetic signals (see tests/test_vad.py).

Algorithm
---------
1. Frame the signal (25 ms window, 10 ms hop), compute energy in dB.
2. Estimate the noise floor (20th percentile) and speech peak (85th
   percentile) of the energy distribution.
3. If the peak is below ``min_speech_db`` the file is treated as silence.
4. Threshold = max(floor + 30% of (peak - floor), peak - 45 dB).
5. Apply hangover rules: drop speech bursts < ``min_speech_s`` and merge
   silences shorter than ``min_pause_s``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-10


@dataclass(frozen=True)
class VadConfig:
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    floor_pct: float = 20.0
    peak_pct: float = 85.0
    floor_frac: float = 0.3
    max_dyn_range_db: float = 45.0
    min_speech_db: float = -45.0
    min_speech_s: float = 0.05
    min_pause_s: float = 0.12


@dataclass
class VadResult:
    """VAD output with frame-level mask and second-level segments."""

    speech_mask: np.ndarray  # bool per hop frame
    hop_s: float
    segments: list[tuple[float, float]] = field(default_factory=list)

    @property
    def speech_ratio(self) -> float:
        """Fraction of frames detected as speech."""
        if self.speech_mask.size == 0:
            return 0.0
        return float(self.speech_mask.mean())

    @property
    def speech_seconds(self) -> float:
        return float(self.speech_mask.sum() * self.hop_s)

    @property
    def total_seconds(self) -> float:
        return float(self.speech_mask.size * self.hop_s)

    @property
    def speech_durations(self) -> np.ndarray:
        """Durations (s) of the contiguous speech segments."""
        return np.array([e - s for s, e in self.segments], dtype=np.float64)

    @property
    def pause_durations(self) -> np.ndarray:
        """Durations (s) of the gaps *between* speech segments."""
        durs = []
        for (_, e1), (s2, _) in zip(self.segments[:-1], self.segments[1:], strict=False):
            durs.append(s2 - e1)
        return np.array(durs, dtype=np.float64)


def frame_energies_db(x: np.ndarray, sr: int, cfg: VadConfig | None = None) -> np.ndarray:
    """Frame energies in dB using the VAD framing (window-aligned, hop-strided)."""
    cfg = cfg or VadConfig()
    win = max(1, int(sr * cfg.frame_ms / 1000))
    hop = max(1, int(sr * cfg.hop_ms / 1000))
    if len(x) < win:
        n = 1
        idx = np.zeros((1, len(x)), dtype=int)
    else:
        n = 1 + (len(x) - win) // hop
        idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    frames = x[idx]
    energy = (frames.astype(np.float64) ** 2).mean(axis=1)
    return 10.0 * np.log10(energy + _EPS)


def _mask_to_segments(mask: np.ndarray, hop_s: float) -> list[tuple[float, float]]:
    if mask.size == 0:
        return []
    segs: list[list[float]] = []
    in_seg = False
    for i, v in enumerate(mask):
        if v and not in_seg:
            in_seg = True
            segs.append([i * hop_s, (i + 1) * hop_s])
        elif v and in_seg:
            segs[-1][1] = (i + 1) * hop_s
        elif not v:
            in_seg = False
    return [(s, e) for s, e in segs]


def energy_vad(samples: np.ndarray, sr: int, config: VadConfig | None = None) -> VadResult:
    """Run energy VAD on a mono waveform."""
    cfg = config or VadConfig()
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"expected mono 1-D samples, got shape {x.shape}")

    hop_s = cfg.hop_ms / 1000.0
    if x.size == 0:
        return VadResult(speech_mask=np.zeros(0, dtype=bool), hop_s=hop_s, segments=[])

    db = frame_energies_db(x, sr, cfg)

    peak_db = float(np.percentile(db, cfg.peak_pct))
    if peak_db < cfg.min_speech_db:
        mask = np.zeros(db.shape, dtype=bool)
        return VadResult(speech_mask=mask, hop_s=hop_s, segments=[])

    floor_db = float(np.percentile(db, cfg.floor_pct))
    spread = max(peak_db - floor_db, 0.0)
    thr = max(floor_db + cfg.floor_frac * spread, peak_db - cfg.max_dyn_range_db)
    mask = db >= thr

    # hangover: merge silences < min_pause, then drop speech < min_speech
    segments = _mask_to_segments(mask, hop_s)
    merged: list[list[float]] = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] < cfg.min_pause_s:
            merged[-1][1] = seg[1]
        else:
            merged.append([seg[0], seg[1]])
    segments = [(s, e) for s, e in merged if e - s >= cfg.min_speech_s]

    # rebuild the mask from surviving segments
    mask = np.zeros(db.shape, dtype=bool)
    n_frames = db.shape[0]
    for s, e in segments:
        i0 = min(round(s / hop_s), n_frames)
        i1 = min(round(e / hop_s), n_frames)
        mask[i0:i1] = True

    return VadResult(speech_mask=mask, hop_s=hop_s, segments=segments)
