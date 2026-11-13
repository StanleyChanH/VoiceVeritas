"""Prosodic and acoustic feature extraction (interpretable Track A).

Literature-grounded cues for read vs spontaneous speech:

- **Pause structure** (the single most salient family): duration distribution
  of silences and speech runs — read speech has metronomic, short pauses;
  spontaneous speech has heavy-tailed pauses (planning, hesitation).
- **F0 dynamics**: read speech has compressed range and smooth declination;
  spontaneous speech wanders.
- **Period perturbation**: jitter/shimmer, HNR — recitation tends to a
  steadier phonation.
- **Rhythm regularity**: nPVI of speech-run durations; syllable-nucleus rate
  variability.
- **Spectral shape**: centroid/flatness statistics as cheap channel/prosody
  correlates.

All features are NaN-safe (LightGBM consumes np.nan natively); names and
order are fixed by :data:`FEATURE_NAMES` so serialized models stay valid
across versions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import parselmouth
from parselmouth import praat

from voiceveritas.vad import VadResult, energy_vad

_EPS = 1e-10
_PITCH_FLOOR = 60.0
_PITCH_CEIL = 350.0


def _semiTones(f_lo: float, f_hi: float) -> float:
    if f_lo <= 0 or f_hi <= 0:
        return np.nan
    return 12.0 * np.log2(f_hi / f_lo)


def _safe(fn):  # type: ignore[no-untyped-def]
    """Wrap a praat call so failures degrade to NaN instead of crashing."""
    try:
        value = fn()
        return float(value) if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def _npvi(durs: np.ndarray) -> float:
    """Normalized Pairwise Variability Index of consecutive interval durations."""
    if durs.size < 2:
        return np.nan
    pairs = durs[:-1] + durs[1:]
    ok = pairs > 0
    if not ok.any():
        return np.nan
    diffs = np.abs(np.diff(durs))[ok]
    return float(100.0 * np.mean(diffs / (pairs[ok] / 2.0)))


def _stats_block(prefix: str, values: np.ndarray) -> dict[str, float]:
    """mean/std/p90 summary for a duration array; NaN when empty."""
    if values.size == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan, f"{prefix}_p90": np.nan}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def _f0_features(snd: parselmouth.Sound) -> tuple[dict[str, float], np.ndarray]:
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=_PITCH_FLOOR, pitch_ceiling=_PITCH_CEIL)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]

    feats: dict[str, float] = {"f0_voiced_ratio": float((f0 > 0).mean()) if f0.size else np.nan}
    if voiced.size < 2:
        nan_block = dict.fromkeys(
            (
                "f0_mean",
                "f0_median",
                "f0_std",
                "f0_p10",
                "f0_p90",
                "f0_range_semitones",
                "f0_slope_st_per_s",
                "f0_jitter_frame",
            ),
            np.nan,
        )
        feats.update(nan_block)
        return feats, voiced

    t_voiced = np.flatnonzero(f0 > 0) * 0.01
    slope_st_per_s = np.nan
    if t_voiced.size >= 2 and t_voiced[-1] > t_voiced[0]:
        slope = np.polyfit(t_voiced, 12.0 * np.log2(voiced / voiced.mean()), 1)[0]
        slope_st_per_s = float(slope)

    f0_semi = 12.0 * np.log2(voiced / 440.0)  # stable domain for diffs
    feats.update(
        {
            "f0_mean": float(np.mean(voiced)),
            "f0_median": float(np.median(voiced)),
            "f0_std": float(np.std(voiced)),
            "f0_p10": float(np.percentile(voiced, 10)),
            "f0_p90": float(np.percentile(voiced, 90)),
            "f0_range_semitones": _semiTones(
                float(np.percentile(voiced, 10)), float(np.percentile(voiced, 90))
            ),
            "f0_slope_st_per_s": slope_st_per_s,
            "f0_jitter_frame": float(np.mean(np.abs(np.diff(f0_semi)))),
        }
    )
    return feats, voiced


def _phonation_features(snd: parselmouth.Sound) -> dict[str, float]:
    def _jitter() -> float:
        pp = praat.call(snd, "To PointProcess (periodic, cc)", _PITCH_FLOOR, _PITCH_CEIL)
        return praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

    def _shimmer() -> float:
        pp = praat.call(snd, "To PointProcess (periodic, cc)", _PITCH_FLOOR, _PITCH_CEIL)
        return praat.call((snd, pp), "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    def _hnr() -> float:
        harmonicity = snd.to_harmonicity()
        vals = harmonicity.values[harmonicity.values > -100]
        return float(np.mean(vals)) if vals.size else np.nan

    return {
        "jitter_local": _safe(_jitter),
        "shimmer_local": _safe(_shimmer),
        "hnr_db": _safe(_hnr),
    }


def _syllable_rate(x: np.ndarray, sr: int, vad: VadResult) -> tuple[float, float]:
    """Envelope peak counting → syllable-nucleus rate (per speech second).

    Returns (rate_mean, rate_std_over_segments); NaN when no speech.
    """
    from scipy.signal import butter, find_peaks, sosfilt

    sos = butter(4, 8.0, btype="low", fs=sr, output="sos")
    env = sosfilt(sos, np.abs(x))
    # downsample envelope to 100 Hz grid
    hop = sr // 100
    n = env.size // hop
    if n == 0:
        return np.nan, np.nan
    env_ds = env[: n * hop].reshape(n, hop).max(axis=1)
    grid = 100.0

    per_seg_rates: list[float] = []
    total_peaks = 0
    total_speech = 0.0
    for s, e in vad.segments:
        i0, i1 = int(s * grid), min(int(e * grid), n)
        if i1 - i0 < int(0.15 * grid):
            continue
        seg_env = env_ds[i0:i1]
        thr = 0.3 * float(np.percentile(seg_env, 95))
        peaks, _ = find_peaks(seg_env, height=max(thr, _EPS), distance=int(0.12 * grid))
        dur = (i1 - i0) / grid
        per_seg_rates.append(len(peaks) / dur)
        total_peaks += len(peaks)
        total_speech += dur

    if total_speech <= 0:
        return np.nan, np.nan
    return total_peaks / total_speech, float(np.std(per_seg_rates)) if per_seg_rates else np.nan


def _spectral_features(x: np.ndarray, sr: int) -> dict[str, float]:
    import librosa

    centroid = librosa.feature.spectral_centroid(y=x, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=x)[0]
    zcr = librosa.feature.zero_crossing_rate(x)[0]
    return {
        "spec_centroid_mean": float(np.mean(centroid)),
        "spec_centroid_std": float(np.std(centroid)),
        "spec_flatness_mean": float(np.mean(flatness)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
    }


def _energy_features(x: np.ndarray, sr: int, vad: VadResult) -> dict[str, float]:
    from voiceveritas.vad.energy_vad import frame_energies_db

    db = frame_energies_db(x, sr)
    mask = vad.speech_mask[: db.size]
    speech_db = db[mask] if mask.any() else db
    return {
        "energy_db_std": float(np.std(speech_db)),
        "energy_db_range": float(np.percentile(speech_db, 95) - np.percentile(speech_db, 5)),
    }


FEATURE_NAMES: tuple[str, ...] = (
    # NOTE: no duration feature, deliberately. Utterance length is a
    # dataset/collection artifact (a shortcut that leaks source identity,
    # see docs/benchmark.md) — not a prosodic cue a fair detector should use.
    # F0
    "f0_voiced_ratio",
    "f0_mean",
    "f0_median",
    "f0_std",
    "f0_p10",
    "f0_p90",
    "f0_range_semitones",
    "f0_slope_st_per_s",
    "f0_jitter_frame",
    # phonation
    "jitter_local",
    "shimmer_local",
    "hnr_db",
    # energy
    "energy_db_std",
    "energy_db_range",
    # pause / speech-run structure
    "speech_ratio",
    "speech_seg_rate",
    "pause_total_ratio",
    "long_pause_ratio",
    "speechseg_mean",
    "speechseg_std",
    "speechseg_p90",
    "pause_mean",
    "pause_std",
    "pause_p90",
    "pause_cv",
    # rhythm
    "npvi_speech",
    "npvi_pause",
    "syllable_rate",
    "syllable_rate_std",
    # spectral
    "spec_centroid_mean",
    "spec_centroid_std",
    "spec_flatness_mean",
    "zcr_mean",
    "zcr_std",
)


@dataclass
class FeatureResult:
    """Fixed-order prosodic feature vector with names and NaN mask."""

    values: np.ndarray  # shape (len(FEATURE_NAMES),), float64, may contain NaN
    names: tuple[str, ...] = FEATURE_NAMES

    @property
    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.values)

    def to_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values.tolist(), strict=True))


def extract_prosody(
    samples: np.ndarray,
    sr: int,
    vad: VadResult | None = None,
) -> FeatureResult:
    """Extract the Track-A prosodic feature vector from a mono waveform."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"expected non-empty mono samples, got shape {x.shape}")
    if vad is None:
        vad = energy_vad(x, sr)

    snd = parselmouth.Sound(x, sr)

    feats: dict[str, float] = {}

    f0_feats, _ = _f0_features(snd)
    feats.update(f0_feats)
    feats.update(_phonation_features(snd))
    feats.update(_energy_features(x, sr, vad))

    speech_durs = vad.speech_durations
    pause_durs = vad.pause_durations
    total_s = vad.total_seconds
    feats.update(
        {
            "speech_ratio": vad.speech_ratio,
            "speech_seg_rate": len(vad.segments) / total_s if total_s > 0 else np.nan,
            "pause_total_ratio": (
                float(pause_durs.sum() / total_s) if pause_durs.size and total_s > 0 else np.nan
            ),
            "long_pause_ratio": (
                float(pause_durs[pause_durs > 0.6].sum() / pause_durs.sum())
                if pause_durs.size and pause_durs.sum() > 0
                else np.nan
            ),
            "pause_cv": (
                float(pause_durs.std() / pause_durs.mean())
                if pause_durs.size > 1 and pause_durs.mean() > 0
                else np.nan
            ),
            "npvi_speech": _npvi(speech_durs),
            "npvi_pause": _npvi(pause_durs),
        }
    )
    feats.update(_stats_block("speechseg", speech_durs))
    feats.update(_stats_block("pause", pause_durs))

    rate, rate_std = _syllable_rate(x, sr, vad)
    feats["syllable_rate"] = rate
    feats["syllable_rate_std"] = rate_std

    feats.update(_spectral_features(x, sr))

    vector = np.array([feats.get(name, np.nan) for name in FEATURE_NAMES], dtype=np.float64)
    return FeatureResult(values=vector)
