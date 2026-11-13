"""Acoustic perturbations and robustness evaluation.

Interview audio arrives in hostile condition: cheap laptop mics, compressed
video-call audio, background noise. Before trusting a score you need to
know *how fast quality degrades the detector*. This module implements the
perturbation family from the project plan (white/pink noise at controlled
SNR, telephony band-limiting, sample-rate round trips, MP3 bitrate
starvation) plus a harness that measures metric drift per condition.

All perturbations are deterministic given a seed — results are reportable.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from voiceveritas.audio import load_audio
from voiceveritas.manifest import load_manifest
from voiceveritas.metrics import compute_binary_metrics
from voiceveritas.models.base import ReadSpeechModel

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# perturbation primitives
# ----------------------------------------------------------------------
def _realized_snr_db(x: np.ndarray, y: np.ndarray) -> float:
    """SNR of signal x against contamination y, in dB."""
    p_sig = float(np.mean(x.astype(np.float64) ** 2))
    p_noi = float(np.mean(y.astype(np.float64) ** 2))
    if p_noi <= 0:
        return float("inf")
    return 10.0 * np.log10(p_sig / p_noi)


def add_white_noise(x: np.ndarray, sr: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise at the target SNR (whole-signal power)."""
    p_sig = float(np.mean(x.astype(np.float64) ** 2))
    p_noi = p_sig / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(p_noi), size=x.shape).astype(np.float32)
    return (x + noise).astype(np.float32)


def add_pink_noise(x: np.ndarray, sr: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """1/f-shaped noise (fans, HVAC — the realistic interview background)."""
    n = x.shape[0]
    white = rng.normal(0.0, 1.0, size=n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    shape = 1.0 / np.sqrt(np.maximum(freqs, 1.0))  # ~1/sqrt(f) power shaping
    pink = np.fft.irfft(spectrum * shape, n=n)
    pink = pink / (np.sqrt(np.mean(pink**2)) + 1e-12)
    p_sig = float(np.mean(x.astype(np.float64) ** 2))
    p_noi = p_sig / (10.0 ** (snr_db / 10.0))
    return (x + pink * np.sqrt(p_noi)).astype(np.float32)


def apply_gain(x: np.ndarray, sr: int, db: float) -> np.ndarray:
    """Level shift (mic gain mistakes); clipping reflects ADC behaviour."""
    y = x * (10.0 ** (db / 20.0))
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def telephony_band(
    x: np.ndarray, sr: int, low_hz: float = 300.0, high_hz: float = 3400.0
) -> np.ndarray:
    """Band-pass to the classic telephone band (worst-case VoIP codec)."""
    from scipy.signal import butter, sosfilt

    sos = butter(6, [low_hz, high_hz], btype="band", fs=sr, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def resample_loop(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Down-sample and back (narrow-band 8 kHz VoIP round trip)."""
    import librosa

    y = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    y = librosa.resample(y, orig_sr=target_sr, target_sr=sr)
    # resample may change length slightly; align to original
    if len(y) >= len(x):
        return y[: len(x)].astype(np.float32)
    return np.pad(y, (0, len(x) - len(y))).astype(np.float32)


def mp3_transcode(x: np.ndarray, sr: int, bitrate_kbps: int) -> np.ndarray | None:
    """Encode to MP3 and decode back (lossy codec stress).

    Returns None (and logs) when ffmpeg is unavailable — the caller treats
    the condition as skipped rather than failed.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found — skipping mp3 transcode condition")
        return None
    with tempfile.TemporaryDirectory() as tmp:
        import soundfile as sf

        wav_in = Path(tmp) / "in.wav"
        mp3 = Path(tmp) / "out.mp3"
        wav_out = Path(tmp) / "out.wav"
        sf.write(str(wav_in), x, sr, subtype="PCM_16")
        enc = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(wav_in),
                "-b:a",
                f"{bitrate_kbps}k",
                str(mp3),
            ],
            capture_output=True,
        )
        dec = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(mp3), str(wav_out)],
            capture_output=True,
        )
        if enc.returncode != 0 or dec.returncode != 0 or not wav_out.exists():
            logger.warning(
                "mp3 transcode failed: %s", (enc.stderr or dec.stderr).decode(errors="replace")
            )
            return None
        return load_audio(wav_out, sample_rate=sr).samples


# ----------------------------------------------------------------------
# condition registry + evaluation harness
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Perturbation:
    """One named stress condition."""

    name: str
    description: str
    fn_name: str  # perturbation function key
    kwargs: dict = field(default_factory=dict)


def default_conditions(snr_grid_db: tuple[float, ...] = (20.0, 10.0, 5.0)) -> list[Perturbation]:
    """The standard stress matrix from the project plan."""
    conds = [Perturbation("clean", "unmodified audio", "identity", {})]
    for snr in snr_grid_db:
        conds.append(
            Perturbation(
                f"white_noise_{snr:g}db", "additive Gaussian noise", "white", {"snr_db": snr}
            )
        )
        conds.append(
            Perturbation(f"pink_noise_{snr:g}db", "1/f background noise", "pink", {"snr_db": snr})
        )
    conds.append(Perturbation("telephony_300_3400", "telephone band-pass", "telephony", {}))
    conds.append(Perturbation("resample_8k", "8 kHz round trip", "resample", {"target_sr": 8000}))
    conds.append(Perturbation("mp3_32k", "32 kbps MP3 transcode", "mp3", {"bitrate_kbps": 32}))
    conds.append(Perturbation("mp3_16k", "16 kbps MP3 transcode", "mp3", {"bitrate_kbps": 16}))
    return conds


def _perturb(
    x: np.ndarray, sr: int, cond: Perturbation, rng: np.random.Generator
) -> np.ndarray | None:
    if cond.fn_name == "identity":
        return x
    if cond.fn_name == "white":
        return add_white_noise(x, sr, rng=rng, **cond.kwargs)
    if cond.fn_name == "pink":
        return add_pink_noise(x, sr, rng=rng, **cond.kwargs)
    if cond.fn_name == "gain":
        return apply_gain(x, sr, **cond.kwargs)
    if cond.fn_name == "telephony":
        return telephony_band(x, sr, **cond.kwargs)
    if cond.fn_name == "resample":
        return resample_loop(x, sr, **cond.kwargs)
    if cond.fn_name == "mp3":
        return mp3_transcode(x, sr, **cond.kwargs)
    raise ValueError(f"unknown perturbation fn: {cond.fn_name}")


def run_robustness_eval(
    model: ReadSpeechModel,
    manifest_path: str | Path,
    conditions: list[Perturbation] | None = None,
    seed: int = 42,
    split: str = "test",
) -> list[dict]:
    """Score the model on the manifest under every condition; report drift.

    Returns one row per condition: metrics plus ``delta_auc_roc`` vs clean.
    """
    conditions = conditions or default_conditions()
    utts = [u for u in load_manifest(manifest_path) if u.split == split]
    if not utts:  # manifests without splits: use everything
        utts = load_manifest(manifest_path)
    if not utts:
        raise ValueError(f"no utterances in {manifest_path}")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    clean_auc: float | None = None

    for cond in conditions:
        y_true: list[int] = []
        scores: list[float] = []
        skipped = 0
        for utt in utts:
            try:
                loaded = load_audio(utt.path)
                perturbed = _perturb(loaded.samples, loaded.sample_rate, cond, rng)
            except Exception as exc:
                logger.warning("%s: skipping %s (%s)", cond.name, utt.utt_id, exc)
                skipped += 1
                continue
            if perturbed is None:
                skipped = len(utts)
                break
            result = model.predict(perturbed, loaded.sample_rate)
            y_true.append(utt.label)
            scores.append(result.read_probability)
        if len(set(y_true)) < 2:
            rows.append(
                {
                    "condition": cond.name,
                    "status": "skipped",
                    "reason": "single class or unavailable",
                }
            )
            continue
        metrics = compute_binary_metrics(y_true, scores, label=cond.name)
        if cond.name == "clean":
            clean_auc = metrics["auc_roc"]
        rows.append(
            {
                "condition": cond.name,
                "description": cond.description,
                "n": metrics["n"],
                "skipped": skipped,
                "auc_roc": metrics["auc_roc"],
                "auc_pr": metrics["auc_pr"],
                "eer": metrics["eer"],
                "fpr_at_tpr95": metrics["fpr_at_tpr95"],
                "delta_auc_roc": (
                    None
                    if clean_auc is None or cond.name == "clean"
                    else round(metrics["auc_roc"] - clean_auc, 4)
                ),
            }
        )
    return rows
