"""Experiment configuration.

Configs are dataclasses serialized as YAML (``configs/*.yaml``). Keep them
small and explicit — every knob here exists because an experiment needed it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    """Configuration for the Track-A (prosody + LightGBM) trainer."""

    # data
    manifests: list[str] = field(default_factory=list)  # JSONL manifest paths
    max_duration_s: float = 30.0  # utterances longer than this are trimmed (middle cut)
    min_duration_s: float = 2.0  # drop ultra-short utterances
    feature_cache_dir: str = "runs/feature_cache"

    # split (used only when the manifest has no splits yet)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42

    # model
    lgbm_params: dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 10,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "verbosity": -1,
        }
    )
    num_boost_round: int = 500
    early_stopping_rounds: int = 50
    calibrate: bool = True  # isotonic calibration on the validation split

    # ASR lexical/alignment block (requires the [asr] extra at train AND
    # predict time; transcripts cached under feature_cache_dir)
    use_asr: bool = False
    compute_ppl: bool = False  # GPT-2 perplexity; needs [deep] extra, slow
    # whisper confidence features are channel-quality proxies (studio vs
    # meeting mic) — a cross-corpus leak, excluded by default (see
    # docs/benchmark.md). Enable only for same-channel training data.
    use_asr_conf: bool = False

    # noise-aware training: augment train-split audio (white/pink noise,
    # 10-25 dB SNR, +/-3 dB gain) before prosody extraction; val/test stay
    # clean so degradation is measured against reality (docs/robustness.md)
    augment: bool = False

    # output
    run_dir: str = ""  # default: runs/<timestamp>_<name>
    name: str = "baseline"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)


@dataclass
class DeepTrainConfig:
    """Configuration for the Track-B (frozen wav2vec2 + MLP head) trainer."""

    manifests: list[str] = field(default_factory=list)
    backbone: str = "facebook/wav2vec2-base"
    max_duration_s: float = 30.0
    min_duration_s: float = 2.0
    feature_cache_dir: str = "runs/feature_cache"

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42

    hidden_dim: int = 256
    lr: float = 1e-3
    epochs: int = 100
    patience: int = 10

    run_dir: str = ""
    name: str = "deep"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeepTrainConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> DeepTrainConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)
