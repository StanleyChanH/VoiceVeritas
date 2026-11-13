"""End-to-end smoke test: synthetic corpus -> manifest -> train -> reload -> predict.

This validates the whole Track-A pipeline without any network access or
real datasets. Marked slow (trains a real, if tiny, LightGBM).

Note on LightGBM params: the corpus is tiny, so ``min_child_samples`` must be
small — with the default 10, a 9-vs-9 split is rejected and the booster
degenerates to a constant predictor (a nice trap this test guards against).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import SR, synthesize_speechlike
from voiceveritas.config import TrainConfig
from voiceveritas.manifest import Utterance, load_manifest, write_manifest
from voiceveritas.models import load_model
from voiceveritas.models.prosody_lgbm import ProsodyLGBM
from voiceveritas.train import train_baseline

SMOKE_LGBM = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.1,
    "num_leaves": 7,
    "min_child_samples": 2,
    "verbosity": -1,
}


@pytest.fixture(scope="module")
def synthetic_manifest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic corpus: 24 speakers x 2 utts x 2 styles, 6 s each.

    12 speakers per label is the minimum for the greedy splitter to produce
    non-empty train/val/test per label at the default ratios.
    """
    import soundfile as sf

    root = tmp_path_factory.mktemp("corpus")
    utts: list[Utterance] = []
    for spk in range(24):
        style = "read" if spk % 2 == 0 else "spontaneous"
        for i in range(2):
            rng = np.random.default_rng(1000 + spk * 10 + i)
            x = synthesize_speechlike(rng, 6.0, style=style)
            path = root / f"spk{spk:02d}_{i}.wav"
            sf.write(str(path), x, SR, subtype="PCM_16")
            utts.append(
                Utterance(
                    utt_id=f"synth_{spk:02d}_{i}",
                    path=str(path),
                    label=1 if style == "read" else 0,
                    duration=6.0,
                    speaker_id=f"spk{spk:02d}",
                    dataset="synthetic",
                )
            )
    manifest = root / "manifest.jsonl"
    write_manifest(manifest, utts)
    return manifest


@pytest.fixture(scope="module")
def shared_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cache")


@pytest.fixture(scope="module")
def trained_run(synthetic_manifest: Path, shared_cache: Path) -> Path:
    config = TrainConfig(
        manifests=[str(synthetic_manifest)],
        feature_cache_dir=str(shared_cache),
        lgbm_params=SMOKE_LGBM,
        num_boost_round=60,
        early_stopping_rounds=15,
        calibrate=False,
        name="smoke",
    )
    train_baseline(config)
    return _latest_run_dir()


def _latest_run_dir() -> Path:
    runs = sorted(Path("runs").glob("*_smoke"))
    assert runs, "no smoke run directory created"
    return runs[-1]


class TestCalibrationPath:
    """The isotonic-calibration code path (caught a real sklearn API break)."""

    def test_calibrated_model_trains_and_predicts(
        self, synthetic_manifest: Path, shared_cache: Path, tmp_path: Path
    ):
        config = TrainConfig(
            manifests=[str(synthetic_manifest)],
            feature_cache_dir=str(shared_cache),
            lgbm_params=SMOKE_LGBM,
            num_boost_round=30,
            early_stopping_rounds=10,
            calibrate=True,
            run_dir=str(tmp_path / "run_cal"),
        )
        report = train_baseline(config)
        assert (tmp_path / "run_cal" / "model.joblib").exists()
        assert report["val"]["auc_roc"] is not None

        model = load_model(tmp_path / "run_cal" / "model.joblib")
        x = synthesize_speechlike(np.random.default_rng(7), 5.0, style="spontaneous")
        result = model.predict(x, SR)
        assert 0.0 <= result.read_probability <= 1.0


class TestTrainSmoke:
    def test_train_produces_artifacts_and_learns(self, trained_run: Path):
        for artifact in ("model.joblib", "metrics.json", "config.yaml", "feature_importance.csv"):
            assert (trained_run / artifact).exists(), f"missing {artifact}"
        metrics = json.loads((trained_run / "metrics.json").read_text())
        assert metrics["val"]["auc_roc"] is not None
        # caricatured styles must be learnable; guards against the
        # constant-predictor degeneration described in the module docstring
        assert metrics["val"]["auc_roc"] > 0.8

    def test_manifest_has_speaker_disjoint_splits(self, synthetic_manifest: Path):
        utts = load_manifest(synthetic_manifest)
        assert {u.split for u in utts} >= {"train", "val", "test"}
        by_split: dict[str, set[str]] = {}
        for u in utts:
            by_split.setdefault(u.split, set()).add(u.speaker_id)
        flat = [s for v in by_split.values() for s in v]
        assert len(flat) == len(set(flat)), "speaker leaked across splits"

    def test_model_reload_and_predict(self, trained_run: Path, rng):
        model = load_model(trained_run / "model.joblib")
        assert isinstance(model, ProsodyLGBM)

        read_x = synthesize_speechlike(rng, 5.0, style="read")
        spon_x = synthesize_speechlike(rng, 5.0, style="spontaneous")
        p_read = model.predict(read_x, SR)
        p_spon = model.predict(spon_x, SR)
        for p in (p_read, p_spon):
            assert 0.0 <= p.read_probability <= 1.0
            assert len(p.segments) >= 1
        assert p_read.read_probability > p_spon.read_probability
        assert p_read.top_contributors is not None

    def test_feature_contract_enforced(self, tmp_path):
        import joblib

        payload = {
            "type": "prosody_lgbm",
            "created": "test",
            "package_version": "0.0.0",
            "feature_names": ["not", "the", "right", "features"],
            "model": None,
        }
        path = tmp_path / "bogus_model.joblib"
        joblib.dump(payload, path)
        with pytest.raises(ValueError, match="feature contract"):
            ProsodyLGBM.load(path)
