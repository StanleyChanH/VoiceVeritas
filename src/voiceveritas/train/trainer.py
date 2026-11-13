"""Track-A trainer: prosody features -> LightGBM -> isotonic calibration.

Each run writes a self-contained directory::

    runs/<timestamp>_<name>/
        config.yaml          # resolved configuration
        metrics.json         # val + test metrics
        model.joblib         # calibrated model artifact
        feature_importance.csv
        manifest_stats.json  # dataset snapshot
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from voiceveritas.config import TrainConfig
from voiceveritas.features import FEATURE_NAMES
from voiceveritas.manifest import load_manifest, manifest_stats, speaker_disjoint_split
from voiceveritas.metrics import compute_binary_metrics
from voiceveritas.models.prosody_lgbm import ProsodyLGBM
from voiceveritas.train.dataset import build_feature_dataset

logger = logging.getLogger(__name__)


def _resolve_run_dir(config: TrainConfig) -> Path:
    if config.run_dir:
        return Path(config.run_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"{stamp}_{config.name}"


def _fit_isotonic_calibration(model: lgb.LGBMClassifier, X_val: np.ndarray, y_val: np.ndarray):
    """Attach isotonic calibration fitted on the validation split.

    sklearn >= 1.6 removed ``cv="prefit"``; with ``FrozenEstimator`` the
    calibrator is still fit via ``cross_val_predict``, whose fold count must
    not exceed the smallest class count in the calibration split — hence the
    adaptive ``cv``.
    """
    try:
        from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6

        min_class = int(np.min(np.bincount(np.asarray(y_val, dtype=int))))
        cv = max(2, min(5, min_class))
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic", cv=cv)
    except ImportError:  # pragma: no cover - legacy sklearn
        calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated


def train_baseline(config: TrainConfig, transcriber=None) -> dict:
    """Train, calibrate, evaluate, and persist a Track-A run.

    ``transcriber`` overrides the real ASR backend (test injection).

    Returns the run summary dict (also written to ``<run_dir>/metrics.json``).
    """
    if not config.manifests:
        raise ValueError("no manifests configured")

    # Ensure splits exist (idempotent: respects manifests that carry splits
    # only when every utterance already has a non-default assignment).
    for manifest_path in config.manifests:
        utts = load_manifest(manifest_path)
        if len({u.split for u in utts}) < 2:
            speaker_disjoint_split(
                utts,
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
                seed=config.seed,
            )
            from voiceveritas.manifest import write_manifest

            write_manifest(manifest_path, utts)
            logger.info("assigned speaker-disjoint splits to %s", manifest_path)

    dataset = build_feature_dataset(
        config.manifests,
        cache_dir=config.feature_cache_dir,
        max_duration_s=config.max_duration_s,
        min_duration_s=config.min_duration_s,
        with_asr=config.use_asr,
        transcriber=transcriber,
        compute_ppl=config.compute_ppl,
    )
    if config.use_asr:
        from voiceveritas.models.prosody_asr_lgbm import COMBINED_NAMES

        feature_names = list(COMBINED_NAMES)
    else:
        feature_names = list(FEATURE_NAMES)

    train = dataset.by_split("train")
    val = dataset.by_split("val")
    test = dataset.by_split("test")
    if len(val.y) == 0:  # tiny datasets: reuse train as val (smoke tests only)
        logger.warning("empty val split — falling back to train-as-val (smoke mode)")
        val = train

    logger.info(
        "train=%d val=%d test=%d read_ratio(train)=%.2f",
        len(train.y),
        len(val.y),
        len(test.y),
        float(train.y.mean()),
    )

    model = lgb.LGBMClassifier(
        **config.lgbm_params,
        n_estimators=config.num_boost_round,
        class_weight="balanced",
        random_state=config.seed,
        verbose=-1,
    )
    model.fit(
        train.X,
        train.y,
        eval_set=[(val.X, val.y)],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    best_iter = getattr(model, "best_iteration_", None) or config.num_boost_round

    calibrated = model  # type: ignore[assignment]
    if config.calibrate and len(np.unique(val.y)) == 2:
        calibrated = _fit_isotonic_calibration(model, val.X, val.y)

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------
    def scores(m, ds):  # type: ignore[no-untyped-def]
        return m.predict_proba(ds.X)[:, 1]

    report: dict = {
        "run": {"name": config.name, "feature_names": feature_names},
        "data": manifest_stats(load_manifest(config.manifests[0]))
        if len(config.manifests) == 1
        else {},
        "best_iteration": int(best_iter),
    }
    for split_name, ds in (("val", val), ("test", test)):
        if len(ds.y) == 0:
            continue
        if len(np.unique(ds.y)) < 2:
            logger.warning("split %s has a single class — metrics limited", split_name)
        report[split_name] = compute_binary_metrics(ds.y, scores(calibrated, ds), label=split_name)

    # ------------------------------------------------------------------
    # persist
    # ------------------------------------------------------------------
    run_dir = _resolve_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.yaml")

    if config.use_asr:
        from voiceveritas.models.prosody_asr_lgbm import ProsodyASRLGBM

        predictor: ProsodyLGBM = ProsodyASRLGBM(
            model=calibrated,
            feature_names=feature_names,
            metadata={"run_dir": str(run_dir), "config": config.to_dict()},
            compute_ppl=config.compute_ppl,
        )
    else:
        predictor = ProsodyLGBM(
            model=calibrated,
            feature_names=feature_names,
            metadata={"run_dir": str(run_dir), "config": config.to_dict()},
        )
    predictor.save(run_dir / "model.joblib", train_stats=report.get("data", {}))

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "gain": model.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(run_dir / "feature_importance.csv", index=False)

    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("run written to %s", run_dir)
    return report
