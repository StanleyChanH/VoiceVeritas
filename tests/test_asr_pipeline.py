"""End-to-end ASR-track pipeline test with a stub transcriber (no model, no network).

Proves the plumbing — dataset builder word cache, combined feature matrix,
ProsodyASRLGBM training/saving/loading/predicting — without faster-whisper.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.conftest import SR, synthesize_speechlike
from voiceveritas.config import TrainConfig
from voiceveritas.manifest import Utterance, write_manifest
from voiceveritas.models import load_model
from voiceveritas.models.prosody_asr_lgbm import COMBINED_NAMES, ProsodyASRLGBM
from voiceveritas.train import train_baseline
from voiceveritas.train.dataset import build_feature_dataset

SMOKE_LGBM = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.1,
    "num_leaves": 7,
    "min_child_samples": 2,
    "verbosity": -1,
}


def stub_transcriber(samples: np.ndarray, sr: int) -> list:
    """Deterministic fake ASR: word cadence keyed to signal roughness.

    Rough (high zero-crossing variance) audio gets hesitant transcripts;
    smooth audio gets clean formal transcripts. Not a real ASR — just a
    label-correlated stub so the trainer has signal to learn from.
    """
    from voiceveritas.asr import WordTimes

    frames = samples[: (len(samples) // sr) * sr].reshape(-1, sr)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    rough = float(np.std(zcr)) > 0.02

    if rough:
        seq = ["i", "um", "think", "the", "uh", "the", "point", "is", "like", "hard"]
        pause, dur = 0.32, 0.16
    else:
        seq = ["the", "comprehensive", "analysis", "demonstrates", "significant", "improvement"]
        pause, dur = 0.06, 0.28

    words, t = [], 0.01
    for word in seq:
        words.append(WordTimes(word, t, t + dur, 0.99 if not rough else 0.7))
        t += dur + pause
    return words


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("asrcorpus")
    utts = []
    for spk in range(12):
        style = "read" if spk % 2 == 0 else "spontaneous"
        for i in range(2):
            rng = np.random.default_rng(7000 + spk * 10 + i)
            x = synthesize_speechlike(rng, 6.0, style=style)
            p = root / f"spk{spk:02d}_{i}.wav"
            sf.write(str(p), x, SR, subtype="PCM_16")
            utts.append(
                Utterance(
                    f"asrsyn_{spk:02d}_{i}",
                    str(p),
                    1 if style == "read" else 0,
                    6.0,
                    f"spk{spk:02d}",
                    "synthetic",
                )
            )
    m = root / "m.jsonl"
    write_manifest(m, utts)
    return m


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cache")


class TestAsrPipeline:
    def test_dataset_combined_dimensions_and_cache(self, corpus: Path, cache: Path):
        ds = build_feature_dataset(
            [str(corpus)], cache_dir=cache, with_asr=True, transcriber=stub_transcriber
        )
        assert ds.X.shape[1] == len(COMBINED_NAMES) == 34 + 15
        # prosody-only matrix keeps its own dimension (namespace isolation)
        ds_prosody = build_feature_dataset([str(corpus)], cache_dir=cache)
        assert ds_prosody.X.shape[1] == 34
        # word cache persisted and reused (second call hits it)
        from voiceveritas.asr.transcribe import load_word_cache

        wc = load_word_cache(cache)
        assert len(wc) == 24

    def test_train_save_load_predict(self, corpus: Path, cache: Path, tmp_path: Path, monkeypatch):
        # stub transcription at *predict* time too
        import voiceveritas.models.prosody_asr_lgbm as pa

        monkeypatch.setattr(pa, "transcribe_words", stub_transcriber)

        run_dir = tmp_path / "asr_run"
        train_baseline(
            TrainConfig(
                manifests=[str(corpus)],
                feature_cache_dir=str(cache),
                lgbm_params=SMOKE_LGBM,
                num_boost_round=40,
                early_stopping_rounds=10,
                calibrate=False,
                use_asr=True,
                name="asrsmoke",
                run_dir=str(run_dir),
            ),
            transcriber=stub_transcriber,
        )
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["run"]["feature_names"] == list(COMBINED_NAMES)
        assert metrics["val"]["auc_roc"] > 0.8

        # importance csv carries combined names (incl. ASR features)
        imp = (run_dir / "feature_importance.csv").read_text()
        assert "filler_rate_per_min" in imp

        model = load_model(run_dir / "model.joblib")
        assert isinstance(model, ProsodyASRLGBM)

        smooth = synthesize_speechlike(np.random.default_rng(11), 5.0, style="read")
        rough = synthesize_speechlike(np.random.default_rng(12), 5.0, style="spontaneous")
        p_read = model.predict(smooth, SR)
        p_spon = model.predict(rough, SR)
        assert 0.0 <= p_read.read_probability <= 1.0
        assert p_read.read_probability > p_spon.read_probability

    def test_conf_features_masked_by_default(self, corpus: Path, cache: Path):
        """The channel-proxy conf columns must be NaN unless opted in."""
        from voiceveritas.asr.features import ASR_FEATURE_NAMES
        from voiceveritas.models.prosody_asr_lgbm import COMBINED_NAMES

        masked = build_feature_dataset(
            [str(corpus)], cache_dir=cache, with_asr=True, transcriber=stub_transcriber
        )
        n_prosody = len(COMBINED_NAMES) - len(ASR_FEATURE_NAMES)
        for name in ("asr_conf_mean", "asr_conf_std"):
            col = masked.X[:, n_prosody + ASR_FEATURE_NAMES.index(name)]
            assert not col[~np.isnan(col)].any() if col.dtype.kind == "f" else True
            assert np.isnan(col).all(), f"{name} should be masked"

        unmasked = build_feature_dataset(
            [str(corpus)],
            cache_dir=cache,
            with_asr=True,
            transcriber=stub_transcriber,
            use_asr_conf=True,
        )
        col = unmasked.X[:, n_prosody + ASR_FEATURE_NAMES.index("asr_conf_mean")]
        assert np.isfinite(col).any()

    def test_contract_mismatch_rejected(self, tmp_path: Path):
        import joblib

        payload = {
            "type": "prosody_asr_lgbm",
            "feature_names": ["wrong"],
            "model": None,
        }
        p = tmp_path / "bad.joblib"
        joblib.dump(payload, p)
        with pytest.raises(ValueError, match="contract"):
            ProsodyASRLGBM.load(p)
