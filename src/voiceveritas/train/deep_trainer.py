"""Track-B trainer: precompute frozen-backbone embeddings, train an MLP head.

CPU-friendly by design: each utterance passes through the backbone exactly
once (cached to disk); the head then trains in seconds. Splits reuse the
manifest's speaker-disjoint assignment, mirroring Track A exactly.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from voiceveritas.audio import load_audio
from voiceveritas.config import DeepTrainConfig
from voiceveritas.manifest import (
    load_manifest,
    manifest_stats,
    speaker_disjoint_split,
    write_manifest,
)
from voiceveritas.metrics import compute_binary_metrics
from voiceveritas.models.wav2vec_head import EMB_VERSION, Wav2vecHead, embed_window
from voiceveritas.train.dataset import _cache_key, _middle_trim

logger = logging.getLogger(__name__)


def build_embedding_dataset(
    config: DeepTrainConfig,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Extract (or load cached) pooled embeddings for all manifest utterances."""
    utterances: list = []
    for p in config.manifests:
        utterances.extend(load_manifest(p))

    cache_path = Path(config.feature_cache_dir) / f"{EMB_VERSION}.npz"
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cache = {k: cached[k] for k in cached.files}
        logger.info("loaded %d cached embeddings", len(cache))

    rows, ys, splits, speakers = [], [], [], []
    for utt in utterances:
        key = _cache_key(utt, EMB_VERSION)
        emb = cache.get(key)
        if emb is None:
            loaded = load_audio(utt.path, sample_rate=16000)
            trimmed = _middle_trim(loaded.samples, loaded.sample_rate, config.max_duration_s)
            if len(trimmed) / 16000 < config.min_duration_s:
                continue
            emb = embed_window(trimmed, 16000, config.backbone)
            cache[key] = emb
        rows.append(emb)
        ys.append(utt.label)
        splits.append(utt.split)
        speakers.append(utt.speaker_id)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **cache)  # type: ignore[arg-type]
    X = np.vstack(rows).astype(np.float32)
    return X, np.asarray(ys, dtype=int), splits, speakers


def train_deep(config: DeepTrainConfig) -> dict:
    """Train, evaluate, persist a Track-B run. Returns the report dict."""
    import torch
    import torch.nn as nn

    if not config.manifests:
        raise ValueError("no manifests configured")

    # assign splits if missing (same logic as Track A)
    for path in config.manifests:
        utts = load_manifest(path)
        if len({u.split for u in utts}) < 2:
            speaker_disjoint_split(
                utts, train_ratio=config.train_ratio, val_ratio=config.val_ratio, seed=config.seed
            )
            write_manifest(path, utts)

    X, y, splits, _speakers = build_embedding_dataset(config)
    tr = np.array([s == "train" for s in splits])
    va = np.array([s == "val" for s in splits])
    te = np.array([s == "test" for s in splits])
    if not va.any():
        logger.warning("empty val split — reusing train as val (smoke mode)")
        va = tr

    mean = X[tr].mean(axis=0)
    scale = X[tr].std(axis=0) + 1e-6
    Z = (X - mean) / scale

    torch.manual_seed(config.seed)

    class Head(nn.Module):
        def __init__(self, in_dim: int, hidden: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, 1)
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x).squeeze(-1)

    net = Head(Z.shape[1], config.hidden_dim)
    pos_weight = torch.tensor([(tr & (y == 0)).sum() / max((tr & (y == 1)).sum(), 1)]).float()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=1e-4)

    def to_tensor(idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(Z[idx]),
            torch.from_numpy(y[idx]).float(),
        )

    Xtr, ytr = to_tensor(tr)
    Xva, _yva = to_tensor(va)

    best_val_auc, best_state, patience = -1.0, None, 0
    for epoch in range(config.epochs):
        net.train()
        opt.zero_grad()
        loss = loss_fn(net(Xtr), ytr)
        loss.backward()
        opt.step()

        net.eval()
        with torch.inference_mode():
            val_scores = torch.sigmoid(net(Xva)).numpy()
        if len(np.unique(y[va])) == 2:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(y[va], val_scores))
            if auc > best_val_auc:
                best_val_auc, patience = auc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                patience += 1
                if patience >= config.patience:
                    logger.info("early stop at epoch %d (best val AUC %.3f)", epoch, best_val_auc)
                    break
    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()

    def scores(idx: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return torch.sigmoid(net(torch.from_numpy(Z[idx]))).numpy()

    report: dict = {
        "run": {"name": config.name, "model": "wav2vec_head", "backbone": config.backbone},
        "best_val_auc": best_val_auc,
    }
    for split_name, idx in (("val", va), ("test", te)):
        if idx.any():
            report[split_name] = compute_binary_metrics(y[idx], scores(idx), label=split_name)

    run_dir = (
        Path(config.run_dir)
        if config.run_dir
        else Path("runs") / (f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.name}")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.yaml")

    Wav2vecHead(
        state_dict=net.state_dict(),
        hidden_dim=config.hidden_dim,
        embed_dim=Z.shape[1],
        scaler_mean=mean.astype(np.float32),
        scaler_scale=scale.astype(np.float32),
        backbone_name=config.backbone,
        metadata={"run_dir": str(run_dir), "config": config.to_dict()},
    ).save(run_dir / "model.joblib", train_stats=manifest_stats(load_manifest(config.manifests[0])))

    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("deep run written to %s (best val AUC %.3f)", run_dir, best_val_auc)
    return report
