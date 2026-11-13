"""Training pipelines."""

from voiceveritas.train.dataset import FeatureDataset, build_feature_dataset
from voiceveritas.train.trainer import train_baseline

__all__ = ["FeatureDataset", "build_feature_dataset", "train_baseline"]
