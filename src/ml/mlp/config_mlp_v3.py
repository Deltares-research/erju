"""config_mlp_v1.py — Configuration for MLP v1 training.

All hyperparameters, paths, and training settings in one place.
Edit this file to create new experiment versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ── Paths ─────────────────────────────────────────────────────────────────────
PARQUET_PATH = Path(
    r"P:\11210978-erju-ai\holten_parquet\parquet_v002_20260408_151746\dataset.parquet"
)
MODELS_ROOT = Path(r"P:\11210978-erju-ai\holten_models")


# ── Data config ───────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    target_col: str = "target_pgv_z_mms"
    group_col: str = "event_id"
    # Columns to drop before building feature matrix
    identifier_cols: List[str] = field(
        default_factory=lambda: [
            "event_id",
            "site_id",
            "sensor_id",
        ]
    )
    string_cols: List[str] = field(
        default_factory=lambda: [
            "train_type",
        ]
    )
    # Sensors with known unit issues — excluded at build time in parquet v2,
    # but kept here for documentation.
    exclude_sensor_ids: List[str] = field(
        default_factory=lambda: [
            "MP14",
            "MP15",
            "MP16",
            "MP17",
            "MP18",
            "MP19",
        ]
    )
    # Fraction of events held out as final test set (never seen during training)
    test_size: float = 0.15
    # Fraction of remaining (train_val) events used as validation
    val_size: float = 0.15
    random_seed: int = 42


# ── Feature engineering config ────────────────────────────────────────────────
@dataclass
class FeatureEngineeringConfig:
    add_geometry_features: bool = True  # log1p_distance, inv_distance_sq
    log_transform_target: bool = True  # train on log1p(y), report RMSE in mm/s


# ── Model architecture ─────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "relu"  # "relu" | "tanh" | "gelu"
    dropout: float = 0.2  # regularise the deeper network


# ── Training config ───────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    epochs: int = 300
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0  # L2 regularisation (0 = off for v1)
    # Early stopping: stop if val loss does not improve for this many epochs
    patience: int = 50
    # Save a checkpoint every N epochs (in addition to the best-model checkpoint)
    checkpoint_every_n_epochs: int = 50
    # TensorBoard log directory (relative to build folder)
    tensorboard_subdir: str = "runs"


# ── Top-level config ──────────────────────────────────────────────────────────
@dataclass
class Config:
    version_name: str = "mlp_v003"
    parquet_path: Path = PARQUET_PATH
    models_root: Path = MODELS_ROOT
    data: DataConfig = field(default_factory=DataConfig)
    fe: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CONFIG = Config()
