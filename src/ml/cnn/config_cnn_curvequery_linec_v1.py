"""Configuration for query-conditioned curve-prior + residual model (line-C subset).

Prediction:
  y_pred(r) = c_hat  -  n_track * log(r / r0)  +  epsilon_hat(r)

Variants:
  Q1  curve-only       (enable_residual_head=False)
  Q2  curve + residual  (lambda_residual=0.05, mp4_weight=1.0)
  Q3  curve + residual + MP4 weighting  (mp4_weight=2.0)
  Q4  curve + residual + stronger regularisation  (lambda_residual=0.20)

Holdout modes (holdout_sensor):
  None       → all-sensor training
  "MP4" | "MP8" | "MP10" | "MP1" | "MP2"  → held-out-sensor generalisation
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _get_data_root() -> Path:
    return Path(r"P:\11210978-erju-ai") if os.name == "nt" else Path("/p/11210978-erju-ai")


@dataclass
class DataConfig:
    """Data loading and preprocessing."""
    parquet_root:      Path = field(default_factory=lambda: _get_data_root() / "holten_parquet")
    parquet_v2_path:   str  = ""
    waveform_root:     Path = field(default_factory=lambda: _get_data_root() / "holten_waveform")
    waveform_glob:     str  = "holten_waveform_v003_ch51_*"
    waveform_build_dir: str = ""
    waveform_scale:    float = 1e-6

    # Parquet column names — must match parquet v002 exactly
    sensor_col: str = "sensor_id"
    pgv_col:    str = "target_pgv_z_mms"
    event_col:  str = "event_id"
    track_col:  str = "track_number"
    speed_col:  str = "train_speed_kmh"

    # Line-C side -1 subset (sensor order is fixed)
    line_c_sensors: List[str] = field(
        default_factory=lambda: ["MP4", "MP8", "MP10", "MP1", "MP2"]
    )

    # Event-level split fractions
    train_fraction: float = 0.65
    val_fraction:   float = 0.15
    test_fraction:  float = 0.20
    seed_split:     int   = 42


@dataclass
class FeatureConfig:
    """Physical geometry and attenuation parameters."""
    r0:            float       = 10.0
    r_track1:      List[float] = field(default_factory=lambda: [2.5, 4.0, 8.0, 16.0, 23.0])
    r_track2:      List[float] = field(default_factory=lambda: [6.5, 8.0, 12.0, 20.0, 27.0])
    n_track1_init: float = 1.0777   # oracle reference — overwritten at fit time
    n_track2_init: float = 1.3300


@dataclass
class ModelConfig:
    """Architecture hyperparameters."""
    # Shared 2D CNN encoder (identical to CurvePriorCNN2D)
    conv_channels: List[int] = field(default_factory=lambda: [16, 32, 32, 64])
    kernel_ch:     List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    kernel_time:   List[int] = field(default_factory=lambda: [15, 9, 7, 5])
    stride_time:   List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    use_batchnorm: bool       = True

    # Metadata MLP  [speed, speed_missing, train_type, track_norm] → 4 inputs
    metadata_hidden: List[int] = field(default_factory=lambda: [32, 16])

    # Intensity head
    intensity_hidden: List[int] = field(default_factory=lambda: [64, 32])

    # Query embedding  [log(r/r0), r/r0, track/2] → 3 inputs
    query_hidden: List[int] = field(default_factory=lambda: [32, 16])

    # Residual head
    residual_hidden:      List[int] = field(default_factory=lambda: [64, 32])
    enable_residual_head: bool      = True   # False for Q1

    # Loss weights
    mp4_weight:      float = 1.0    # 2.0 for Q3 / Q4
    lambda_residual: float = 0.05   # 0.0 for Q1, 0.20 for Q4
    alpha_intensity: float = 0.2    # weight of auxiliary intensity loss L_c


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    seed:   int = 42
    device: str = "cuda"

    learning_rate: float = 3e-4
    weight_decay:  float = 1e-5
    optimizer:     str   = "adamw"

    loss_fn:     str   = "huber_log"
    huber_delta: float = 0.5

    batch_size:              int   = 16
    epochs:                  int   = 100
    patience_early_stopping: int   = 15
    gradient_clip:           float = 1.0

    pred_log_clamp_min: float = -10.0
    pred_log_clamp_max: float =   5.0


@dataclass
class OutputConfig:
    """Output paths."""
    output_root: Path = field(default_factory=lambda: _get_data_root() / "holten_models")
    output_tag:  str  = "cnn_curvequery_linec_v001"


@dataclass
class Config:
    """Master configuration for query-conditioned curve-prior model."""
    data:     DataConfig    = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model:    ModelConfig   = field(default_factory=ModelConfig)
    train:    TrainConfig   = field(default_factory=TrainConfig)
    output:   OutputConfig  = field(default_factory=OutputConfig)


def get_variant_config(
    variant: str,
    holdout_sensor: Optional[str] = None,
) -> Config:
    """Return a fully configured Config for the given variant.

    Args:
        variant:        "Q1" | "Q2" | "Q3" | "Q4"
        holdout_sensor: None (all-sensor) or sensor name to hold out

    Q1 — curve-only:                  no residual head
    Q2 — curve + residual:            lambda_residual=0.05, mp4_weight=1.0
    Q3 — curve + residual + MP4:      lambda_residual=0.05, mp4_weight=2.0
    Q4 — curve + stronger reg + MP4:  lambda_residual=0.20, mp4_weight=2.0
    """
    cfg = Config()

    if variant == "Q1":
        cfg.model.enable_residual_head = False
        cfg.model.lambda_residual      = 0.0
        cfg.model.mp4_weight           = 1.0

    elif variant == "Q2":
        cfg.model.enable_residual_head = True
        cfg.model.lambda_residual      = 0.05
        cfg.model.mp4_weight           = 1.0

    elif variant == "Q3":
        cfg.model.enable_residual_head = True
        cfg.model.lambda_residual      = 0.05
        cfg.model.mp4_weight           = 2.0

    elif variant == "Q4":
        cfg.model.enable_residual_head = True
        cfg.model.lambda_residual      = 0.20
        cfg.model.mp4_weight           = 2.0

    else:
        raise ValueError(f"Unknown variant: {variant!r}.  Choose Q1, Q2, Q3, or Q4.")

    return cfg
