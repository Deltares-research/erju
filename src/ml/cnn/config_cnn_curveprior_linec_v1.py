"""Configuration for track-conditioned curve-prior + residual model on line-C subset.

Physics-informed prediction:
  y_pred = c_hat - n_track * log(r/r0) + epsilon_hat

where:
  c_hat = predicted event intensity (learned)
  n_track = track-specific attenuation exponent (fitted on train split)
  epsilon_hat = residual correction (optional, learned)
  r = track-specific distance (fixed per sensor)
  r0 = reference distance (10 m)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _get_data_root():
    """Platform-aware path resolver for network-mounted data."""
    if os.name == "nt":
        return Path(r"P:\11210978-erju-ai")
    else:
        return Path("/p/11210978-erju-ai")


@dataclass
class DataConfig:
    """Data loading and preprocessing."""
    
    parquet_root: Path = field(default_factory=lambda: _get_data_root() / "holten_parquet")
    parquet_v2_path: str = ""
    
    waveform_root: Path = field(default_factory=lambda: _get_data_root() / "holten_waveform")
    waveform_glob: str = "holten_waveform_v003_ch51_*"
    waveform_build_dir: str = ""
    waveform_scale: float = 1e-6
    
    sensor_col: str = "sensor_id"
    pgv_col: str = "target_pgv_z_mms"
    event_col: str = "event_id"
    track_col: str = "track_number"
    
    # Line-C side -1 subset with 5 sensors
    line_c_sensors: list = field(default_factory=lambda: ["MP4", "MP8", "MP10", "MP1", "MP2"])
    
    # Event-level splits
    train_fraction: float = 0.65
    val_fraction: float = 0.15
    test_fraction: float = 0.20
    seed_split: int = 42


@dataclass
class FeatureConfig:
    """Feature engineering."""
    
    use_metadata: bool = True
    speed_col: str = "train_speed_kmh"
    
    # Physical distances for each track (meters)
    # These are active-source track distances, not distance-to-FO
    r_track1: list = field(default_factory=lambda: [2.5, 4.0, 8.0, 16.0, 23.0])
    r_track2: list = field(default_factory=lambda: [6.5, 8.0, 12.0, 20.0, 27.0])
    r0: float = 10.0  # Reference distance for attenuation curve
    
    # Attenuation exponents
    # Fitted on train split only (not used as input, only for target calculation)
    n_track1_init: float = 1.0777  # From oracle audit
    n_track2_init: float = 1.3300  # From oracle audit


@dataclass
class ModelConfig:
    """Architecture hyperparameters."""
    
    # Shared 2D CNN encoder (same as two-head)
    conv_channels: list = field(default_factory=lambda: [16, 32, 32, 64])
    kernel_ch: list = field(default_factory=lambda: [3, 3, 3, 3])
    kernel_time: list = field(default_factory=lambda: [15, 9, 7, 5])
    stride_ch: list = field(default_factory=lambda: [1, 1, 1, 1])
    stride_time: list = field(default_factory=lambda: [2, 2, 2, 2])
    use_batchnorm: bool = True
    conv_dropout: float = 0.2
    activation: str = "relu"
    
    # Metadata embedding
    metadata_hidden: list = field(default_factory=lambda: [32, 16])
    metadata_dropout: float = 0.1
    
    # Output layers
    intensity_hidden: list = field(default_factory=lambda: [64, 32])
    intensity_dropout: float = 0.2
    
    residual_hidden: list = field(default_factory=lambda: [64, 32])
    residual_dropout: float = 0.2
    n_outputs: int = 5  # 5 sensors
    
    # Loss weighting
    use_pgv_weighting: bool = False  # Set per variant
    mp4_weight: float = 1.0
    
    # Residual regularization
    lambda_residual: float = 0.05
    
    # Auxiliary curve-fitting loss
    alpha_intensity: float = 0.2
    
    # Monotonicity penalty (optional)
    lambda_monotonic: float = 0.0


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    
    seed: int = 42
    device: str = "cuda"
    
    learning_rate: float = 3e-4
    learning_rate_decay: float = 0.95
    learning_rate_decay_steps: int = 1000
    optimizer: str = "adamw"
    weight_decay: float = 1e-5
    
    loss_fn: str = "huber_log"  # mse_log or huber_log
    huber_delta: float = 0.5
    
    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = True
    
    epochs: int = 100
    patience_early_stopping: int = 15
    
    gradient_clip: float = 1.0
    
    # Prediction clamping for numerical stability
    pred_log_clamp_min: float = -10.0
    pred_log_clamp_max: float = 5.0
    
    # Attenuation exponent mode: fit_corrected, fixed_oracle, fit_current_old
    n_mode: str = "fit_corrected"


@dataclass
class EvalConfig:
    """Evaluation and logging."""
    verbose: int = 1


@dataclass
class OutputConfig:
    """Output paths."""
    output_root: Path = field(default_factory=lambda: _get_data_root() / "holten_models")
    output_tag: str = "cnn_curveprior_linec_v001"


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def get_variant_config(variant: str, n_mode: str = "fit_corrected") -> Config:
    """Get configuration preset for variant.
    
    Variants:
      P1 — curve-only (no residual)
      P2 — curve + residual (lambda_residual=0.05)
      P3 — curve + residual + MP4 weighting
      P4 — curve + residual + MP4 weighting + monotonicity
    
    Args:
      variant: Model variant (P1, P2, P3, P4)
      n_mode: Attenuation exponent mode:
        - "fit_corrected": Fit on train split using event-intercept method
        - "fixed_oracle": Use oracle reference values (1.0777, 1.3300)
        - "fit_current_old": Use current (wrong) fitting method (for debug only)
    """
    cfg = Config()
    cfg.train.n_mode = n_mode
    
    if variant == "P1":
        # Curve-only: no residual, no weighting
        cfg.model.lambda_residual = 0.0
        cfg.model.alpha_intensity = 0.2
        cfg.model.use_pgv_weighting = False
        cfg.model.lambda_monotonic = 0.0
    
    elif variant == "P2":
        # Curve + residual: regularized residual
        cfg.model.lambda_residual = 0.05
        cfg.model.alpha_intensity = 0.2
        cfg.model.use_pgv_weighting = False
        cfg.model.lambda_monotonic = 0.0
    
    elif variant == "P3":
        # Curve + residual + MP4 weighting
        cfg.model.lambda_residual = 0.05
        cfg.model.alpha_intensity = 0.2
        cfg.model.use_pgv_weighting = True
        cfg.model.mp4_weight = 2.0
        cfg.model.lambda_monotonic = 0.0
    
    elif variant == "P4":
        # Curve + residual + MP4 weighting + monotonicity
        cfg.model.lambda_residual = 0.05
        cfg.model.alpha_intensity = 0.2
        cfg.model.use_pgv_weighting = True
        cfg.model.mp4_weight = 2.0
        cfg.model.lambda_monotonic = 0.1
    
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    return cfg
