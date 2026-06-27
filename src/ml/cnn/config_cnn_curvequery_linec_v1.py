"""
Configuration for query-conditioned curve-prior + residual model.

Enables prediction at arbitrary receiver distances via query features.

Model: y_pred(r) = c_hat - n_track*log(r/r0) + epsilon_hat(r)
  where epsilon_hat depends on event embedding + distance query features.

Variants:
  Q1: curve-only (epsilon_hat = 0)
  Q2: curve + residual (lambda_epsilon=0.05)
  Q3: curve + residual + MP4 weighting (lambda_epsilon=0.05, mp4_weight=2.0)
  Q4: curve + residual + strong regularization (lambda_epsilon=0.20, mp4_weight=2.0)

Held-out modes:
  all: use all five sensors
  holdout_MP4, holdout_MP8, holdout_MP10, holdout_MP1, holdout_MP2
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class FeaturesConfig:
    """Features and geometry."""
    n_track1: float = 1.0655  # fitted on train split
    n_track2: float = 1.3246  # fitted on train split
    r0_ref: float = 10.0  # reference distance (m)
    
    # Track geometry
    track1_distances: List[float] = field(default_factory=lambda: [2.5, 4.0, 8.0, 16.0, 23.0])
    track2_distances: List[float] = field(default_factory=lambda: [6.5, 8.0, 12.0, 20.0, 27.0])
    sensor_names: List[str] = field(default_factory=lambda: ["MP4", "MP8", "MP10", "MP1", "MP2"])


@dataclass
class DataConfig:
    """Data and preprocessing."""
    # FO waveform
    ch51_build_range: tuple = (1165, 1215)
    ch51_slice_start: int = 19
    ch51_slice_end: int = 40
    n_channels_selected: int = 21
    n_samples_per_channel: int = 7500
    
    # PGV target
    target_metric: str = "pgv_z"  # vertical only
    
    # Splits
    train_fraction: float = 0.65
    val_fraction: float = 0.15
    test_fraction: float = 0.20


@dataclass
class EncoderConfig:
    """Shared 2D CNN encoder."""
    conv_channels: List[int] = field(default_factory=lambda: [16, 32, 32, 64])
    kernel_ch: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    kernel_time: List[int] = field(default_factory=lambda: [15, 9, 7, 5])
    stride_time: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    padding: str = "same"
    activation: str = "relu"
    batch_norm: bool = True


@dataclass
class HeadConfig:
    """Intensity and residual heads."""
    intensity_hidden: List[int] = field(default_factory=lambda: [64, 32])
    residual_hidden: List[int] = field(default_factory=lambda: [64, 32])
    activation: str = "relu"
    
    # Query feature embedding (optional)
    query_embedding_dim: int = 16
    query_hidden: List[int] = field(default_factory=lambda: [32, 16])


@dataclass
class LossConfig:
    """Loss function configuration."""
    loss_type: str = "huber"
    huber_delta: float = 0.5
    
    # Regularization
    lambda_intensity: float = 0.2  # auxiliary intensity loss weight
    lambda_epsilon: float = 0.05  # residual L2 penalty (override per variant)
    
    # Clamping for numerical stability
    pred_log_clamp: tuple = (-10, 5)
    
    # Output weighting
    mp4_weight: float = 1.0  # override per variant


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    variant: str = "Q1"  # Q1, Q2, Q3, Q4
    holdout_sensor: Optional[str] = None  # None for all-sensor, "MP4" for holdout_MP4, etc.
    n_mode: str = "fit_corrected"  # always use corrected fitting
    
    optimizer: str = "AdamW"
    lr: float = 3e-4
    gradient_clip: float = 1.0
    
    epochs: int = 100
    batch_size: int = 16
    patience: int = 15  # early stopping
    
    seed: int = 42
    device: str = "cuda"
    
    # Query model specific
    enable_residual_head: bool = True  # False for Q1, True for Q2/Q3/Q4
    use_query_features: bool = True  # enable distance-dependent residuals


@dataclass
class QueryFeaturesConfig:
    """Features for distance query."""
    feature_set: str = "minimal"  # "minimal": log_r_ratio, r_active_m, track_id
                                   # "extended": + distance_index, sensor_id_code
    
    # Minimal features (always used)
    use_log_r_ratio: bool = True
    use_r_active_m: bool = True
    use_track_id: bool = True
    
    # Extended features (diagnostic, not for generalization)
    use_distance_index: bool = False  # position in track (0-4)
    use_sensor_id_code: bool = False  # one-hot for specific sensor
    
    n_query_features_minimal: int = 3  # log_r_ratio + r_m + track_id


@dataclass
class FullConfig:
    """Complete configuration."""
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    query_features: QueryFeaturesConfig = field(default_factory=QueryFeaturesConfig)


def get_variant_config(
    variant: str,
    holdout_sensor: Optional[str] = None,
    n_mode: str = "fit_corrected",
) -> FullConfig:
    """
    Get configuration for a specific variant.
    
    Args:
        variant: "Q1", "Q2", "Q3", "Q4"
        holdout_sensor: None (all-sensor) or sensor name ("MP4", "MP8", etc.)
        n_mode: always "fit_corrected" for query models
    
    Returns:
        FullConfig with variant-specific settings.
    """
    
    cfg = FullConfig()
    cfg.train.variant = variant
    cfg.train.holdout_sensor = holdout_sensor
    cfg.train.n_mode = n_mode
    
    # Variant-specific settings
    if variant == "Q1":
        # Curve-only: no residuals
        cfg.train.enable_residual_head = False
        cfg.loss.lambda_epsilon = 0.0
        cfg.loss.mp4_weight = 1.0
        
    elif variant == "Q2":
        # Curve + residual (light regularization)
        cfg.train.enable_residual_head = True
        cfg.loss.lambda_epsilon = 0.05
        cfg.loss.mp4_weight = 1.0
        
    elif variant == "Q3":
        # Curve + residual + MP4 weighting
        cfg.train.enable_residual_head = True
        cfg.loss.lambda_epsilon = 0.05
        cfg.loss.mp4_weight = 2.0
        
    elif variant == "Q4":
        # Curve + residual + strong regularization
        cfg.train.enable_residual_head = True
        cfg.loss.lambda_epsilon = 0.20
        cfg.loss.mp4_weight = 2.0
        
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    return cfg
