"""Configuration for track-conditioned two-head multi-output CNN on line-C subset.

Event-level prediction:
  FO waveform (21 channels) + metadata → track-specific attenuation profile
  
Target formulation:
  Track 1: [log(PGV) @ 2.5m, 4m, 8m, 16m, 23m]
  Track 2: [log(PGV) @ 6.5m, 8m, 12m, 20m, 27m]

Output heads:
  - head_track_1 → 5 outputs
  - head_track_2 → 5 outputs
  
Loss: use only the head matching the event's active track
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
    
    # Parquet dataset
    parquet_root: Path = field(default_factory=lambda: _get_data_root() / "holten_parquet")
    parquet_v2_path: str = ""  # If empty, auto-discover parquet_v002_*
    
    # Waveform builds
    waveform_root: Path = field(default_factory=lambda: _get_data_root() / "holten_waveform")
    waveform_glob: str = "holten_waveform_v003_ch21_linec_*"
    waveform_build_dir: str = ""  # If empty, auto-discover
    waveform_scale: float = 1e-6  # Scale int16 → microstrain
    
    # Columns
    sensor_col: str = "sensor_id"
    pgv_col: str = "target_pgv_z_mms"
    event_col: str = "event_id"
    track_col: str = "track_number"
    
    # Line-C side -1 subset
    line_c_sensors: list = field(default_factory=lambda: ["MP4", "MP8", "MP10", "MP1", "MP2"])
    line_c_side: int = -1
    
    # Exclude any known bad sensors
    exclude_sensor_ids: list = field(default_factory=list)
    
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
    speed_fill_percentile: float = 50.0  # Median
    add_missing_speed_flag: bool = True
    
    # Do NOT use distance features in this fixed-output model
    # (distance is implicit in the output head selection)
    use_distance_features: bool = False


@dataclass
class ModelConfig:
    """Architecture hyperparameters."""
    
    # Shared 2D CNN encoder on waveform
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
    
    # Output heads (track 1 and track 2)
    head_hidden: list = field(default_factory=lambda: [64, 32])
    head_dropout: float = 0.2
    n_outputs: int = 5  # 5 sensors per track
    
    # Monotonicity enforcement (Variant B)
    enforce_monotonicity: bool = False
    monotonic_parameterization: bool = True  # vs penalty-based
    monotonic_weight: float = 0.1
    
    # High-PGV weighting (Variant C)
    use_pgv_weighting: bool = False
    pgv_weight_threshold: float = 4.0  # mm/s
    pgv_weight_high: float = 2.0
    pgv_weight_low: float = 1.0
    mp4_weight: float = 1.0  # Can override per-sensor


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    
    seed: int = 42
    device: str = "cuda"  # cuda or cpu
    
    # Optimization
    learning_rate: float = 1e-3
    learning_rate_decay: float = 0.95
    learning_rate_decay_steps: int = 1000
    optimizer: str = "adam"  # adam, adamw, sgd
    weight_decay: float = 1e-5
    
    # Loss
    loss_fn: str = "mse_log"  # mse_log, huber_log
    huber_delta: float = 0.5  # For Huber loss
    
    # Batching and scheduling
    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = True
    
    epochs: int = 50
    patience_early_stopping: int = 10
    patience_reduce_lr: int = 5
    
    gradient_clip: float = 1.0


@dataclass
class EvalConfig:
    """Evaluation and logging."""
    
    verbose: int = 1  # 0: silent, 1: per epoch, 2: detailed
    log_interval: int = 100  # steps
    
    # Metrics to compute
    compute_per_track_metrics: bool = True
    compute_per_sensor_metrics: bool = True
    compute_high_pgv_diagnostics: bool = True
    compute_monotonicity_diagnostics: bool = True
    
    # Plots to generate
    plot_learning_curve: bool = True
    plot_measured_vs_predicted: bool = True
    plot_residuals_vs_pgv: bool = True
    plot_profile_examples: bool = True
    plot_monotonicity: bool = True
    plot_per_track: bool = True


@dataclass
class OutputConfig:
    """Output directory and naming."""
    
    output_root: Path = field(default_factory=lambda: _get_data_root() / "holten_models")
    output_tag: str = "cnn_twohead_linec_v001"
    
    # Auto-created subdirectories in main output folder
    create_timestamped_folder: bool = True


# ==================================================================================
# MAIN CONFIG OBJECT
# ==================================================================================

@dataclass
class Config:
    """Master configuration for track-conditioned two-head CNN."""
    
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Model variant selector
    variant: str = "A"  # A, B, C, D (combinations of features)


# ==================================================================================
# VARIANT PRESETS
# ==================================================================================

def get_variant_config(variant: str) -> Config:
    """Load preset config for a specific variant.
    
    Variants:
    A — direct two-head, unweighted MSE/log
    B — direct two-head, MP4-weighted
    C — monotonic two-head, unweighted
    D — monotonic two-head, MP4-weighted
    """
    cfg = Config()
    cfg.variant = variant
    
    if variant == "A":
        # Direct, unweighted
        cfg.model.enforce_monotonicity = False
        cfg.model.use_pgv_weighting = False
    
    elif variant == "B":
        # Direct, MP4-weighted
        cfg.model.enforce_monotonicity = False
        cfg.model.use_pgv_weighting = True
        cfg.model.mp4_weight = 2.0
    
    elif variant == "C":
        # Monotonic, unweighted
        cfg.model.enforce_monotonicity = True
        cfg.model.monotonic_parameterization = True
        cfg.model.monotonic_weight = 0.1
        cfg.model.use_pgv_weighting = False
    
    elif variant == "D":
        # Monotonic, MP4-weighted
        cfg.model.enforce_monotonicity = True
        cfg.model.monotonic_parameterization = True
        cfg.model.monotonic_weight = 0.1
        cfg.model.use_pgv_weighting = True
        cfg.model.mp4_weight = 2.0
    
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    return cfg


# ==================================================================================
# DEFAULT CONFIG INSTANCE
# ==================================================================================

CONFIG = Config()
