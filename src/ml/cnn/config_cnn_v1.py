"""Configuration for the single-channel FO waveform CNN baseline (cnn v1).

Model
-----
    FO channel-1194 waveform (1, T)
        -> 1D CNN encoder -> event embedding
    concat( embedding , scalar features )
        -> small MLP head -> log(PGV_z)

Scalar features = distance features (+ optional train metadata).  The waveform
is NOT normalized (amplitude is physically meaningful); only the scalar
features are standardized, fit on the training fold only.

Splits reuse the exact event-level test split from the XGBoost benchmark
(test_fraction=0.15, seed=42) so results are directly comparable to v4.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class DataConfig:
    """Dataset locations and join columns."""

    # Auto-discover the latest build if these are left as None.
    waveform_build_dir: str | None = None         # holten_waveform_v001_*
    parquet_v2_path: str | None = None            # sensor-level dataset.parquet

    waveform_root: str = r"P:\11210978-erju-ai\holten_waveform"
    parquet_root: str = r"P:\11210978-erju-ai\holten_parquet"

    event_col: str = "event_id"
    sensor_col: str = "sensor_id"
    pgv_col: str = "target_pgv_z_mms"
    distance_col: str = "effective_distance_to_active_track_m"
    side_col: str = "acc_side_of_track"

    # Fixed GLOBAL waveform scale applied to every event identically (NOT
    # per-event normalization).  Raw FO strain is ~1e-6; multiplying by 1e6
    # converts to microstrain so inputs are O(1), which keeps Conv/BatchNorm
    # numerically well-conditioned.  Relative amplitude is fully preserved.
    waveform_scale: float = 1.0e6

    exclude_sensor_ids: List[str] = field(
        default_factory=lambda: ["MP14", "MP15", "MP16", "MP17", "MP18", "MP19"]
    )


@dataclass
class FeatureConfig:
    """Scalar (non-waveform) feature engineering."""

    r0_m: float = 10.0

    # Distance features (always on).
    use_inverse_distance: bool = True        # adds 1/r and 1/sqrt(r)

    # Metadata ablation switch:
    #   False -> Mode A: waveform + distance only
    #   True  -> Mode B: waveform + distance + train metadata
    use_metadata: bool = True
    metadata_cols: List[str] = field(
        default_factory=lambda: ["train_speed_kmh", "train_type_code", "track_number"]
    )
    # train_speed_kmh missing handling: median impute (train fold only) + flag.
    speed_col: str = "train_speed_kmh"
    add_missing_speed_flag: bool = True


@dataclass
class ModelConfig:
    """1D CNN encoder + MLP head."""

    use_waveform: bool = True                # False -> scalar-only MLP baseline
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 128])
    conv_kernels: List[int] = field(default_factory=lambda: [15, 9, 7, 5])
    conv_strides: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    use_batchnorm: bool = True               # configurable per the amplitude note
    conv_dropout: float = 0.0
    head_hidden: List[int] = field(default_factory=lambda: [64])
    head_dropout: float = 0.1
    activation: str = "relu"                 # "relu" | "gelu"


@dataclass
class SplitConfig:
    test_fraction: float = 0.15
    test_seed: int = 42                      # SAME as XGBoost v4 test split
    val_fraction: float = 0.15
    val_seed: int = 42                       # +1 applied internally -> 43


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 30                       # early stopping on val loss
    target_transform: str = "log"            # natural log of PGV_z
    target_eps: float = 1e-6
    num_workers: int = 0
    seed: int = 1234
    use_tensorboard: bool = True
    tensorboard_subdir: str = "runs"
    lr_scheduler: bool = False               # ReduceLROnPlateau (off by default)


@dataclass
class OutputConfig:
    version_name: str = "cnn_v001"
    output_root_folder: str = r"P:\11210978-erju-ai\holten_models"
    best_model_filename: str = "best_model.pt"
    summary_filename: str = "summary.json"
    config_snapshot_filename: str = "config_snapshot.json"
    predictions_filename: str = "predictions.parquet"
    history_filename: str = "training_history.csv"
    split_manifest_filename: str = "split_manifest.json"
    plots_subfolder: str = "plots"


@dataclass
class CNNv1Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    benchmark_v4_rmse: float = 1.79          # XGBoost v4 direct sensor-level

    experiment_notes: str = (
        "cnn_v1: first beyond-tabular baseline. Single FO channel 1194 waveform "
        "(1-100 Hz bandpass, 250 Hz, 30 s energy-centred crop, no normalization) "
        "+ corrected distance features + optional train metadata -> log(PGV_z). "
        "Event-level splits identical to XGBoost v4 (test seed 42)."
    )

    def output_root_path(self) -> Path:
        return Path(self.output.output_root_folder)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


CONFIG = CNNv1Config()
