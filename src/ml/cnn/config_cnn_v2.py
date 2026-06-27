"""Configuration for the multi-channel FO waveform CNN (cnn v2).

Model
-----
    FO channel window (1, 21, T)  -- 21 channels x time
        -> compact 2D CNN over channel x time -> event embedding
    concat( embedding , scalar features )
        -> MLP head -> log(PGV_z)

The 2D convolutions span a few channels x many time samples, so the network
can pick up the inclined train-passage wavefront (the space-time slope) that a
single channel cannot expose.

Reuses cnn_v1 conventions: fixed global waveform scale (strain->microstrain),
event-level splits identical to XGBoost v4, scalar features standardized on the
training fold only, waveform NOT normalized.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class DataConfig:
    waveform_build_dir: str | None = None         # holten_waveform_v002_*
    parquet_v2_path: str | None = None

    waveform_root: str = r"P:\11210978-erju-ai\holten_waveform"
    waveform_glob: str = "holten_waveform_v002_*"
    parquet_root: str = r"P:\11210978-erju-ai\holten_parquet"

    event_col: str = "event_id"
    sensor_col: str = "sensor_id"
    pgv_col: str = "target_pgv_z_mms"
    distance_col: str = "effective_distance_to_active_track_m"
    side_col: str = "acc_side_of_track"

    waveform_scale: float = 1.0e6                 # global strain->microstrain

    exclude_sensor_ids: List[str] = field(
        default_factory=lambda: ["MP14", "MP15", "MP16", "MP17", "MP18", "MP19"]
    )


@dataclass
class FeatureConfig:
    r0_m: float = 10.0
    use_inverse_distance: bool = True
    use_metadata: bool = True
    metadata_cols: List[str] = field(
        default_factory=lambda: ["train_speed_kmh", "train_type_code", "track_number"]
    )
    speed_col: str = "train_speed_kmh"
    add_missing_speed_flag: bool = True


@dataclass
class ModelConfig:
    """Compact 2D CNN over (channel x time)."""

    use_waveform: bool = True
    conv_channels: List[int] = field(default_factory=lambda: [16, 32, 64, 64])
    kernel_ch: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    kernel_time: List[int] = field(default_factory=lambda: [15, 9, 7, 5])
    stride_ch: List[int] = field(default_factory=lambda: [1, 1, 2, 2])
    stride_time: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    use_batchnorm: bool = True
    conv_dropout: float = 0.0
    head_hidden: List[int] = field(default_factory=lambda: [64])
    head_dropout: float = 0.1
    activation: str = "relu"


@dataclass
class SplitConfig:
    test_fraction: float = 0.15
    test_seed: int = 42                            # SAME as XGBoost v4 / cnn_v1
    val_fraction: float = 0.15
    val_seed: int = 42


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 30
    target_transform: str = "log"
    target_eps: float = 1e-6
    num_workers: int = 0
    seed: int = 1234
    use_tensorboard: bool = True
    tensorboard_subdir: str = "runs"
    lr_scheduler: bool = False


@dataclass
class OutputConfig:
    version_name: str = "cnn_v002"
    output_root_folder: str = r"P:\11210978-erju-ai\holten_models"
    best_model_filename: str = "best_model.pt"
    summary_filename: str = "summary.json"
    config_snapshot_filename: str = "config_snapshot.json"
    predictions_filename: str = "predictions.parquet"
    history_filename: str = "training_history.csv"
    split_manifest_filename: str = "split_manifest.json"
    plots_subfolder: str = "plots"


@dataclass
class CNNv2Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    benchmark_v4_rmse: float = 1.79
    benchmark_cnn_v1_rmse: float = 1.876           # best cnn_v1 (no-BN Mode B)
    benchmark_scalar_only_rmse: float = 1.879

    experiment_notes: str = (
        "cnn_v2: multi-channel FO waveform (21 ch, 1184-1204) -> 2D CNN over "
        "channel x time -> log(PGV_z). Same split/preprocessing as cnn_v1."
    )

    def output_root_path(self) -> Path:
        return Path(self.output.output_root_folder)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


CONFIG = CNNv2Config()
