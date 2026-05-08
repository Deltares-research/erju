"""
Configuration for XGBoost v5 training.

v5 changes vs v4:
  - Input: Parquet v3 (per-line FO features from 5 track lines A/B/C/D/E,
    signed longitudinal offsets, effective distance to active track,
    physics-based train_type_family instead of alphabetical train_type_code).
  - Categorical features updated: train_type_family_code replaces train_type_code.
  - Corrected sensor geometry (track 1 at Y=4.0m, track 2 at Y=8.0m, separation 4.0m).
  - All other settings (log1p target, geometry features, hyperparameters, learning
    curve monitoring, early stopping) are identical to v4.

v4 benchmark (Parquet v2): OOF RMSE=1.79 mm/s, Test RMSE=1.79 mm/s.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class SplitConfig:
    test_fraction: float = 0.15
    random_seed: int = 42
    n_cv_folds: int = 5


@dataclass
class XGBModelConfig:
    objective: str = "reg:squarederror"
    tree_method: str = "hist"
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 1000
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42
    early_stopping_rounds: int = 50
    eval_metric: str = "rmse"
    verbosity: int = 1


@dataclass
class FeatureConfig:
    target_col: str = "target_pgv_z_mms"
    group_col: str = "event_id"
    identifier_cols: List[str] = field(
        default_factory=lambda: ["event_id", "site_id", "sensor_id"]
    )
    # v5: train_type_family_code (8 physics groups) instead of train_type_code
    categorical_cols: List[str] = field(
        default_factory=lambda: [
            "train_type_family_code",
            "acc_side_of_track",
            "track_number",
            "sensor_line_code",
        ]
    )
    string_cols: List[str] = field(
        default_factory=lambda: ["train_type", "train_type_family"]
    )


@dataclass
class FeatureEngineeringConfig:
    log_transform_target: bool = True
    add_geometry_features: bool = True


@dataclass
class OutputConfig:
    version_name: str = "xgb_v005"
    final_model_filename: str = "model_final.ubj"
    config_snapshot_filename: str = "config_snapshot.json"
    split_manifest_filename: str = "split_manifest.json"
    oof_predictions_filename: str = "oof_predictions.parquet"
    training_history_filename: str = "training_history.json"
    feature_importance_filename: str = "feature_importance.csv"
    build_log_filename: str = "build_log.txt"
    plots_subfolder: str = "plots"
    fold_curves_subfolder: str = "plots/fold_curves"


@dataclass
class XGBTrainingConfig:
    """Top-level config for XGBoost v5 training."""

    # Parquet v3 — per-line FO features, corrected geometry
    input_parquet: str = (
        r"P:\11210978-erju-ai\holten_parquet\parquet_v003_20260508_231009\dataset.parquet"
    )
    output_root_folder: str = r"P:\11210978-erju-ai\holten_models"

    split: SplitConfig = field(default_factory=SplitConfig)
    model: XGBModelConfig = field(default_factory=XGBModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    fe: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    exclude_sensor_ids: List[str] = field(
        default_factory=lambda: ["MP14", "MP15", "MP16", "MP17", "MP18", "MP19"]
    )

    experiment_notes: str = (
        "v005: Parquet v3 — per-line FO features (5 track lines A/B/C/D/E with ±5-channel sub-windows), "
        "signed longitudinal offsets (metres from sensor's line to each other line), "
        "effective distance to active track (adds 4.0m when track_number==2), "
        "physics-based train_type_family (8 groups: GO/ICM/ICR/SNG/SPR/DDZ/Locomotive/Other), "
        "corrected sensor geometry (FO at Y=0, track 1 at Y=4.0m, track 2 at Y=8.0m). "
        "Categorical features: train_type_family_code (replaces train_type_code), sensor_line_code added. "
        "Feature count: ~372 features (5×21×3=315 spectral + 5×3×3=45 time-domain + 5 offsets + geometry). "
        "Hyperparameters identical to v4 (max_depth=6, lr=0.05, log1p target). "
        "v4 benchmark (parquet v2): OOF RMSE=1.79 mm/s, Test RMSE=1.79 mm/s."
    )

    verbose_folds: bool = True

    def input_parquet_path(self) -> Path:
        return Path(self.input_parquet)

    def output_root_path(self) -> Path:
        return Path(self.output_root_folder)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


CONFIG = XGBTrainingConfig()
