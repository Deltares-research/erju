"""
Configuration for XGBoost v4 training.

v4 changes vs v3:
  - Input: Parquet v2 (1/3-octave band FO features instead of linear 5-Hz bins)
  - Enhanced learning-curve monitoring: per-round eval history collected per fold,
    per-fold learning curve plots, aggregated CV curve with mean +/- std bands,
    early stopping inside each fold, overfit detection.
  - All other settings (log1p target, geometry features, sensor exclusions,
    hyperparameters) are identical to v3.
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
    categorical_cols: List[str] = field(
        default_factory=lambda: ["train_type_code", "acc_side_of_track", "track_number"]
    )
    string_cols: List[str] = field(default_factory=lambda: ["train_type"])


@dataclass
class FeatureEngineeringConfig:
    log_transform_target: bool = True
    add_geometry_features: bool = True


@dataclass
class OutputConfig:
    version_name: str = "xgb_v004"
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
    """Top-level config for XGBoost v4 training."""

    # Update this path after build_parquet_v2.py finishes
    input_parquet: str = (
        r"P:\11210978-erju-ai\holten_parquet\FILL_IN_V2_PARQUET_PATH\dataset.parquet"
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
        "v004: same as v003 (log1p target, geometry features, max_depth=6, MP14-MP19 excluded) "
        "but trained on Parquet v2 which uses 1/3-octave band FO spectral features "
        "(21 ISO 18405:2017 bands from 1 Hz to 100 Hz) instead of the uniform 5-Hz linear bins. "
        "Also adds enhanced learning-curve monitoring: per-round eval history per fold, "
        "per-fold learning curve plots, aggregated CV curve with mean +/- std bands, "
        "and overfit detection per fold. "
        "v003 results (parquet v1, linear bands): OOF RMSE=1.87, MAE=1.10, R2=0.41."
    )

    verbose_folds: bool = True

    def input_parquet_path(self) -> Path:
        return Path(self.input_parquet)

    def output_root_path(self) -> Path:
        return Path(self.output_root_folder)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


CONFIG = XGBTrainingConfig()
