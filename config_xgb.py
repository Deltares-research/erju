"""
Configuration for XGBoost v1 training.

All settings are explicit so every training run is reproducible and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class SplitConfig:
    """Event-level split settings.

    Splitting is always done by event_id (GroupKFold) to prevent data leakage
    between training and evaluation sets.
    """

    test_fraction: float = 0.15  # fraction of events held out as final test set
    random_seed: int = 42
    n_cv_folds: int = 5  # GroupKFold folds over the non-test events


@dataclass
class XGBModelConfig:
    """XGBoost hyperparameters.

    Defaults are conservative starting points for ~1700 independent events.
    Tune max_depth and subsample after the first baseline run.
    """

    objective: str = "reg:squarederror"
    tree_method: str = "hist"  # fast histogram method
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 1000  # upper bound; early stopping decides actual rounds
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5  # reduces overfitting on small leaf nodes
    reg_alpha: float = 0.0  # L1
    reg_lambda: float = 1.0  # L2
    random_state: int = 42

    # Early stopping
    early_stopping_rounds: int = 50
    eval_metric: str = "rmse"

    # Verbosity: 0=silent, 1=warning, 2=info
    verbosity: int = 1


@dataclass
class FeatureConfig:
    """Which columns to use as predictors vs identifiers vs target."""

    target_col: str = "target_pgv_z_mms"
    group_col: str = "event_id"

    # Columns to drop before training — not predictors
    identifier_cols: List[str] = field(
        default_factory=lambda: ["event_id", "site_id", "sensor_id"]
    )
    # Categorical columns to leave as-is (XGBoost handles int-encoded categoricals natively)
    categorical_cols: List[str] = field(
        default_factory=lambda: ["train_type_code", "acc_side_of_track", "track_number"]
    )
    # String columns to drop (not encoded)
    string_cols: List[str] = field(default_factory=lambda: ["train_type"])


@dataclass
class FeatureEngineeringConfig:
    """Derived feature and target transform settings.

    These transforms operate on columns already in the Parquet and are
    applied at training time — never baked into the Parquet itself.
    Every flag is recorded in config_snapshot.json for full traceability.
    """

    # Apply log1p to target before training; predictions are back-transformed
    # with expm1 so all reported metrics and artifacts are in mm/s.
    log_transform_target: bool = True

    # Add physics-inspired geometry features derived from acc_distance_to_track_m:
    #   feat_log1p_distance  = log(1 + d)
    #   feat_inv_distance_sq = 1 / (d^2 + 1)   (vibration attenuation proxy)
    add_geometry_features: bool = True


@dataclass
class OutputConfig:
    """Output naming for model artifacts."""

    version_name: str = "xgb_v002"
    final_model_filename: str = "model_final.ubj"
    config_snapshot_filename: str = "config_snapshot.json"
    split_manifest_filename: str = "split_manifest.json"
    oof_predictions_filename: str = "oof_predictions.parquet"
    training_history_filename: str = "training_history.json"
    feature_importance_filename: str = "feature_importance.csv"
    build_log_filename: str = "build_log.txt"
    plots_subfolder: str = "plots"


@dataclass
class XGBTrainingConfig:
    """Top-level config for XGBoost v1 training."""

    input_parquet: str = (
        r"P:\11210978-erju-ai\holten_parquet\parquet_v001_20260406_031129\dataset.parquet"
    )
    output_root_folder: str = r"P:\11210978-erju-ai\holten_models"

    split: SplitConfig = field(default_factory=SplitConfig)
    model: XGBModelConfig = field(default_factory=XGBModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    fe: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Human-readable notes for this experiment — saved verbatim to config_snapshot.json.
    # Intended for paper writing and cross-experiment comparison.
    experiment_notes: str = (
        "v002: log1p target transform + physics-inspired geometry features "
        "(feat_log1p_distance, feat_inv_distance_sq) + max_depth=6. "
        "v001 baseline: OOF R2=0.22, Test RMSE=6.39 mm/s, target skewness=4.26. "
        "Distant sensors MP14-MP18 had MAE > 6 mm/s in v001."
    )

    # If True, print per-fold progress during GroupKFold
    verbose_folds: bool = True

    def input_parquet_path(self) -> Path:
        return Path(self.input_parquet)

    def output_root_path(self) -> Path:
        return Path(self.output_root_folder)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


CONFIG = XGBTrainingConfig()
