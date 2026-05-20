"""Configuration for XGBoost v8 — two-stage residual-on-physics model.

Architecture
------------
Stage 1  (event-level):
    FO features + train metadata  →  predict c_i
    (same as XGBoost v6, but now only used to build a physics prior)

Stage 2  (sensor-level):
    event features + distance features + z_phys + c_hat  →  log(PGV)
    where z_phys = c_hat - n_global * log(r / r0)

Two variants are evaluated:
    Variant A  — stage-2 target is log(PGV) directly, z_phys is a feature
    Variant B  — stage-2 target is the residual ε = log(PGV) - z_phys,
                 final prediction = z_phys + ε_hat

Leakage prevention
------------------
Inside each outer CV fold the stage-1 model generates out-of-fold (OOF)
c_hat predictions for the training events via an inner CV.  Those OOF
predictions — not the true fitted c_i — are used to compute the physics
prior z_phys for the stage-2 training rows.  This ensures stage-2 sees
the same quality of physics prior during training as it will at inference.

n_global is re-fitted from training sensor rows inside each outer fold.
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
    n_inner_folds: int = 5  # inner CV for OOF c_hat generation


@dataclass
class Stage1ModelConfig:
    """XGBoost config for the event-level c_i predictor."""

    objective: str = "reg:squarederror"
    tree_method: str = "hist"
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 1000
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    early_stopping_rounds: int = 50
    eval_metric: str = "rmse"
    # Fixed rounds used for the inner-CV OOF generation (no early stopping there)
    inner_oof_n_estimators: int = 60


@dataclass
class Stage2ModelConfig:
    """XGBoost config for the sensor-level log(PGV) predictor."""

    objective: str = "reg:squarederror"
    tree_method: str = "hist"
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 1000
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    early_stopping_rounds: int = 50
    eval_metric: str = "rmse"


@dataclass
class PhysicsConfig:
    r0_m: float = 10.0
    # n_global is re-fitted per fold from training sensor rows
    # This value is a fallback for final-model builds where all data is used
    n_global_fallback: float = 0.8162


@dataclass
class FeatureConfig:
    """Column roles for both stages."""

    # Event identifier
    group_col: str = "event_id"

    # Sensor-identifier column: drop from features
    sensor_id_col: str = "sensor_id"

    # Raw PGV target column in v2 (not a feature)
    pgv_col: str = "target_pgv_z_mms"

    # Distance column (kept as a raw feature AND used for physics transforms)
    distance_col: str = "effective_distance_to_active_track_m"

    # Side-of-track (kept as feature — will be label-encoded if string)
    side_col: str = "acc_side_of_track"

    # Columns that are pure v4 attenuation artefacts (not available at inference)
    # These are also excluded from stage-1 feature matrix
    attenuation_cols: List[str] = field(
        default_factory=lambda: [
            "c_i",
            "n_i",
            "n_global",
            "n_sensors_used",
            "rmse_log_fit",
            "r2_fit",
            "quality_flag",
            "rmse_log",
            "r2",
            "fit_status",
        ]
    )

    # String columns to drop from both feature matrices
    string_cols: List[str] = field(
        default_factory=lambda: [
            "train_type",
            "train_type_family",
        ]
    )

    # Sensor IDs to exclude (same as v4)
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


@dataclass
class EvalConfig:
    sensor_parquet: str = (
        r"P:\11210978-erju-ai\holten_parquet"
        r"\parquet_v002_20260408_151746\dataset.parquet"
    )


@dataclass
class OutputConfig:
    version_name: str = "xgb_v008"
    model_s1_filename: str = "model_stage1_final.ubj"
    model_s2a_filename: str = "model_stage2_varA_final.ubj"
    model_s2b_filename: str = "model_stage2_varB_final.ubj"
    summary_filename: str = "summary.json"
    config_snapshot_filename: str = "config_snapshot.json"
    oof_predictions_filename: str = "oof_predictions.parquet"
    plots_subfolder: str = "plots"


@dataclass
class XGBv8Config:
    """Top-level config for XGBoost v8 — residual-on-physics."""

    # Parquet v4 event-level dataset (auto-discovered if empty)
    input_parquet_v4: str = ""

    # Parquet v2 sensor-level dataset
    input_parquet_v2: str = (
        r"P:\11210978-erju-ai\holten_parquet"
        r"\parquet_v002_20260408_151746\dataset.parquet"
    )

    # Global fit summary from parquet_v4 build (n_global, r0)
    global_fit_summary: str = ""  # auto-discovered if empty

    output_root_folder: str = r"P:\11210978-erju-ai\holten_models"

    split: SplitConfig = field(default_factory=SplitConfig)
    stage1: Stage1ModelConfig = field(default_factory=Stage1ModelConfig)
    stage2: Stage2ModelConfig = field(default_factory=Stage2ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    verbose_folds: bool = True

    experiment_notes: str = (
        "XGBoost v8 — two-stage residual-on-physics model. "
        "Stage 1: event features -> c_i. "
        "Stage 2: sensor features + z_phys -> log(PGV) [Variant A] "
        "or residual epsilon [Variant B]. "
        "OOF c_hat used for training physics prior to avoid leakage. "
        "n_global re-fitted per outer CV fold from training events only."
    )

    def output_root_path(self) -> Path:
        return Path(self.output_root_folder)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


CONFIG = XGBv8Config()
