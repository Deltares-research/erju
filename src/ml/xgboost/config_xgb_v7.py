"""
Configuration for XGBoost v7 — Scenario 2: per-event attenuation exponent.

Scenario 2: event-specific (c_i, n_i)
---------------------------------------
The model jointly predicts two targets per event:
    c_i ... log-intensity at reference distance r0
    n_i ... event-specific attenuation exponent

Implementation uses sklearn MultiOutputRegressor wrapping XGBRegressor
(trains two independent XGBoost models, one per target).

At inference, PGV at any sensor distance r is reconstructed as:
    PGV_pred = exp(c_pred - n_pred * log(r / r0))

Limitation: n_i was estimated from typically 3-9 sensor points per event,
so it may be noisy.  This scenario is worth testing but is expected to be
harder than Scenario 1 unless event-specific exponents contain real signal.

Input : Parquet v4 dataset_s2.parquet (events with quality_flag=1 only)
Target: [c_i, n_i]
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
    max_depth: int = 5
    learning_rate: float = 0.05
    n_estimators: int = 800
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    early_stopping_rounds: int = 50
    eval_metric: str = "rmse"
    verbosity: int = 1


@dataclass
class FeatureConfig:
    target_ci_col: str = "c_i"
    target_ni_col: str = "n_i"
    group_col: str = "event_id"
    identifier_cols: List[str] = field(
        default_factory=lambda: [
            "event_id",
            "site_id",
        ]
    )
    # Columns from the attenuation fit — not ML features
    curve_param_cols: List[str] = field(
        default_factory=lambda: [
            "c_i",
            "n_i",
            "n_sensors_used",
            "rmse_log",
            "r2",
            "fit_status",
            "quality_flag",
        ]
    )
    string_cols: List[str] = field(
        default_factory=lambda: [
            "train_type",
            "train_type_family",
        ]
    )
    sensor_cols: List[str] = field(
        default_factory=lambda: [
            "sensor_id",
            "acc_distance_to_track_m",
            "target_pgv_z_mms",
            "acc_side_of_track",
            "effective_distance_to_active_track_m",
            # v4 Scenario-1 columns that may also be present
            "c_i_global",  # if any
            "n_global",
            "rmse_log_fit",
            "r2_fit",
            "n_sensors_used",
        ]
    )


@dataclass
class AttenuationConfig:
    r0_m: float = 10.0
    # n_global is read from global_fit_summary.json for comparison
    global_fit_summary: str = ""


@dataclass
class EvalConfig:
    sensor_parquet: str = (
        r"P:\11210978-erju-ai\holten_parquet"
        r"\parquet_v002_20260408_151746\dataset.parquet"
    )
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
    distance_col: str = "acc_distance_to_track_m"
    pgv_col: str = "target_pgv_z_mms"


@dataclass
class OutputConfig:
    version_name: str = "xgb_v007"
    model_ci_filename: str = "model_ci.ubj"
    model_ni_filename: str = "model_ni.ubj"
    config_snapshot_filename: str = "config_snapshot.json"
    summary_filename: str = "summary.json"
    oof_predictions_filename: str = "oof_predictions.parquet"
    plots_subfolder: str = "plots"


@dataclass
class XGBv7Config:
    """Top-level configuration for XGBoost v7 (Scenario 2)."""

    # Update after running build_parquet_v4.py — point to dataset_s2.parquet
    input_parquet: str = ""

    output_root_folder: str = r"P:\11210978-erju-ai\holten_models"

    split: SplitConfig = field(default_factory=SplitConfig)
    model: XGBModelConfig = field(default_factory=XGBModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    attenuation: AttenuationConfig = field(default_factory=AttenuationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    verbose_folds: bool = True

    experiment_notes: str = (
        "XGBoost v7 — Scenario 2 (per-event n). "
        "Jointly predicts c_i and n_i per event using two separate XGBoost models "
        "(MultiOutputRegressor). "
        "PGV reconstructed at sensor distances via "
        "PGV = exp(c_i - n_i * log(r/r0)). "
        "Input: Parquet v4 dataset_s2.parquet (good-fit events only). "
        "Limitation: n_i estimated from 3-9 sensors per event — may be noisy. "
        "Baseline comparison: XGBoost v4, Test RMSE = 1.79 mm/s; "
        "XGBoost v6 (Scenario 1), see summary.json."
    )

    def input_parquet_path(self) -> Path:
        return Path(self.input_parquet)

    def output_root_path(self) -> Path:
        return Path(self.output_root_folder)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


CONFIG = XGBv7Config()
