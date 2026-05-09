"""
Configuration for XGBoost v6 — Scenario 1 physics-informed model.

Scenario 1: global attenuation exponent
----------------------------------------
The model predicts c_i = log(PGV at reference distance r0) for each event.
Spatial attenuation is handled by a fixed global exponent n_global (fitted
from the data in build_parquet_v4.py, stored in global_fit_summary.json).

At inference, PGV at any sensor distance r is reconstructed as:
    PGV_pred = exp(c_pred - n_global * log(r / r0))

This separates the ML task into:
    "predict event intensity" (ML) + "apply physics propagation" (closed-form)

Input : Parquet v4 dataset.parquet (event-level, one row per train event)
Target: c_i  (log-intensity at reference distance r0 = 10 m)
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
    min_child_weight: int = 3
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    early_stopping_rounds: int = 50
    eval_metric: str = "rmse"
    verbosity: int = 1


@dataclass
class FeatureConfig:
    target_col: str = "c_i"
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
            "n_global",
            "n_sensors_used",
            "rmse_log_fit",
            "r2_fit",
        ]
    )
    string_cols: List[str] = field(
        default_factory=lambda: [
            "train_type",
            "train_type_family",
        ]
    )
    # sensor-level columns that should not appear in event-level dataset but guard anyway
    sensor_cols: List[str] = field(
        default_factory=lambda: [
            "sensor_id",
            "acc_distance_to_track_m",
            "target_pgv_z_mms",
            "acc_side_of_track",
            "effective_distance_to_active_track_m",
        ]
    )


@dataclass
class AttenuationConfig:
    """Physics parameters — must match build_parquet_v4 settings."""

    r0_m: float = 10.0
    # n_global is read at runtime from global_fit_summary.json
    global_fit_summary: str = ""


@dataclass
class EvalConfig:
    """Paths needed for sensor-level reconstruction evaluation."""

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
    version_name: str = "xgb_v006"
    model_filename: str = "model_final.ubj"
    config_snapshot_filename: str = "config_snapshot.json"
    summary_filename: str = "summary.json"
    oof_predictions_filename: str = "oof_predictions.parquet"
    plots_subfolder: str = "plots"


@dataclass
class XGBv6Config:
    """Top-level configuration for XGBoost v6 (Scenario 1)."""

    # Update this path after running build_parquet_v4.py
    input_parquet: str = ""  # e.g. ...parquet_v004_YYYYMMDD_HHMMSS/dataset.parquet

    output_root_folder: str = r"P:\11210978-erju-ai\holten_models"

    split: SplitConfig = field(default_factory=SplitConfig)
    model: XGBModelConfig = field(default_factory=XGBModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    attenuation: AttenuationConfig = field(default_factory=AttenuationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    verbose_folds: bool = True

    experiment_notes: str = (
        "XGBoost v6 — Scenario 1 (global n). "
        "Predicts event log-intensity c_i from FO + texture + train-metadata. "
        "PGV reconstructed at sensor distances via PGV = exp(c_i - n_global * log(r/r0)). "
        "Input: Parquet v4 (event-level, one row per train event). "
        "Baseline comparison: XGBoost v4, Test RMSE = 1.79 mm/s."
    )

    def input_parquet_path(self) -> Path:
        return Path(self.input_parquet)

    def output_root_path(self) -> Path:
        return Path(self.output_root_folder)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


CONFIG = XGBv6Config()
