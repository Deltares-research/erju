"""Configuration for Parquet v4 — event-level, attenuation-curve targets.

v4 is derived from the sensor-level Parquet v2 (NOT from raw NetCDF).
The build script:
  1. Loads the v2 Parquet.
  2. Aggregates FO / train-metadata features to one row per event.
  3. Adds 5 spectral texture features from the octave-band mean powers.
  4. Fits Scenario 1: global attenuation exponent → per-event c_i targets.
  5. Fits Scenario 2: per-event (c_i, n_i) targets.
  6. Saves the event-level dataset to Parquet.

Update `input_parquet` after running build_parquet_v2.py to point to the
latest v2 build folder.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class AttenuationConfig:
    """Physics parameters for the power-law attenuation model."""

    r0_m: float = 10.0
    """Reference distance in metres.  All c_i values are at this distance."""

    min_sensors_global: int = 2
    """Minimum valid sensor rows per event for global-n fit inclusion."""

    min_sensors_event: int = 3
    """Minimum valid sensor rows per event for per-event curve fitting."""

    max_abs_n: float = 5.0
    """Events with |n_i| > this are flagged as quality_flag=0 in Scenario 2."""


@dataclass
class OutputConfig:
    version_name: str = "parquet_v004"
    dataset_filename: str = "dataset.parquet"
    scenario1_params_filename: str = "attenuation_global.parquet"
    scenario2_params_filename: str = "attenuation_per_event.parquet"
    global_fit_summary_filename: str = "global_fit_summary.json"
    build_log_filename: str = "build_log.txt"


@dataclass
class ParquetV4Config:
    """Top-level configuration for building Parquet v4."""

    # ---- Input (v2 sensor-level Parquet) -----------------------------------
    input_parquet: str = (
        r"P:\11210978-erju-ai\holten_parquet"
        r"\parquet_v002_20260408_151746\dataset.parquet"
    )

    # ---- Output root -------------------------------------------------------
    output_root_folder: str = r"P:\11210978-erju-ai\holten_parquet"

    # ---- Columns to exclude from the event-level feature set ---------------
    sensor_specific_cols: List[str] = field(
        default_factory=lambda: [
            "sensor_id",
            "acc_distance_to_track_m",
            "target_pgv_z_mms",
            "acc_side_of_track",
            # v3-specific columns that may not be present in v2 — safely ignored
            "effective_distance_to_active_track_m",
            "sensor_line_code",
            "train_type_family",
            "train_type_family_code",
        ]
    )

    # ---- Sensor-level columns used only for attenuation fitting ------------
    distance_col: str = "effective_distance_to_active_track_m"
    pgv_col: str = "target_pgv_z_mms"
    event_col: str = "event_id"

    # ---- Sensor exclusions (same as v2 / XGBoost v4) -----------------------
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

    # ---- Attenuation model config ------------------------------------------
    attenuation: AttenuationConfig = field(default_factory=AttenuationConfig)

    # ---- Output config -----------------------------------------------------
    output: OutputConfig = field(default_factory=OutputConfig)

    def input_parquet_path(self) -> Path:
        return Path(self.input_parquet)

    def output_root_path(self) -> Path:
        return Path(self.output_root_folder)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


CONFIG = ParquetV4Config()
