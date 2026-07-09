"""Configuration for Parquet dataset building (v5).

This file mirrors `config_parquet_v3.py` but sets v5-specific output
defaults so builds go into a distinct folder and cannot clobber v3
artefacts by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Any

# Re-use the ISO 1/3-octave band definition from v2 (no change)
from src.db.parquet.config_parquet_v2 import OCTAVE_BANDS


# ---------------------------------------------------------------------------
# Sub-configs (copied from v3)
# ---------------------------------------------------------------------------


@dataclass
class TargetConfig:
    name: str = "target_pgv_z_mms"
    use_axis: str = "z"


@dataclass
class FOProcessingConfig:
    """FO signal conditioning — identical to v1/v2."""

    bandpass_freqmin: float = 1.0
    bandpass_freqmax: float = 100.0
    bandpass_corners: int = 5
    welch_nperseg: int = 1024
    welch_nfft: int = 10000


@dataclass
class PerLineFOConfig:
    line_centers: Dict[str, int] = field(
        default_factory=lambda: {
            "line_A": 1184,
            "line_B": 1192,
            "line_C": 1194,
            "line_D": 1196,
            "line_E": 1204,
        }
    )
    line_half_window: int = 5


@dataclass
class TrackGeometryConfig:
    track_separation_m: float = 4.0
    track_1_to_fo_m: float = 4.0
    track_2_to_fo_m: float = 8.0

    sensor_distance_override_m: Dict[str, float] = field(
        default_factory=lambda: {
            "MP1": 16.0,
            "MP2": 23.0,
            "MP4": 2.5,
            "MP7": 4.0,
            "MP8": 4.0,
            "MP9": 4.0,
            "MP10": 8.0,
            "MP12": 5.0,
            "MP13": 5.0,
        }
    )


@dataclass
class SensorLineMapConfig:
    sensor_line_map: Dict[str, str] = field(
        default_factory=lambda: {
            "MP1": "line_C",
            "MP2": "line_C",
            "MP4": "line_C",
            "MP7": "line_B",
            "MP8": "line_C",
            "MP9": "line_D",
            "MP10": "line_C",
            "MP12": "line_A",
            "MP13": "line_E",
        }
    )


@dataclass
class FeatureFamilyConfig:
    fo_time_domain: bool = True
    fo_spectral_octave_bands: bool = True
    include_train_type_family: bool = True


@dataclass
class InclusionRulesConfig:
    require_fo_data: bool = True
    require_z_axis: bool = True
    require_finite_target: bool = True
    require_side_of_track: int = -1


@dataclass
class OutputConfig:
    # Distinct version name and file layout for v5
    version_name: str = "parquet_v005"
    parquet_filename: str = "dataset.parquet"
    config_snapshot_filename: str = "parquet_config_snapshot.json"
    summary_filename: str = "build_summary.json"
    log_filename: str = "build_log.txt"
    parquet_engine: str = "pyarrow"


@dataclass
class ParquetV5Config:
    """Top-level configuration for Parquet v5 build.

    Defaults are conservatively set so that v5 output goes to a new
    folder and will not overwrite v3 outputs.
    """

    input_netcdf_folders: List[str] = field(
        default_factory=lambda: [
            r"P:\11210978-erju-ai\holten_db\netcdf_20260409_134732",
            r"P:\11210978-erju-ai\holten_db\netcdf_20260409_163947",
        ]
    )
    # Use an explicit new output root for v5 to avoid clobbering v3
    output_root_folder: str = r"P:\11210978-erju-ai\holten_parquet_v005"
    netcdf_glob: str = "**/*.nc"

    row_unit: str = "one_row_per_event_sensor"

    target: TargetConfig = field(default_factory=TargetConfig)
    fo_processing: FOProcessingConfig = field(default_factory=FOProcessingConfig)
    per_line_fo: PerLineFOConfig = field(default_factory=PerLineFOConfig)
    track_geometry: TrackGeometryConfig = field(default_factory=TrackGeometryConfig)
    sensor_line_map: SensorLineMapConfig = field(default_factory=SensorLineMapConfig)
    feature_families: FeatureFamilyConfig = field(default_factory=FeatureFamilyConfig)
    inclusion_rules: InclusionRulesConfig = field(default_factory=InclusionRulesConfig)

    octave_bands: List[Dict[str, float]] = field(default_factory=lambda: OCTAVE_BANDS)

    channel_reductions: List[str] = field(
        default_factory=lambda: ["mean", "max", "std"]
    )

    sort_rows_by: List[str] = field(default_factory=lambda: ["event_id", "sensor_id"])

    max_files: int = 0

    exclude_sensor_ids: List[str] = field(
        default_factory=lambda: ["MP14", "MP15", "MP16", "MP17", "MP18", "MP19"]
    )

    output: OutputConfig = field(default_factory=OutputConfig)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def input_folder_paths(self) -> List[Path]:
        return [Path(p) for p in self.input_netcdf_folders]

    def output_root_path(self) -> Path:
        return Path(self.output_root_folder)


CONFIG = ParquetV5Config()
