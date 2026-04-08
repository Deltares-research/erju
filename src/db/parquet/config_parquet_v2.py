"""
Configuration for Parquet dataset building (v2).

v2 change vs v1:
  - FO spectral features use 1/3-octave bands (ISO 18405:2017 base-2 formula)
    starting from 1 Hz, instead of the uniform 5-Hz linear bins used in v1.
  - This gives 21 bands covering ~0.89 Hz – 100 Hz with logarithmically
    increasing bandwidth, better matching structural vibration physics.
  - All other settings (FO processing, target definition, exclusions, etc.)
    are identical to v1.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


# ---------------------------------------------------------------------------
# 1/3-octave band definition  (ISO 18405:2017, base-2)
# ---------------------------------------------------------------------------
# Nominal centre frequencies from 1 Hz up to and including 100 Hz.
# Edges: f_lower = f_c / 2^(1/6),  f_upper = f_c * 2^(1/6)
# The upper edge of the last band is clipped to 100 Hz so we never request
# features above the bandpass filter cut-off.

_OCT_NOMINALS_HZ: List[float] = [
    1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3,
    8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0,
]

_FACTOR = 2 ** (1 / 6)  # edge = fc * 2^(+/-1/6)

OCTAVE_BANDS: List[Dict[str, float]] = [
    {
        "nominal_hz": fc,
        "lower_hz": fc / _FACTOR,
        "upper_hz": min(fc * _FACTOR, 100.0),
    }
    for fc in _OCT_NOMINALS_HZ
]


def _oct_feature_name(nominal_hz: float) -> str:
    """Column name prefix for a 1/3-octave band, e.g. 'fo_oct_001hz'."""
    if nominal_hz < 10:
        return f"fo_oct_{nominal_hz:.2f}hz".replace(".", "_")
    return f"fo_oct_{int(nominal_hz):03d}hz"


# ---------------------------------------------------------------------------
# Dataclasses (mirroring v1 structure for consistency)
# ---------------------------------------------------------------------------


@dataclass
class TargetConfig:
    name: str = "target_pgv_z_mms"
    use_axis: str = "z"


@dataclass
class FOProcessingConfig:
    bandpass_freqmin: float = 1.0
    bandpass_freqmax: float = 100.0
    bandpass_corners: int = 5
    welch_nperseg: int = 1024
    welch_nfft: int = 10000


@dataclass
class FeatureFamilyConfig:
    fo_time_domain: bool = True
    fo_spectral_octave_bands: bool = True
    include_train_type_code: bool = True


@dataclass
class InclusionRulesConfig:
    require_fo_data: bool = True
    require_z_axis: bool = True
    require_finite_target: bool = True


@dataclass
class OutputConfig:
    version_name: str = "parquet_v002"
    parquet_filename: str = "dataset.parquet"
    config_snapshot_filename: str = "parquet_config_snapshot.json"
    summary_filename: str = "build_summary.json"
    log_filename: str = "build_log.txt"
    parquet_engine: str = "pyarrow"


@dataclass
class ParquetV2Config:
    """Top-level config for Parquet v2 build."""

    # IO — same NetCDF sources as v1
    input_netcdf_folders: List[str] = field(
        default_factory=lambda: [
            r"P:\11210978-erju-ai\holten_db\netcdf_20260406_194947_10mGL_3000channels",
            r"P:\11210978-erju-ai\holten_db\netcdf_20260407_105509_10mGL_6000channels",
        ]
    )
    output_root_folder: str = r"P:\11210978-erju-ai\holten_parquet"
    netcdf_glob: str = "**/*.nc"

    row_unit: str = "one_row_per_event_sensor"

    target: TargetConfig = field(default_factory=TargetConfig)
    fo_processing: FOProcessingConfig = field(default_factory=FOProcessingConfig)
    feature_families: FeatureFamilyConfig = field(default_factory=FeatureFamilyConfig)
    inclusion_rules: InclusionRulesConfig = field(default_factory=InclusionRulesConfig)

    # 1/3-octave bands — auto-generated from OCTAVE_BANDS above
    octave_bands: List[Dict[str, float]] = field(
        default_factory=lambda: OCTAVE_BANDS
    )

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


CONFIG = ParquetV2Config()
