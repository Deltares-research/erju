"""
Configuration for Parquet dataset building (v1).

This config is intentionally explicit so every build is reproducible and easy to audit.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Tuple, Dict, Any


@dataclass
class TargetConfig:
    """Target definition settings.

    v1: the /acc/<SENSOR_ID>/acceleration_mps2 variable stores pre-processed
    velocity in mm/s.  The target is therefore the peak absolute value of
    the z-channel — no integration is performed.
    """

    name: str = "target_pgv_z_mms"
    use_axis: str = "z"


@dataclass
class FOProcessingConfig:
    """FO signal transformation settings.

    The raw optical-phase data stored in /fo/strain is converted to strain
    using the same formula as OptasenseFOdata.from_opticalphase_to_strain(),
    reading fibre_refractive_index and gauge_length directly from /meta_fo.
    A Butterworth bandpass filter is then applied before feature extraction.

    Welch PSD settings match the approach used in compare_data.py via
    SignalProcessingTools (Hamming window, nperseg=1024, nfft=10000,
    scaling='density').
    """

    bandpass_freqmin: float = 1.0  # Hz
    bandpass_freqmax: float = 100.0  # Hz
    bandpass_corners: int = 5

    # Welch PSD parameters (used for spectral band features)
    welch_nperseg: int = 1024  # window length in samples (Hamming)
    welch_nfft: int = 10000  # FFT points (zero-padded); matches compare_data.py


@dataclass
class FeatureFamilyConfig:
    """Feature family switches."""

    fo_time_domain: bool = True
    fo_spectral_bands: bool = True
    include_train_type_code: bool = True


@dataclass
class InclusionRulesConfig:
    """Rules to include rows in Parquet output."""

    require_fo_data: bool = True
    require_z_axis: bool = True
    require_finite_target: bool = True


@dataclass
class OutputConfig:
    """Output naming and engine settings."""

    version_name: str = "parquet_v001"
    parquet_filename: str = "dataset.parquet"
    config_snapshot_filename: str = "parquet_config_snapshot.json"
    summary_filename: str = "build_summary.json"
    log_filename: str = "build_log.txt"
    parquet_engine: str = "pyarrow"


@dataclass
class ParquetV1Config:
    """Top-level config for Parquet v1 build."""

    # IO
    input_netcdf_folders: List[str] = field(
        default_factory=lambda: [
            r"P:\11210978-erju-ai\holten_db\netcdf_20260405_001947_10mGL_3000channels",
            r"P:\11210978-erju-ai\holten_db\netcdf_20260405_111215_10mGL_6000channels",
        ]
    )
    output_root_folder: str = r"P:\11210978-erju-ai\holten_parquet"
    netcdf_glob: str = "**/*.nc"

    # Row unit
    row_unit: str = "one_row_per_event_sensor"

    # Feature design
    target: TargetConfig = field(default_factory=TargetConfig)
    fo_processing: FOProcessingConfig = field(default_factory=FOProcessingConfig)
    feature_families: FeatureFamilyConfig = field(default_factory=FeatureFamilyConfig)
    inclusion_rules: InclusionRulesConfig = field(default_factory=InclusionRulesConfig)

    # FO spectral bins in Hz: [low, high)
    frequency_bins_hz: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.0, 5.0),
            (5.0, 10.0),
            (10.0, 15.0),
            (15.0, 20.0),
            (20.0, 25.0),
            (25.0, 30.0),
            (30.0, 40.0),
            (40.0, 50.0),
            (50.0, 60.0),
            (60.0, 70.0),
            (70.0, 80.0),
            (80.0, 90.0),
            (90.0, 100.0),
        ]
    )

    # Channel-level reduction methods for fixed-size FO features
    channel_reductions: List[str] = field(
        default_factory=lambda: ["mean", "max", "std"]
    )

    # Optional deterministic sorting
    sort_rows_by: List[str] = field(default_factory=lambda: ["event_id", "sensor_id"])

    # Debug: limit to first N files (0 = no limit).
    # Set to a small number (e.g. 5) for quick smoke-tests.
    max_files: int = 0

    output: OutputConfig = field(default_factory=OutputConfig)

    def as_dict(self) -> Dict[str, Any]:
        """Return config as plain nested dictionary for JSON snapshot."""
        return asdict(self)

    def input_folder_paths(self) -> List[Path]:
        return [Path(p) for p in self.input_netcdf_folders]

    def output_root_path(self) -> Path:
        return Path(self.output_root_folder)


CONFIG = ParquetV1Config()
