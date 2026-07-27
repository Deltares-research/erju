"""Configuration for Parquet dataset building (v5).

v5 changes vs v2:
  - Input: patched, fully-complete NetCDF databases (netcdf_20260409_*).
  - FO features computed per track line (A/B/C/D/E) using ±5-channel sub-windows
    around each line's center channel, instead of the full 51-channel average.
  - Each row receives features from all 5 lines plus signed longitudinal offsets
    (in metres) from the sensor's own line to every other line.
  - Effective distance to the active track (acc_distance_to_track_m + 4.0 m when
    track_number == 2) is added to encode which physical rail was active.
  - Per-sensor distances corrected from updated geometry (FO at Y=0, track 1 at
    Y=4.0 m, track 2 at Y=8.0 m) — overrides the rounded values in the NetCDF.
  - Train type mapped to 8 physics-based families (GO/ICM/ICR/SNG/SPR/DDZ/
    Locomotive/Other) instead of alphabetical integer code.
  - Only sensors with acc_side_of_track == -1 are included:
    MP1, MP2, MP4, MP7, MP8, MP9, MP10, MP12, MP13.
  - MP14-MP19 always excluded (in-track sensors, acceleration in g).

Approximate feature count per row:
  5 lines × 21 octave bands × 3 reductions   = 315  (spectral)
  5 lines × 3 td stats  × 3 reductions        =  45  (time-domain)
  5 signed longitudinal offset features        =   5
  acc_distance_to_track_m                      =   1
  effective_distance_to_active_track_m         =   1
  sensor_line_code                             =   1
  train_type_family_code, train_speed_kmh,
  track_number, acc_side_of_track              =   4
  ─────────────────────────────────────────────────
  Total numeric features                       ~ 372
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Any

# Re-use the ISO 1/3-octave band definition from v2 (no change)
from src.db.parquet.config_parquet_v2 import OCTAVE_BANDS

# ---------------------------------------------------------------------------
# Sub-configs
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
    """Per-line FO sub-window definition.

    Each of the 5 track lines has a center FO channel.  Features are computed
    over the ±line_half_window channels around that center (11 channels per
    line).  All 5 windows fall within the stored ch. 1169–1219 range.
    """

    # Center channel for each track line (from holten.json / line_geometry)
    line_centers: Dict[str, int] = field(
        default_factory=lambda: {
            "line_A": 1184,
            "line_B": 1192,
            "line_C": 1194,
            "line_D": 1196,
            "line_E": 1204,
        }
    )
    # Half-window in channels; ±5 → 11 channels per line
    line_half_window: int = 5


@dataclass
class TrackGeometryConfig:
    """Track separation used to compute effective sensor distance.

    acc_distance_to_track_m is always measured from track 1 (the closer rail).
    When a train runs on track 2, the effective distance increases by
    track_separation_m.

    Holten geometry (corrected):
      Coordinate system: FO at Y=0, positive Y toward tracks.
      FO cable  → track 1 centre : 4.0 m  (track 1 at Y=4.0)
      FO cable  → track 2 centre : 8.0 m  (track 2 at Y=8.0)
      track 1   → track 2 separation : 4.0 m

    Per-sensor perpendicular distances to track 1 (corrected from geometry):
      MP4  : 2.5 m  (sensor Y=1.5)
      MP7  : 4.0 m  (sensor Y=0.0)
      MP8  : 4.0 m  (sensor Y=0.0)
      MP9  : 4.0 m  (sensor Y=0.0)
      MP10 : 8.0 m  (sensor Y=-4.0)
      MP12 : 5.0 m  (sensor Y=-1.0)
      MP13 : 5.0 m  (sensor Y=-1.0)
      MP1  : 16.0 m (sensor Y=-12.0)
      MP2  : 23.0 m (sensor Y=-19.0)

    sensor_distance_override_m is used at parquet-build time to override
    the distances stored in the (already-built, fixed) NetCDF files.
    """

    track_separation_m: float = 4.0
    track_1_to_fo_m: float = 4.0  # informational only
    track_2_to_fo_m: float = 8.0  # informational only

    # Corrected per-sensor distance to track 1 (overrides NetCDF values).
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
    """Mapping: sensor_id → track line for side=-1 sensors only.

    Side=+1 sensors (MP3, MP5, MP6) and in-track sensors (MP14-MP19)
    are absent because they are excluded from v3 output.
    """

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
    # Only include sensors on this side of the track (-1 = same side as FO).
    # Set to None to include all sides.
    require_side_of_track: int = -1


@dataclass
class OutputConfig:
    version_name: str = "parquet_v005"
    parquet_filename: str = "dataset.parquet"
    config_snapshot_filename: str = "parquet_config_snapshot.json"
    summary_filename: str = "build_summary.json"
    log_filename: str = "build_log.txt"
    parquet_engine: str = "pyarrow"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class ParquetV5Config:
    """Top-level configuration for Parquet v5 build."""

    # IO — patched, fully-complete NetCDF databases (bug-fixed Sep 2 events)
    input_netcdf_folders: List[str] = field(
        default_factory=lambda: [
            r"P:\11210978-erju-ai\holten_db\netcdf_20260409_134732",
            r"P:\11210978-erju-ai\holten_db\netcdf_20260409_163947",
        ]
    )
    output_root_folder: str = r"P:\11210978-erju-ai\holten_parquet"
    netcdf_glob: str = "**/*.nc"

    row_unit: str = "one_row_per_event_sensor"

    target: TargetConfig = field(default_factory=TargetConfig)
    fo_processing: FOProcessingConfig = field(default_factory=FOProcessingConfig)
    per_line_fo: PerLineFOConfig = field(default_factory=PerLineFOConfig)
    track_geometry: TrackGeometryConfig = field(default_factory=TrackGeometryConfig)
    sensor_line_map: SensorLineMapConfig = field(default_factory=SensorLineMapConfig)
    feature_families: FeatureFamilyConfig = field(default_factory=FeatureFamilyConfig)
    inclusion_rules: InclusionRulesConfig = field(default_factory=InclusionRulesConfig)

    # 1/3-octave bands — same definition as v2
    octave_bands: List[Dict[str, float]] = field(default_factory=lambda: OCTAVE_BANDS)

    channel_reductions: List[str] = field(
        default_factory=lambda: ["mean", "max", "std"]
    )

    sort_rows_by: List[str] = field(default_factory=lambda: ["event_id", "sensor_id"])

    # Set > 0 to limit files for smoke-testing (0 = no limit)
    max_files: int = 0

    # In-track sensors with incompatible units — excluded unconditionally
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
