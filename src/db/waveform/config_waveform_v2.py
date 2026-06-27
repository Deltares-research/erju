"""Configuration for the multi-channel FO waveform dataset (waveform v2).

v2 vs v1
--------
v1 stored a single channel (1194) per event: shape (N_events, T).
v2 stores a local channel window 1184-1204 (21 channels) per event:
shape (N_events, 21, T).  This exposes the space-time structure of the train
passage (the inclined wavefront across channels) to a 2D CNN.

All other processing matches v1: 1-100 Hz bandpass, 1000->250 Hz resample,
30 s energy-centred crop/pad, NO per-event normalization.  The crop window is
computed once from the centre channel (1194) and applied identically to all
21 channels so the shared time axis stays aligned.

Input paths come from config_parquet_v2 (single source of truth).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from src.db.parquet.config_parquet_v2 import CONFIG as PARQUET_V2_CFG


@dataclass
class SignalConfig:
    """FO signal preprocessing (identical to waveform v1)."""

    center_channel: int = 1194           # reference channel for the crop window
    channel_lo: int = 1184               # inclusive lower channel ID
    channel_hi: int = 1204               # inclusive upper channel ID  -> 21 channels
    channel_stride: int = 1              # take every Nth channel (1 = all)
    channel_spacing_m: float = 1.0       # FO channel pitch (consecutive IDs are 1 m apart)
    gauge_length_m: float = 10.0         # DAS spatial averaging window (NOT the pitch)
    bandpass_freqmin: float = 1.0
    bandpass_freqmax: float = 100.0
    bandpass_corners: int = 5
    native_fs_hz: float = 1000.0
    target_fs_hz: float = 250.0

    @property
    def n_channels(self) -> int:
        return (self.channel_hi - self.channel_lo) // self.channel_stride + 1

    @property
    def effective_pitch_m(self) -> float:
        """Spacing between selected channels [m] (pitch x stride)."""
        return self.channel_spacing_m * self.channel_stride

    @property
    def aperture_m(self) -> float:
        """Total spatial aperture of the channel window [m]."""
        return (self.n_channels - 1) * self.effective_pitch_m


@dataclass
class WindowConfig:
    """Fixed-length crop/pad settings (identical to v1)."""

    fixed_length_s: float = 30.0
    crop_center_method: str = "energy_centroid"
    envelope_smooth_s: float = 0.5
    pad_value: float = 0.0


@dataclass
class TargetConfig:
    pgv_col: str = "target_pgv_z_mms"
    event_col: str = "event_id"
    sensor_col: str = "sensor_id"
    distance_col: str = "effective_distance_to_active_track_m"
    raw_distance_col: str = "acc_distance_to_track_m"
    side_col: str = "acc_side_of_track"
    metadata_cols: List[str] = field(
        default_factory=lambda: [
            "train_type",
            "train_type_code",
            "train_speed_kmh",
            "track_number",
        ]
    )
    r0_m: float = 10.0


@dataclass
class OutputConfig:
    version_name: str = "holten_waveform_v002"
    waveforms_filename: str = "waveforms.npy"
    event_index_filename: str = "event_index.parquet"
    build_config_filename: str = "build_config.json"
    build_log_filename: str = "build_log.txt"
    plots_subfolder: str = "plots"
    output_root_folder: str = r"P:\11210978-erju-ai\holten_waveform"


@dataclass
class WaveformV2Config:
    signal: SignalConfig = field(default_factory=SignalConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    exclude_sensor_ids: List[str] = field(
        default_factory=lambda: list(PARQUET_V2_CFG.exclude_sensor_ids)
    )

    max_files: int = 0
    n_crop_diagnostic_plots: int = 6

    @property
    def fixed_length_samples(self) -> int:
        return int(round(self.window.fixed_length_s * self.signal.target_fs_hz))

    def input_folder_paths(self) -> List[Path]:
        return PARQUET_V2_CFG.input_folder_paths()

    def netcdf_glob(self) -> str:
        return PARQUET_V2_CFG.netcdf_glob

    def parquet_root(self) -> Path:
        return Path(PARQUET_V2_CFG.output_root_folder)

    def output_root_path(self) -> Path:
        return Path(self.output.output_root_folder)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


CONFIG = WaveformV2Config()
