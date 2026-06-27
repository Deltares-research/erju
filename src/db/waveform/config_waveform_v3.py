"""Configuration for the wide-aperture multi-channel FO waveform dataset (v3).

v3 vs v2
--------
v2: 21 channels (1184-1204), 20 m aperture at 1 m pitch.
v3: wide aperture using the full stored channel block 1165-1215 (51 channels,
    50 m), with an optional channel stride to reduce gauge-length redundancy.

Motivation: the 10 m gauge length spatially smooths the signal, so a 20 m
window holds only ~2 independent gauge windows.  A 50 m aperture gives ~5
independent windows and captures the along-track amplitude-decay profile.

Variants are produced by overriding ``signal.channel_stride`` and
``output.version_name`` at run time:
    stride 1 -> 51 channels @ 1 m  (holten_waveform_v003_ch51)
    stride 5 -> 11 channels @ 5 m  (holten_waveform_v003_ch11)

Reuses the v2 processing engine (waveform_v2_utils).  Crop window is computed
once from the centre channel and applied to all channels; no per-event
normalization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from src.db.parquet.config_parquet_v2 import CONFIG as PARQUET_V2_CFG


@dataclass
class SignalConfig:
    center_channel: int = 1194
    channel_lo: int = 1165               # full stored block lower bound
    channel_hi: int = 1215               # full stored block upper bound -> 51 ch @1m
    channel_stride: int = 1              # 1 -> 51 ch; 5 -> 11 ch
    channel_spacing_m: float = 1.0
    gauge_length_m: float = 10.0
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
        return self.channel_spacing_m * self.channel_stride

    @property
    def aperture_m(self) -> float:
        return (self.n_channels - 1) * self.effective_pitch_m


@dataclass
class WindowConfig:
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
            "train_type", "train_type_code", "train_speed_kmh", "track_number",
        ]
    )
    r0_m: float = 10.0


@dataclass
class OutputConfig:
    version_name: str = "holten_waveform_v003"
    waveforms_filename: str = "waveforms.npy"
    event_index_filename: str = "event_index.parquet"
    build_config_filename: str = "build_config.json"
    build_log_filename: str = "build_log.txt"
    plots_subfolder: str = "plots"
    output_root_folder: str = r"P:\11210978-erju-ai\holten_waveform"


@dataclass
class WaveformV3Config:
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


CONFIG = WaveformV3Config()
