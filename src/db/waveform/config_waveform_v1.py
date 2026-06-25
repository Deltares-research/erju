"""Configuration for the single-channel FO waveform dataset (waveform v1).

This is the first beyond-tabular dataset: one raw FO waveform per event taken
from a single channel (1194), preprocessed identically to the tabular feature
pipeline (1-100 Hz bandpass), downsampled to 250 Hz, and cropped/padded to a
fixed length around the channel-1194 energy centre.

Input paths are *not* duplicated here — the NetCDF source folders come from
``config_parquet_v2`` so there is a single source of truth.  Only
waveform-specific settings are defined in this file.

One sample = one event.  Sensor-level targets (PGV_z, distance) live in the
Parquet v2 dataset and are joined at training time by ``event_id``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

# Single source of truth for the NetCDF input folders and event exclusions.
from src.db.parquet.config_parquet_v2 import CONFIG as PARQUET_V2_CFG


@dataclass
class SignalConfig:
    """FO signal preprocessing for the waveform model input.

    The stored ``/fo/strain`` is full-band.  We apply the same Butterworth
    bandpass as the tabular pipeline, then resample 1000 Hz -> 250 Hz.  This
    approximately preserves the target 1-100 Hz band (Nyquist at 250 Hz is
    125 Hz) while reducing the sequence length 4x.  No per-event amplitude
    normalization is applied — amplitude is physically meaningful for PGV.
    """

    target_channel: int = 1194           # absolute FO channel ID to extract
    bandpass_freqmin: float = 1.0        # Hz, same as tabular pipeline
    bandpass_freqmax: float = 100.0      # Hz
    bandpass_corners: int = 5            # Butterworth order
    native_fs_hz: float = 1000.0         # expected source sampling rate
    target_fs_hz: float = 250.0          # resample target (4x decimation)


@dataclass
class WindowConfig:
    """Fixed-length crop/pad settings.

    The crop is centred on the channel-1194 energy centroid (FO-only; no
    accelerometer information is used).  Events longer than ``fixed_length_s``
    are cropped, shorter events are zero-padded.  Diagnostics record exactly
    what happened to each event.
    """

    fixed_length_s: float = 30.0         # v1 baseline window length [s]
    # crop centre method: "energy_centroid" (robust) or "peak_envelope"
    crop_center_method: str = "energy_centroid"
    # Envelope smoothing window for the energy measure [s].  A short moving
    # RMS makes the centroid robust to single-sample spikes.
    envelope_smooth_s: float = 0.5
    pad_value: float = 0.0               # zero-pad short events

    @property
    def fixed_length_samples(self) -> int:
        # Filled in lazily by the builder using target_fs_hz; kept here as a
        # convenience when target_fs_hz is the default 250 Hz.
        return int(round(self.fixed_length_s * 250.0))


@dataclass
class TargetConfig:
    """Target / join settings (sensor-level rows come from Parquet v2)."""

    pgv_col: str = "target_pgv_z_mms"
    event_col: str = "event_id"
    sensor_col: str = "sensor_id"
    # Corrected distance (primary).  Raw distance kept for diagnostics only.
    distance_col: str = "effective_distance_to_active_track_m"
    raw_distance_col: str = "acc_distance_to_track_m"
    side_col: str = "acc_side_of_track"
    # Metadata columns carried into the event index for the model.
    metadata_cols: List[str] = field(
        default_factory=lambda: [
            "train_type",
            "train_type_code",
            "train_speed_kmh",
            "track_number",
        ]
    )
    r0_m: float = 10.0                   # reference distance for log(r/r0)


@dataclass
class OutputConfig:
    version_name: str = "holten_waveform_v001"
    waveforms_filename: str = "waveforms.npy"
    event_index_filename: str = "event_index.parquet"
    build_config_filename: str = "build_config.json"
    build_log_filename: str = "build_log.txt"
    plots_subfolder: str = "plots"
    output_root_folder: str = r"P:\11210978-erju-ai\holten_waveform"


@dataclass
class WaveformV1Config:
    """Top-level config for the waveform v1 build."""

    signal: SignalConfig = field(default_factory=SignalConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Sensors to exclude (in-track MP14-MP19), inherited from Parquet v2.
    exclude_sensor_ids: List[str] = field(
        default_factory=lambda: list(PARQUET_V2_CFG.exclude_sensor_ids)
    )

    # Number of NetCDF files to process (0 = all). Smoke-test override.
    max_files: int = 0

    # Number of diagnostic crop plots to draw (short/median/long/p95/hi/lo PGV).
    n_crop_diagnostic_plots: int = 6

    # ---- derived helpers -------------------------------------------------
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


CONFIG = WaveformV1Config()
