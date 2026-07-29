"""Helpers for building Parquet v1 from event-level NetCDF files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from contextlib import contextmanager

try:
    import netCDF4 as nc
except ModuleNotFoundError:
    nc = None  # optional; not needed by assign_train_type_family
import numpy as np
import pandas as pd
from scipy.signal import iirfilter, zpk2sos, sosfiltfilt, welch


@dataclass
class SkipRecord:
    level: str
    event_id: str
    sensor_id: str | None
    reason: str
    file_path: str


@contextmanager
def open_netcdf_event(file_path: Path):
    """Context manager to open one NetCDF event file safely."""
    with nc.Dataset(file_path, "r") as dataset:
        yield dataset


def _to_python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            pass
    return value


def _safe_float(value: Any, default: float = np.nan) -> float:
    value = _to_python_scalar(value)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = -1) -> int:
    value = _to_python_scalar(value)
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_scalar_var(group: nc.Group, var_name: str, default: Any = None) -> Any:
    if group is None or var_name not in group.variables:
        return default
    try:
        return _to_python_scalar(group.variables[var_name][()])
    except Exception:
        return default


def _read_axis_mask_for_sensor(
    dataset: nc.Dataset, sensor_id: str
) -> np.ndarray | None:
    # Preferred location: /acc/<SENSOR_ID>/axis_mask
    try:
        acc_root = dataset.groups.get("acc")
        if acc_root is not None and sensor_id in acc_root.groups:
            sensor_group = acc_root.groups[sensor_id]
            if "axis_mask" in sensor_group.variables:
                mask = np.asarray(
                    sensor_group.variables["axis_mask"][:], dtype=np.int32
                )
                if mask.size >= 3:
                    return mask[:3]
    except Exception:
        pass

    # Fallback: /geometry_acc/axis_mask by /geometry_acc/acc_sensor_id index
    try:
        geometry = dataset.groups.get("geometry_acc") or dataset.groups.get("geometry")
        if (
            geometry is not None
            and "acc_sensor_id" in geometry.variables
            and "axis_mask" in geometry.variables
        ):
            sensor_ids = [
                str(_to_python_scalar(v))
                for v in geometry.variables["acc_sensor_id"][:]
            ]
            if sensor_id in sensor_ids:
                idx = sensor_ids.index(sensor_id)
                mask = np.asarray(
                    geometry.variables["axis_mask"][idx, :], dtype=np.int32
                )
                if mask.size >= 3:
                    return mask[:3]
    except Exception:
        pass

    return None


def compute_pgv_z_mms(velocity_z_mms: np.ndarray) -> float:
    """Return peak ground velocity from the z-channel of the stored velocity trace.

    The /acc/<SENSOR_ID>/velocity_mms variable stores pre-processed velocity
    in mm/s.  The target is simply max(abs(signal_z)).
    """
    v = np.asarray(velocity_z_mms, dtype=np.float64)
    if v.size == 0:
        raise ValueError("Empty velocity trace for PGV computation")
    return float(np.max(np.abs(v)))


def compute_fo_time_domain_features(strain_ch: np.ndarray) -> Dict[str, float]:
    x = np.asarray(strain_ch, dtype=np.float64)
    return {
        "fo_td_rms": float(np.sqrt(np.mean(x * x))),
        "fo_td_max_abs": float(np.max(np.abs(x))),
        "fo_td_std": float(np.std(x)),
    }


def compute_fo_spectral_band_features(
    strain_ch: np.ndarray,
    fs_hz: float,
    frequency_bins_hz: List[Tuple[float, float]],
    nperseg: int,
    nfft: int,
) -> Dict[str, float]:
    """Band-integrated Welch PSD power per frequency bin.

    Uses scipy.signal.welch with a Hamming window, matching the approach in
    compare_data.py (SignalProcessingTools.psd with window=HAMMING,
    window_size=1024, nb_points=10000, scaling='density').

    Feature value = integral of PSD over the band = sum(Pxx[mask] * df),
    in units of strain²/Hz × Hz = strain².  Comparable across events
    regardless of signal length.
    """
    x = np.asarray(strain_ch, dtype=np.float64)
    if x.size < 2 or x.size < nperseg:
        return {
            _band_feature_name(low, high): np.nan for low, high in frequency_bins_hz
        }

    freqs, pxx = welch(
        x,
        fs=fs_hz,
        window="hamming",
        nperseg=nperseg,
        nfft=nfft,
        scaling="density",
        detrend="linear",
    )
    df = freqs[1] - freqs[0]  # frequency resolution

    out = {}
    for low, high in frequency_bins_hz:
        mask = (freqs >= low) & (freqs < high)
        band_name = _band_feature_name(low, high)
        out[band_name] = float(np.sum(pxx[mask]) * df) if np.any(mask) else 0.0
    return out


def _band_feature_name(low: float, high: float) -> str:
    return f"fo_band_{int(low):03d}_{int(high):03d}_energy"


def reduce_channel_features(
    per_channel_features: List[Dict[str, float]], reductions: List[str]
) -> Dict[str, float]:
    if len(per_channel_features) == 0:
        return {}

    feature_names = sorted(per_channel_features[0].keys())
    reduced = {}

    for feat in feature_names:
        values = np.array(
            [f.get(feat, np.nan) for f in per_channel_features], dtype=np.float64
        )

        for red in reductions:
            col = f"{feat}_{red}"
            if red == "mean":
                reduced[col] = float(np.nanmean(values))
            elif red == "max":
                reduced[col] = float(np.nanmax(values))
            elif red == "std":
                reduced[col] = float(np.nanstd(values))
            else:
                raise ValueError(f"Unsupported reduction: {red}")

    return reduced


# ---------------------------------------------------------------------------
# FO signal processing helpers
# ---------------------------------------------------------------------------


def _bandpass_fo(
    data: np.ndarray,
    freqmin: float,
    freqmax: float,
    fs: float,
    corners: int,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to all channels at once.

    Uses sosfiltfilt (forward-backward SOS filter) which is the same approach
    as SignalProcessingTools.TimeSignalProcessing.filter() and
    OptasenseFOdata.bandpass(zerophase=True).  Operates on the full 2-D matrix
    (time × channels) in one call — no per-channel loop needed.
    """
    fe = 0.5 * fs
    low = freqmin / fe
    high = freqmax / fe
    z, p, k = iirfilter(
        corners, [low, high], btype="band", ftype="butter", output="zpk"
    )
    sos = zpk2sos(z, p, k)
    # sosfiltfilt applies the filter forward then backward (zero-phase).
    # axis=0 filters along the time dimension for every channel simultaneously.
    return sosfiltfilt(sos, data, axis=0)


def process_fo_event_signal(
    dataset: nc.Dataset,
    fo_config: Any,
) -> np.ndarray | None:
    """Read FO strain from /fo/strain and return bandpass-filtered strain.

    What is stored in /fo/strain (written by fo_utils.extract_fo_event_data):
      raw int16 counts → demean → Tukey taper (α=0.1) → phase-to-strain conversion
      i.e. the data is ALREADY in strain units (ε), NOT raw optical-phase counts.

    Therefore this function only needs to:
      1. Read the strain array (time × channels) from /fo/strain.
      2. Apply the bandpass filter to remove out-of-band noise.

    Demean and Tukey are intentionally skipped here — both were applied
    during the NetCDF write step and the signal is already conditioned.

    Returns the bandpass-filtered 2-D strain array (time, channels), or None
    when the FO group is absent or the data shape/fs is invalid.
    """
    if "fo" not in dataset.groups:
        return None

    fo_grp = dataset.groups["fo"]
    if "strain" not in fo_grp.variables or "fs_hz" not in fo_grp.variables:
        return None

    strain = np.asarray(fo_grp.variables["strain"][:], dtype=np.float64)
    fs_hz = _safe_float(fo_grp.variables["fs_hz"][()], default=np.nan)

    if (
        strain.ndim != 2
        or strain.shape[0] < 2
        or strain.shape[1] < 1
        or not np.isfinite(fs_hz)
        or fs_hz <= 0
    ):
        return None

    # Bandpass filter the strain signal (1–100 Hz by default).
    return _bandpass_fo(
        data=strain,
        freqmin=fo_config.bandpass_freqmin,
        freqmax=fo_config.bandpass_freqmax,
        fs=fs_hz,
        corners=fo_config.bandpass_corners,
    )


def compute_event_fo_features(
    dataset: nc.Dataset,
    fo_config: Any,
    frequency_bins_hz: List[Tuple[float, float]],
    reductions: List[str],
    include_time_domain: bool,
    include_spectral: bool,
) -> Dict[str, float] | None:
    """Compute event-level FO features from the processed FO signal.

    Time-domain features (per channel → reduced across channels):
      fo_td_rms, fo_td_max_abs, fo_td_std, fo_td_energy

    Spectral features (per channel → reduced across channels):
      fo_band_<low>_<high>_energy: band-integrated Welch PSD power (strain²)
      for each bin in frequency_bins_hz.

    Reductions applied: mean, max, std across all FO channels.
    """
    processed = process_fo_event_signal(dataset=dataset, fo_config=fo_config)
    if processed is None:
        return None

    # fs_hz is guaranteed present after process_fo_event_signal succeeded.
    fs_hz = _safe_float(dataset.groups["fo"].variables["fs_hz"][()], default=np.nan)

    per_ch = []
    for ch_idx in range(processed.shape[1]):
        ch_signal = processed[:, ch_idx]
        ch_feats: Dict[str, float] = {}

        if include_time_domain:
            ch_feats.update(compute_fo_time_domain_features(ch_signal))

        if include_spectral:
            ch_feats.update(
                compute_fo_spectral_band_features(
                    strain_ch=ch_signal,
                    fs_hz=fs_hz,
                    frequency_bins_hz=frequency_bins_hz,
                    nperseg=fo_config.welch_nperseg,
                    nfft=fo_config.welch_nfft,
                )
            )

        per_ch.append(ch_feats)

    return reduce_channel_features(per_channel_features=per_ch, reductions=reductions)


def validate_row_eligibility(
    dataset: nc.Dataset,
    sensor_id: str,
    require_z_axis: bool,
) -> Tuple[bool, str]:
    if "acc" not in dataset.groups:
        return False, "missing_acc_group"

    acc_root = dataset.groups["acc"]
    if sensor_id not in acc_root.groups:
        return False, "missing_sensor_group"

    sensor_group = acc_root.groups[sensor_id]
    # Support both the current name (velocity_mms) and the legacy name used
    # in NetCDF files built before the rename.
    _vel_var = next(
        (
            n
            for n in ("velocity_mms", "acceleration_mps2")
            if n in sensor_group.variables
        ),
        None,
    )
    if _vel_var is None:
        return False, "missing_velocity_mms"

    accel = np.asarray(sensor_group.variables[_vel_var][:], dtype=np.float64)
    if accel.ndim != 2 or accel.shape[1] < 3 or accel.shape[0] < 2:
        return False, "invalid_accel_shape"

    if require_z_axis:
        axis_mask = _read_axis_mask_for_sensor(dataset, sensor_id)
        if axis_mask is None:
            return False, "missing_axis_mask"
        if int(axis_mask[2]) != 1:
            return False, "missing_z_axis"

    return True, "ok"


def load_event_metadata(dataset: nc.Dataset) -> Dict[str, Any]:
    meta = dataset.groups.get("meta_acc") or dataset.groups.get("meta")

    return {
        "event_id": str(_to_python_scalar(getattr(dataset, "event_id", ""))),
        "site_id": str(_to_python_scalar(getattr(dataset, "site_id", ""))),
        "train_type": str(_read_scalar_var(meta, "train_type", default="unknown")),
        "train_speed_kmh": _safe_float(
            _read_scalar_var(meta, "train_speed_kmh", np.nan), default=np.nan
        ),
        "track_number": _safe_int(
            _read_scalar_var(meta, "track_number", -1), default=-1
        ),
    }


def load_sensor_geometry(dataset: nc.Dataset, sensor_id: str) -> Dict[str, Any]:
    geometry = dataset.groups.get("geometry_acc") or dataset.groups.get("geometry")
    if geometry is None:
        return {
            "acc_distance_to_track_m": np.nan,
            "acc_side_of_track": 0,
        }

    if "acc_sensor_id" not in geometry.variables:
        return {
            "acc_distance_to_track_m": np.nan,
            "acc_side_of_track": 0,
        }

    sensor_ids = [
        str(_to_python_scalar(v)) for v in geometry.variables["acc_sensor_id"][:]
    ]
    if sensor_id not in sensor_ids:
        return {
            "acc_distance_to_track_m": np.nan,
            "acc_side_of_track": 0,
        }

    idx = sensor_ids.index(sensor_id)

    dist = np.nan
    side = 0

    if "acc_distance_to_track_m" in geometry.variables:
        dist = _safe_float(
            geometry.variables["acc_distance_to_track_m"][idx], default=np.nan
        )

    if "acc_side_of_track" in geometry.variables:
        side = _safe_int(geometry.variables["acc_side_of_track"][idx], default=0)

    return {
        "acc_distance_to_track_m": dist,
        "acc_side_of_track": side,
    }


def get_sensor_ids(dataset: nc.Dataset) -> List[str]:
    if "acc" not in dataset.groups:
        return []
    return list(dataset.groups["acc"].groups.keys())


def assemble_output_row(
    event_meta: Dict[str, Any],
    sensor_id: str,
    target_pgv_z_mms: float,
    geometry: Dict[str, Any],
    fo_features: Dict[str, float],
) -> Dict[str, Any]:
    row = {
        "event_id": event_meta["event_id"],
        "site_id": event_meta["site_id"],
        "sensor_id": sensor_id,
        "target_pgv_z_mms": target_pgv_z_mms,
        "train_type": event_meta["train_type"],
        "train_speed_kmh": event_meta["train_speed_kmh"],
        "track_number": event_meta["track_number"],
        "acc_distance_to_track_m": geometry["acc_distance_to_track_m"],
        "acc_side_of_track": geometry["acc_side_of_track"],
    }
    row.update(fo_features)
    return row


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def create_build_folder(output_root: Path, version_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_root / f"{version_name}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_build_log(log_path: Path, lines: List[str]) -> None:
    with open(log_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def write_parquet(df: pd.DataFrame, output_path: Path, engine: str) -> None:
    try:
        df.to_parquet(output_path, index=False, engine=engine)
    except ImportError as exc:
        raise ImportError(
            f"Parquet engine '{engine}' is not available. "
            "Install the requested dependency (e.g., pyarrow)."
        ) from exc
