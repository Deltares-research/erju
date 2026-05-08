"""Helpers for building Parquet v3 from event-level NetCDF files.

v3 changes vs v2:
  - FO features computed per track line (A/B/C/D/E) over ±5-channel sub-windows
    instead of the full 51-channel average.
  - Each row gets features from all 5 lines.  Column names carry a line prefix:
      fo_lineA_td_rms_mean, fo_lineA_oct_001hz_mean, fo_lineB_..., etc.
  - Signed longitudinal offsets from the sensor's own line to every other line
    (in metres; 1 FO channel = 1 m along the track).
  - Effective distance to the active track accounts for the 4 m track separation
    when track_number == 2.
  - Train type strings mapped to 8 physics-based families; GO always wins.
  - Only sensors with acc_side_of_track == -1 are included (enforced in the
    build script via InclusionRulesConfig.require_side_of_track).

All unchanged utilities are re-exported from v1_utils / v2_utils so the build
script only needs to import from this module.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple

import netCDF4 as nc
import numpy as np

# ---------------------------------------------------------------------------
# Re-exports from v1 — unchanged utilities
# ---------------------------------------------------------------------------
from src.db.parquet.parquet_v1_utils import (  # noqa: F401
    SkipRecord,
    compute_fo_time_domain_features,
    compute_pgv_z_mms,
    create_build_folder,
    get_sensor_ids,
    load_event_metadata,
    load_sensor_geometry,
    open_netcdf_event,
    process_fo_event_signal,
    reduce_channel_features,
    save_json,
    validate_row_eligibility,
    write_build_log,
    write_parquet,
    _safe_float,
)

# Re-export octave-band helpers from v2
from src.db.parquet.parquet_v2_utils import (  # noqa: F401
    compute_fo_octave_band_features,
    _oct_feature_name,
)


# ---------------------------------------------------------------------------
# Train type family assignment
# ---------------------------------------------------------------------------

# Priority-ordered rules: first matching token wins.
# GO always wins — checked before any passenger type.
_FAMILY_RULES: List[Tuple[str, str]] = [
    ("GO",  "GO"),
    ("ICM", "ICM"),
    ("ICR", "ICR"),
    ("IC(", "ICR"),
    ("SNG", "SNG"),
    ("SPR", "SPR"),
    ("DDZ", "DDZ"),
    ("LL",  "Locomotive"),
    ("LM",  "Locomotive"),
    ("INT", "Locomotive"),
]

FAMILY_NAMES: List[str] = [
    "GO", "ICM", "ICR", "SNG", "SPR", "DDZ", "Locomotive", "Other"
]

FAMILY_TO_CODE: Dict[str, int] = {name: idx for idx, name in enumerate(FAMILY_NAMES)}


def assign_train_type_family(train_type: str) -> str:
    """Return the physics-based family label for a raw train_type string.

    Priority: GO wins over all.  First matching token is used.
    """
    s = str(train_type)
    for token, family in _FAMILY_RULES:
        if token in s:
            return family
    return "Other"


# ---------------------------------------------------------------------------
# Sensor line code
# ---------------------------------------------------------------------------

_LINE_CODE: Dict[str, int] = {
    "line_A": 0,
    "line_B": 1,
    "line_C": 2,
    "line_D": 3,
    "line_E": 4,
}


# ---------------------------------------------------------------------------
# Per-line FO channel extraction helpers
# ---------------------------------------------------------------------------

def _get_line_channel_mask(
    fo_channel_ids: np.ndarray,
    center_channel: int,
    half_window: int,
) -> np.ndarray:
    """Boolean mask selecting channels within [center - half_window, center + half_window].

    Parameters
    ----------
    fo_channel_ids : 1-D int array of absolute channel numbers stored in the NetCDF
    center_channel : center channel for the target track line
    half_window    : number of channels on each side of the center (inclusive)
    """
    return np.abs(fo_channel_ids - center_channel) <= half_window


def _make_nan_line_features(
    line_key: str,
    octave_bands: List[Dict[str, float]],
    reductions: List[str],
    include_time_domain: bool,
    include_spectral: bool,
) -> Dict[str, float]:
    """NaN-filled feature dict for a line whose channel window is empty.

    Column names match what compute_event_fo_features_per_line would produce,
    so the schema stays consistent across all events.

    line_key : 'lineA', 'lineB', etc.  (already stripped of 'line_' prefix)
    """
    prefix = f"fo_{line_key}_"
    out: Dict[str, float] = {}

    if include_time_domain:
        for base in ["td_rms", "td_max_abs", "td_std"]:
            for red in reductions:
                out[f"{prefix}{base}_{red}"] = np.nan

    if include_spectral:
        for band in octave_bands:
            # _oct_feature_name returns e.g. 'fo_oct_001hz'; strip 'fo_' → 'oct_001hz'
            base = _oct_feature_name(band["nominal_hz"])[3:]
            for red in reductions:
                out[f"{prefix}{base}_{red}"] = np.nan

    return out


# ---------------------------------------------------------------------------
# Per-line FO feature computation (core of v3)
# ---------------------------------------------------------------------------

def compute_event_fo_features_per_line(
    dataset: nc.Dataset,
    fo_config: Any,
    octave_bands: List[Dict[str, float]],
    reductions: List[str],
    line_centers: Dict[str, int],
    line_half_window: int,
    include_time_domain: bool,
    include_spectral: bool,
) -> Optional[Dict[str, float]]:
    """Compute per-line FO features for one event.

    Steps
    -----
    1. Bandpass-filter the full stored strain array (n_time × 51 channels).
    2. For each track line (A/B/C/D/E):
       a. Select the sub-window channels (center ± half_window).
       b. Compute time-domain and/or 1/3-octave spectral features per channel.
       c. Reduce across sub-window channels (mean / max / std).
       d. Prefix all column names with 'fo_<lineX>_'.
    3. Return a flat dict containing features for all 5 lines.

    Column naming convention (example for line_A):
      Time-domain  : fo_lineA_td_rms_mean, fo_lineA_td_rms_max, ...
      Spectral     : fo_lineA_oct_001hz_mean, fo_lineA_oct_001hz_max, ...

    Returns None if the FO signal cannot be loaded or processed.
    """
    # Load and bandpass-filter the full strain array (shared across all lines)
    processed = process_fo_event_signal(dataset=dataset, fo_config=fo_config)
    if processed is None:
        return None

    # Read absolute channel IDs from geometry_fo
    if "geometry_fo" not in dataset.groups:
        return None
    geom_fo = dataset.groups["geometry_fo"]
    if "fo_channel_id" not in geom_fo.variables:
        return None
    fo_channel_ids = np.asarray(geom_fo.variables["fo_channel_id"][:], dtype=np.int32)

    fs_hz = _safe_float(
        dataset.groups["fo"].variables["fs_hz"][()], default=np.nan
    )
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        return None

    all_features: Dict[str, float] = {}

    for line_label, center_ch in line_centers.items():
        # e.g. line_label = "line_A"  →  line_key = "lineA"
        line_key = line_label.replace("line_", "line")

        mask = _get_line_channel_mask(fo_channel_ids, center_ch, line_half_window)
        n_in_window = int(np.sum(mask))

        if n_in_window == 0:
            # No stored channels cover this line — fill with NaN
            all_features.update(
                _make_nan_line_features(
                    line_key=line_key,
                    octave_bands=octave_bands,
                    reductions=reductions,
                    include_time_domain=include_time_domain,
                    include_spectral=include_spectral,
                )
            )
            continue

        # Sub-window signal: (n_time, n_in_window)
        sub_window = processed[:, mask]

        # Per-channel features (strip the leading 'fo_' prefix so we can add
        # our own 'fo_lineX_' prefix cleanly without duplication)
        per_ch: List[Dict[str, float]] = []
        for ch_idx in range(sub_window.shape[1]):
            ch_signal = sub_window[:, ch_idx]
            ch_feats: Dict[str, float] = {}

            if include_time_domain:
                # compute_fo_time_domain_features returns {'fo_td_rms': ..., ...}
                # Strip 'fo_' → {'td_rms': ..., 'td_max_abs': ..., 'td_std': ...}
                td = compute_fo_time_domain_features(ch_signal)
                ch_feats.update(
                    {k[3:] if k.startswith("fo_") else k: v for k, v in td.items()}
                )

            if include_spectral:
                # compute_fo_octave_band_features returns {'fo_oct_001hz': ..., ...}
                # Strip 'fo_' → {'oct_001hz': ..., ...}
                spec = compute_fo_octave_band_features(
                    strain_ch=ch_signal,
                    fs_hz=fs_hz,
                    octave_bands=octave_bands,
                    nperseg=fo_config.welch_nperseg,
                    nfft=fo_config.welch_nfft,
                )
                ch_feats.update(
                    {k[3:] if k.startswith("fo_") else k: v for k, v in spec.items()}
                )

            per_ch.append(ch_feats)

        # Reduce across sub-window channels → e.g. 'td_rms_mean', 'oct_001hz_max'
        reduced = reduce_channel_features(per_ch, reductions)

        # Add line prefix: 'fo_lineA_td_rms_mean', etc.
        prefix = f"fo_{line_key}_"
        for col, val in reduced.items():
            all_features[f"{prefix}{col}"] = val

    return all_features if all_features else None


# ---------------------------------------------------------------------------
# Longitudinal offset features (sensor-relative, in metres)
# ---------------------------------------------------------------------------

def compute_longitudinal_offsets(
    sensor_line: str,
    line_centers: Dict[str, int],
) -> Dict[str, float]:
    """Signed offset from this sensor's line center to every other line center.

    Positive = ahead (higher channel / further along track in positive direction),
    negative = behind.  1 FO channel = 1 metre along the track.

    Column names: fo_offset_to_lineA_m, fo_offset_to_lineB_m, ...

    Example — MP8 on line_C (center = 1194):
      fo_offset_to_lineA_m = 1184 - 1194 = -10.0  (10 m behind)
      fo_offset_to_lineB_m = 1192 - 1194 =  -2.0
      fo_offset_to_lineC_m =                 0.0
      fo_offset_to_lineD_m = 1196 - 1194 =  +2.0
      fo_offset_to_lineE_m = 1204 - 1194 = +10.0  (10 m ahead)

    Example — MP12 on line_A (center = 1184):
      fo_offset_to_lineA_m =   0.0
      fo_offset_to_lineB_m = 1192 - 1184 = +8.0
      fo_offset_to_lineC_m = 1194 - 1184 = +10.0
      fo_offset_to_lineD_m = 1196 - 1184 = +12.0
      fo_offset_to_lineE_m = 1204 - 1184 = +20.0
    """
    if sensor_line not in line_centers:
        # Unknown sensor line — return NaN for all offsets
        return {
            f"fo_offset_to_{lbl.replace('line_', 'line')}_m": np.nan
            for lbl in line_centers
        }

    own_center = line_centers[sensor_line]
    return {
        f"fo_offset_to_{lbl.replace('line_', 'line')}_m": float(center - own_center)
        for lbl, center in line_centers.items()
    }


# ---------------------------------------------------------------------------
# Effective distance to the active track
# ---------------------------------------------------------------------------

def compute_effective_distance(
    acc_distance_to_track_m: float,
    track_number: int,
    track_separation_m: float,
) -> float:
    """Perpendicular distance from the sensor to the rail the train is using.

    acc_distance_to_track_m is always measured from track 1 (the closer rail).
    When track_number == 2, the active track is track_separation_m further away.
    track_number == -1 (unknown) returns NaN so the model can handle it.
    """
    if not np.isfinite(acc_distance_to_track_m):
        return np.nan
    if track_number == 2:
        return float(acc_distance_to_track_m + track_separation_m)
    if track_number == 1:
        return float(acc_distance_to_track_m)
    return np.nan  # unknown track


# ---------------------------------------------------------------------------
# Row assembly (v3)
# ---------------------------------------------------------------------------

def assemble_output_row_v3(
    event_meta: Dict[str, Any],
    sensor_id: str,
    target_pgv_z_mms: float,
    geometry: Dict[str, Any],
    sensor_line: str,
    line_centers: Dict[str, int],
    track_separation_m: float,
    fo_features: Dict[str, float],
) -> Dict[str, Any]:
    """Assemble one output row for v3 Parquet.

    Column order:
      identifiers → target → train metadata → geometry → line/offset → FO features
    """
    train_type = event_meta["train_type"]
    family = assign_train_type_family(train_type)
    track_number = event_meta["track_number"]
    acc_dist = geometry["acc_distance_to_track_m"]

    row: Dict[str, Any] = {
        # Identifiers (excluded from ML features)
        "event_id": event_meta["event_id"],
        "site_id": event_meta["site_id"],
        "sensor_id": sensor_id,
        # Target
        "target_pgv_z_mms": target_pgv_z_mms,
        # Train metadata
        "train_type": train_type,
        "train_type_family": family,
        "train_speed_kmh": event_meta["train_speed_kmh"],
        "track_number": track_number,
        # Geometry
        "acc_distance_to_track_m": acc_dist,
        "effective_distance_to_active_track_m": compute_effective_distance(
            acc_distance_to_track_m=acc_dist,
            track_number=track_number,
            track_separation_m=track_separation_m,
        ),
        "acc_side_of_track": geometry["acc_side_of_track"],
        # Line identity and longitudinal positions
        "sensor_line_code": _LINE_CODE.get(sensor_line, -1),
    }

    # Signed longitudinal offsets (sensor-relative, metres)
    row.update(compute_longitudinal_offsets(sensor_line, line_centers))

    # Per-line FO features (315 spectral + 45 time-domain = 360 columns)
    row.update(fo_features)

    return row
