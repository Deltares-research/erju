"""Geometry utilities for the Holten site.

The core function ``apply_corrected_distances`` must be called whenever a
sensor-level DataFrame (one row per event × sensor) is loaded from any
Parquet version whose ``acc_distance_to_track_m`` column was derived from
the raw NetCDF field-label distances.

Why this is needed
------------------
The NetCDF files store ``acc_distance_to_track_m`` values that were copied
from the physical measurement-point labels (e.g. "MP4_Holten_2m_C → 2.0 m").
Those labels are:
  1. Always the distance to **track 1** — independent of which track the train
     actually used during that event.
  2. Rounded / approximate; the correct perpendicular distances derived from
     the surveyed sensor coordinates (holten.json) differ for some sensors
     (e.g. MP4: 2.0 m stored → 2.5 m correct; MP2: 25.0 m → 23.0 m).

The fix
-------
holten.json now contains:
  ``distance_to_track_1_m``  — |y_sensor - 4.0|   (Track 1 at Y=4 m)
  ``distance_to_track_2_m``  — |y_sensor - 8.0|   (Track 2 at Y=8 m)

``apply_corrected_distances`` uses these tables to:
  1. Overwrite ``acc_distance_to_track_m`` with the correct track-1 distance.
  2. Add ``acc_distance_to_track_2_m`` with the correct track-2 distance.
  3. Add ``effective_distance_to_active_track_m`` = distance to the track the
     train was actually running on (from ``track_number`` column).

Distance convention
-------------------
Distance = perpendicular Y-only separation between sensor and rail centreline.
The train is a longitudinal source; its closest approach to a sensor occurs
when the train's position along X equals the sensor's X.  At that moment only
the Y component is physically relevant for attenuation.  X offsets (line A/E
being ±10 m from line C) are handled separately via per-line FO sub-windows
in Parquet v3, not through distance modification here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

# Default site JSON path (relative to repo root)
_DEFAULT_SITE_JSON = Path(__file__).parent.parent.parent / "sites" / "holten.json"


def _load_site_distances(site_json: Union[str, Path, None]) -> tuple[dict, dict]:
    """Load distance_to_track_1_m and distance_to_track_2_m from the site JSON.

    Returns
    -------
    (dist_t1, dist_t2) — dicts mapping sensor_id → float distance [m]
    """
    path = Path(site_json) if site_json is not None else _DEFAULT_SITE_JSON
    with open(path, "r") as f:
        site = json.load(f)

    accel = site["accelerometer"]
    dist_t1_raw = accel["distance_to_track_1_m"]
    dist_t2_raw = accel["distance_to_track_2_m"]

    # Strip the _comment key if present
    dist_t1 = {k: float(v) for k, v in dist_t1_raw.items() if k != "_comment"}
    dist_t2 = {k: float(v) for k, v in dist_t2_raw.items() if k != "_comment"}
    return dist_t1, dist_t2


def apply_corrected_distances(
    df: pd.DataFrame,
    site_json: Union[str, Path, None] = None,
    sensor_col: str = "sensor_id",
    track_col: str = "track_number",
    inplace: bool = False,
) -> pd.DataFrame:
    """Override distance columns in a sensor-level DataFrame with correct geometry.

    Parameters
    ----------
    df
        Sensor-level DataFrame (one row per event × sensor).  Must contain
        ``sensor_col`` and ``track_col``.
    site_json
        Path to the site JSON file.  Defaults to ``sites/holten.json`` in the
        repo root.
    sensor_col
        Column that holds the short sensor ID (e.g. ``'MP4'``).
    track_col
        Column that holds the track number (1 or 2; -1 if unknown).
    inplace
        If False (default) operate on a copy.

    Returns
    -------
    DataFrame with three columns added / overwritten:

    ``acc_distance_to_track_m``
        Correct perpendicular distance to **track 1** [m], from holten.json.
        Replaces the label-derived (potentially wrong) NetCDF value.

    ``acc_distance_to_track_2_m``
        Correct perpendicular distance to **track 2** [m].

    ``effective_distance_to_active_track_m``
        Distance to the track the train was running on.
        = ``acc_distance_to_track_m``   when ``track_number`` == 1
        = ``acc_distance_to_track_2_m`` when ``track_number`` == 2
        = NaN                           when ``track_number`` is unknown / -1

    Notes
    -----
    Rows whose ``sensor_col`` value is not found in holten.json (e.g. MP14–MP19
    if they appear) are left unchanged in ``acc_distance_to_track_m`` and get
    NaN for the two new columns.
    """
    if not inplace:
        df = df.copy()

    dist_t1, dist_t2 = _load_site_distances(site_json)

    # Vectorised lookup via map (fast, NaN for unknown sensor IDs)
    d1 = df[sensor_col].map(dist_t1)
    d2 = df[sensor_col].map(dist_t2)

    # 1. Override acc_distance_to_track_m with correct track-1 distance
    df["acc_distance_to_track_m"] = np.where(
        d1.notna(), d1, df["acc_distance_to_track_m"]
    )

    # 2. Add track-2 distance column
    df["acc_distance_to_track_2_m"] = d2

    # 3. Compute effective distance to the active track
    track = df[track_col].fillna(-1).astype(int)
    effective = np.where(track == 1, d1, np.where(track == 2, d2, np.nan))
    df["effective_distance_to_active_track_m"] = effective.astype(float)

    return df
