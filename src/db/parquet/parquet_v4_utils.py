"""Utilities for building Parquet v4 — event-level, attenuation-curve targets.

v4 vs v2/v3
-----------
v2/v3 have one row per (train event × accelerometer sensor).  This is the
direct-prediction setup where the model learns both *event intensity* and
*spatial attenuation* simultaneously.

v4 is **event-level**: one row per train event.
FO features are identical for every sensor row of the same event, so taking
the first occurrence per event is lossless.  The sensor-level PGV
measurements are used only to fit an attenuation curve; the curve parameters
become the ML targets.

Attenuation model
-----------------
  log( PGV(r) ) = c_i  -  n * log( r / r0 )

  PGV(r) = exp(c_i) * (r / r0)^{-n}

  c_i  ... log-intensity at the reference distance r0
  n    ... attenuation exponent (positive → PGV decreases with distance)
  r0   ... reference distance, default 10.0 m

Scenario 1 — shared exponent
    All events share one global n; each event gets its own c_i.
    Estimated via within-event (fixed-effects) OLS.

Scenario 2 — per-event exponent
    Each event is independently fit for (c_i, n_i).
    Only events with ≥ min_points valid sensors are kept.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Spectral texture features from octave-band mean powers
# ---------------------------------------------------------------------------


def compute_spectral_texture_features_from_bands(
    band_powers: np.ndarray,
) -> Dict[str, float]:
    """Compute 5 spectral texture statistics from a 1-D array of band powers.

    Treats the ordered band-power values as a probability mass function and
    computes distributional moments and information-theoretic measures.
    All statistics are dimensionless.

    Parameters
    ----------
    band_powers
        1-D array of shape (n_bands,) with non-negative power values.

    Returns
    -------
    dict with keys:
        fo_texture_spectral_skewness
        fo_texture_spectral_kurtosis      (excess kurtosis)
        fo_texture_spectral_entropy       (Shannon, normalised to [0, 1])
        fo_texture_spectral_crest_factor  (peak / RMS)
        fo_texture_spectral_centroid_band (weighted mean band index)
    """
    _KEYS = [
        "fo_texture_spectral_skewness",
        "fo_texture_spectral_kurtosis",
        "fo_texture_spectral_entropy",
        "fo_texture_spectral_crest_factor",
        "fo_texture_spectral_centroid_band",
    ]
    b = np.asarray(band_powers, dtype=np.float64)
    b = np.clip(b, 0.0, None)

    if b.sum() < 1e-30:
        return {k: np.nan for k in _KEYS}

    n_bands = len(b)
    p = b / b.sum()  # normalise to probability

    # Band index as the "frequency variable"
    idx = np.arange(n_bands, dtype=np.float64)
    centroid = float(np.dot(p, idx))
    variance = float(np.dot(p, (idx - centroid) ** 2))
    std_p = np.sqrt(variance) if variance > 1e-30 else 1.0

    skewness = float(np.dot(p, ((idx - centroid) / std_p) ** 3))
    kurtosis = float(np.dot(p, ((idx - centroid) / std_p) ** 4)) - 3.0  # excess

    # Shannon entropy (normalised by log(n_bands))
    entropy_raw = float(-np.dot(p, np.log(p + 1e-30)))
    max_entropy = np.log(n_bands)
    entropy_norm = float(entropy_raw / max_entropy) if max_entropy > 0 else 0.0

    # Crest factor: peak / RMS of the un-normalised powers
    peak = float(b.max())
    rms = float(np.sqrt(np.mean(b**2)))
    crest = float(peak / rms) if rms > 1e-30 else np.nan

    return {
        "fo_texture_spectral_skewness": skewness,
        "fo_texture_spectral_kurtosis": kurtosis,
        "fo_texture_spectral_entropy": entropy_norm,
        "fo_texture_spectral_crest_factor": crest,
        "fo_texture_spectral_centroid_band": centroid,
    }


def add_texture_features_to_event_df(
    df: pd.DataFrame,
    octave_band_col_prefix: str = "fo_oct_",
    reduction: str = "mean",
) -> pd.DataFrame:
    """Add 5 spectral texture columns computed across all octave-band features.

    The octave-band *mean* values (or another aggregation) for the 21 bands
    are treated as a spectral shape.  Five summary statistics are computed
    per row and appended as new columns.

    Parameters
    ----------
    df
        Event-level DataFrame (one row per event).
    octave_band_col_prefix
        Common prefix for octave-band feature columns (e.g. ``'fo_oct_'``).
    reduction
        Which reduction suffix to use: ``'mean'``, ``'max'``, or ``'std'``.

    Returns
    -------
    Copy of *df* with 5 new ``fo_texture_spectral_*`` columns appended.
    """
    df = df.copy()
    band_cols = sorted(
        [
            c
            for c in df.columns
            if c.startswith(octave_band_col_prefix) and c.endswith(f"_{reduction}")
        ]
    )
    if not band_cols:
        return df

    texture_rows = [
        compute_spectral_texture_features_from_bands(
            row[band_cols].to_numpy(dtype=np.float64)
        )
        for _, row in df.iterrows()
    ]
    texture_df = pd.DataFrame(texture_rows, index=df.index)
    for col in texture_df.columns:
        df[col] = texture_df[col]
    return df


# ---------------------------------------------------------------------------
# Event-level FO feature aggregation from sensor-level DataFrame
# ---------------------------------------------------------------------------


def aggregate_event_level_features(
    df: pd.DataFrame,
    event_col: str = "event_id",
    drop_sensor_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Collapse a sensor-level DataFrame to one row per event.

    FO features and train-metadata columns are identical across all sensor
    rows of the same event.  We take the *first* occurrence per event.

    Parameters
    ----------
    df
        Sensor-level DataFrame with one row per (event × sensor).
    event_col
        Column name for the event identifier.
    drop_sensor_cols
        Sensor-specific columns to drop *before* deduplication, e.g.
        ``['sensor_id', 'acc_distance_to_track_m', 'target_pgv_z_mms',
           'acc_side_of_track']``.

    Returns
    -------
    Event-level DataFrame with one row per event.
    """
    if drop_sensor_cols is None:
        drop_sensor_cols = []

    cols_to_drop = [c for c in drop_sensor_cols if c in df.columns]
    df_event = (
        df.drop(columns=cols_to_drop)
        .groupby(event_col, sort=False)
        .first()
        .reset_index()
    )
    return df_event


# ---------------------------------------------------------------------------
# Scenario 1 — global attenuation exponent
# ---------------------------------------------------------------------------


def fit_global_attenuation_model(
    df: pd.DataFrame,
    event_col: str = "event_id",
    distance_col: str = "acc_distance_to_track_m",
    pgv_col: str = "target_pgv_z_mms",
    r0: float = 10.0,
) -> Dict:
    """Fit log(PGV) = c_i - n_global * log(distance / r0).

    A single shared attenuation exponent *n_global* is estimated from all
    sensor-level observations via within-event (fixed-effects) OLS.  Each
    event then gets its own intercept *c_i* computed analytically from the
    event mean.

    Parameters
    ----------
    df
        Sensor-level DataFrame (one row per event × sensor).
    event_col
        Column for event identifier.
    distance_col
        Column for sensor distance to track [m].
    pgv_col
        Column for measured PGV [mm/s].
    r0
        Reference distance in metres (default 10.0 m).

    Returns
    -------
    dict with keys:

    ``n_global``        (float) — fitted global attenuation exponent
    ``r0``              (float) — reference distance used
    ``event_params``    (DataFrame) — per-event c_i and diagnostics
    ``n_events``        (int)
    ``n_valid_rows``    (int)
    ``overall_rmse_log``(float)
    ``design_r2``       (float) — R² of the full design on log(PGV)
    """
    # 1. Filter valid rows
    mask = (df[pgv_col].gt(0)) & (df[distance_col].gt(0)) & df[event_col].notna()
    dv = df[mask].copy()
    if len(dv) == 0:
        raise ValueError(
            f"No valid rows after filtering {pgv_col} > 0 and {distance_col} > 0."
        )

    # 2. Log transforms
    dv["_log_pgv"] = np.log(dv[pgv_col].to_numpy(dtype=np.float64))
    dv["_log_dr"] = np.log(dv[distance_col].to_numpy(dtype=np.float64) / r0)

    # 3. Within-event demeaning (Frisch-Waugh / fixed-effects)
    ev_means = dv.groupby(event_col)[["_log_pgv", "_log_dr"]].transform("mean")
    y_dm = dv["_log_pgv"].to_numpy() - ev_means["_log_pgv"].to_numpy()
    x_dm = dv["_log_dr"].to_numpy() - ev_means["_log_dr"].to_numpy()

    denom = float(np.dot(x_dm, x_dm))
    if denom < 1e-20:
        raise ValueError(
            "No within-event variation in log(distance / r0). "
            "All sensors of each event are at the same distance — "
            "cannot estimate n_global."
        )

    slope = float(np.dot(x_dm, y_dm) / denom)
    n_global = -slope  # positive → PGV decreases with distance

    # 4. Recover per-event intercepts: c_i = mean_log_pgv + n_global * mean_log_dr
    agg = dv.groupby(event_col)[["_log_pgv", "_log_dr"]].mean()
    c_i = (agg["_log_pgv"] + n_global * agg["_log_dr"]).rename("c_i")

    # 5. Per-event diagnostics
    dv["_c_i"] = c_i.loc[dv[event_col]].to_numpy()
    dv["_pred"] = dv["_c_i"] - n_global * dv["_log_dr"]
    dv["_res"] = dv["_log_pgv"] - dv["_pred"]

    diag_records = []
    for eid, grp in dv.groupby(event_col):
        res = grp["_res"].to_numpy()
        lp = grp["_log_pgv"].to_numpy()
        ss_res = float(np.dot(res, res))
        ss_tot = float(np.dot(lp - lp.mean(), lp - lp.mean()))
        diag_records.append(
            {
                event_col: eid,
                "n_sensors_used": len(grp),
                "rmse_log_fit": float(np.sqrt(np.mean(res**2))),
                "r2_fit": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else np.nan,
            }
        )

    diag_df = pd.DataFrame(diag_records)
    event_params = (
        c_i.reset_index()
        .rename(columns={"index": event_col})
        .merge(diag_df, on=event_col, how="left")
    )
    event_params["n_global"] = n_global

    # 6. Overall diagnostics
    all_res = dv["_res"].to_numpy()
    all_lp = dv["_log_pgv"].to_numpy()
    overall_rmse = float(np.sqrt(np.mean(all_res**2)))
    ss_res_all = float(np.dot(all_res, all_res))
    ss_tot_all = float(np.dot(all_lp - all_lp.mean(), all_lp - all_lp.mean()))
    design_r2 = float(1.0 - ss_res_all / ss_tot_all) if ss_tot_all > 1e-30 else np.nan

    return {
        "n_global": n_global,
        "r0": r0,
        "event_params": event_params,
        "n_events": int(len(c_i)),
        "n_valid_rows": int(len(dv)),
        "overall_rmse_log": overall_rmse,
        "design_r2": design_r2,
    }


def predict_pgv_global_curve(
    c_pred: np.ndarray,
    distance: np.ndarray,
    n_global: float,
    r0: float = 10.0,
) -> np.ndarray:
    """Reconstruct PGV from predicted event intensity and a global exponent.

    Parameters
    ----------
    c_pred   : 1-D array of predicted log-intensities at reference distance.
    distance : 1-D array of sensor distances to the track [m].
    n_global : global attenuation exponent (positive).
    r0       : reference distance [m].

    Returns
    -------
    PGV predictions in mm/s.
    """
    c = np.asarray(c_pred, dtype=np.float64)
    r = np.asarray(distance, dtype=np.float64)
    r_clipped = np.clip(r, 1e-3, None)
    return np.exp(c - n_global * np.log(r_clipped / r0))


def build_event_level_dataset_for_global_curve(
    df_event_features: pd.DataFrame,
    df_curve_params: pd.DataFrame,
    event_col: str = "event_id",
) -> pd.DataFrame:
    """Merge event-level FO/train features with the Scenario 1 c_i targets.

    Parameters
    ----------
    df_event_features
        Event-level feature DataFrame (one row per event, from
        ``aggregate_event_level_features``).
    df_curve_params
        Output of ``fit_global_attenuation_model``['event_params'].
    event_col
        Column name for event identifier (must appear in both DataFrames).

    Returns
    -------
    Merged event-level DataFrame with c_i, n_global, and diagnostics added.
    """
    merge_cols = [
        c
        for c in [
            event_col,
            "c_i",
            "n_global",
            "n_sensors_used",
            "rmse_log_fit",
            "r2_fit",
        ]
        if c in df_curve_params.columns
    ]
    return df_event_features.merge(
        df_curve_params[merge_cols],
        on=event_col,
        how="inner",
    )


# ---------------------------------------------------------------------------
# Scenario 2 — per-event attenuation exponent
# ---------------------------------------------------------------------------


def fit_event_specific_attenuation_curves(
    df: pd.DataFrame,
    event_col: str = "event_id",
    distance_col: str = "acc_distance_to_track_m",
    pgv_col: str = "target_pgv_z_mms",
    r0: float = 10.0,
    min_points: int = 3,
    max_abs_n: float = 5.0,
) -> pd.DataFrame:
    """Fit log(PGV) = c_i - n_i * log(distance / r0) independently per event.

    For each event a simple OLS line is fit in log–log space using all
    available valid sensor measurements.  Events with fewer than *min_points*
    valid sensors, or where all sensors are at the same distance, are flagged
    and their c_i / n_i set to NaN.

    Parameters
    ----------
    df
        Sensor-level DataFrame (one row per event × sensor).
    event_col
        Column for event identifier.
    distance_col
        Column for sensor distance to track [m].
    pgv_col
        Column for measured PGV [mm/s].
    r0
        Reference distance in metres.
    min_points
        Minimum number of valid sensor points required to attempt a fit.
    max_abs_n
        Events whose |n_i| exceeds this are flagged as ``'n_out_of_range'``
        and receive ``quality_flag=0``.

    Returns
    -------
    DataFrame with one row per event and columns:
        event_id, c_i, n_i, n_sensors_used, rmse_log, r2, fit_status,
        quality_flag
    """
    records = []
    for eid, grp in df.groupby(event_col):
        mask = (grp[pgv_col] > 0) & (grp[distance_col] > 0)
        g = grp[mask]
        n_pts = len(g)

        base = {event_col: eid, "n_sensors_used": n_pts}

        if n_pts < min_points:
            records.append(
                {
                    **base,
                    "c_i": np.nan,
                    "n_i": np.nan,
                    "rmse_log": np.nan,
                    "r2": np.nan,
                    "fit_status": "insufficient_points",
                    "quality_flag": 0,
                }
            )
            continue

        x = np.log(g[distance_col].to_numpy(dtype=np.float64) / r0)
        y = np.log(g[pgv_col].to_numpy(dtype=np.float64))

        if x.std() < 1e-8:
            # All sensors at the same distance — can only estimate c_i
            records.append(
                {
                    **base,
                    "c_i": float(y.mean()),
                    "n_i": 0.0,
                    "rmse_log": float(y.std()),
                    "r2": np.nan,
                    "fit_status": "no_distance_variation",
                    "quality_flag": 0,
                }
            )
            continue

        # OLS: y = c_i + slope * x  →  n_i = -slope
        slope, intercept, r_val, _p, _se = stats.linregress(x, y)
        c_i = float(intercept)
        n_i = float(-slope)
        r2 = float(r_val**2)
        y_pred = intercept + slope * x
        rmse_log = float(np.sqrt(np.mean((y - y_pred) ** 2)))

        quality_flag = 1
        fit_status = "ok"
        if abs(n_i) > max_abs_n:
            quality_flag = 0
            fit_status = "n_out_of_range"

        records.append(
            {
                **base,
                "c_i": c_i,
                "n_i": n_i,
                "rmse_log": rmse_log,
                "r2": r2,
                "fit_status": fit_status,
                "quality_flag": quality_flag,
            }
        )

    return pd.DataFrame(records)


def predict_pgv_event_curve(
    c_pred: np.ndarray,
    n_pred: np.ndarray,
    distance: np.ndarray,
    r0: float = 10.0,
) -> np.ndarray:
    """Reconstruct PGV from predicted per-event curve parameters.

    Parameters
    ----------
    c_pred   : 1-D array of predicted log-intensities at reference distance.
    n_pred   : 1-D array of predicted attenuation exponents.
    distance : 1-D array of sensor distances to the track [m].
    r0       : reference distance [m].

    Returns
    -------
    PGV predictions in mm/s.
    """
    c = np.asarray(c_pred, dtype=np.float64)
    n = np.asarray(n_pred, dtype=np.float64)
    r = np.clip(np.asarray(distance, dtype=np.float64), 1e-3, None)
    return np.exp(c - n * np.log(r / r0))


def build_event_level_dataset_for_event_curves(
    df_event_features: pd.DataFrame,
    df_curve_params: pd.DataFrame,
    event_col: str = "event_id",
    quality_flag_col: str = "quality_flag",
    require_good_fit: bool = True,
) -> pd.DataFrame:
    """Merge event-level FO/train features with the Scenario 2 (c_i, n_i) targets.

    Parameters
    ----------
    df_event_features
        Event-level feature DataFrame.
    df_curve_params
        Output of ``fit_event_specific_attenuation_curves``.
    event_col
        Column for event identifier.
    quality_flag_col
        Column in *df_curve_params* with quality flags (1 = good, 0 = bad).
    require_good_fit
        If True, only events with ``quality_flag == 1`` are kept.

    Returns
    -------
    Merged DataFrame retaining only events with valid curve parameters.
    """
    params = df_curve_params.copy()
    if require_good_fit and quality_flag_col in params.columns:
        params = params[params[quality_flag_col] == 1]

    merge_cols = [
        c
        for c in [
            event_col,
            "c_i",
            "n_i",
            "n_sensors_used",
            "rmse_log",
            "r2",
            "fit_status",
            quality_flag_col,
        ]
        if c in params.columns
    ]
    return df_event_features.merge(
        params[merge_cols],
        on=event_col,
        how="inner",
    )
