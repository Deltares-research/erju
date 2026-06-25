"""
predict_pgv.py — interactive PGV prediction for a single event.

USAGE
-----
1. Set INPUT_MODE and EVENT_ID (or FO_FILE) in the CONFIG section below.
2. Set MODEL to one of: "xgb_v4", "xgb_v6", "xgb_v8_A", "xgb_v8_B"
3. Fill in the MODEL PATHS section (leave a path as "" to auto-discover
   the latest build folder for that version).
4. Run:  python predict_pgv.py

OUTPUT
------
- Table printed to console: sensor | distance | real PGV | predicted PGV | error
- Plot saved as predict_pgv_output.png (also shown interactively if possible)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===========================================================================
# CONFIGURATION — edit this section
# ===========================================================================

# --- Model selection --------------------------------------------------------
# Options: "xgb_v4" | "xgb_v6" | "xgb_v8_A" | "xgb_v8_B" | "all"
MODEL = "all"

# --- Sensor line filter -----------------------------------------------------
# Only sensors belonging to these lines are shown in the table and plot.
# Line letters come from the site config (holten.json measurement point names,
# e.g. "Meetjournal_MP1_Holten_zuid_16m_C" → line "C").
# Set to None to use ALL sensors (no filtering).
#
# Examples:
#   SENSOR_LINES = ["C"]           # line C only (default)
#   SENSOR_LINES = ["C", "D"]      # lines C and D
#   SENSOR_LINES = None            # all sensors
SENSOR_LINES: Optional[list] = ["C"]

# --- Sensor side filter -----------------------------------------------------
# Filter by side of track using the acc_side_of_track column in the dataset.
#   -1 = south / right side (default)
#    1 = north / left side
#    0 = on-sleeper sensors
# Set to None to include all sides.
#
# Examples:
#   SENSOR_SIDE: Optional[list] = [-1]      # south side only (default)
#   SENSOR_SIDE: Optional[list] = [-1, 1]   # both sides
#   SENSOR_SIDE: Optional[list] = None      # no side filter
SENSOR_SIDE: Optional[list] = [-1]

# Path to site JSON (used to resolve sensor → line mapping)
SITE_JSON = r"sites\holten.json"

# --- Input mode -------------------------------------------------------------
# "event_id" : pick a known event from the existing Parquet dataset (default)
# "fo_file"  : provide a raw FO signal file (see FO_FILE_PATH below)
INPUT_MODE = "event_id"

# --- Event selection (used when INPUT_MODE = "event_id") --------------------
# Set to a specific event_id string, or None to let the script pick one.
# To see available event IDs, set PRINT_AVAILABLE_EVENTS = True below.
EVENT_ID: Optional[str] = None  # e.g. "20240829_080626.mat"
PRINT_AVAILABLE_EVENTS = False  # set True to list all event IDs and exit

# --- Raw FO file (used when INPUT_MODE = "fo_file") -------------------------
# NOT YET IMPLEMENTED — requires running the full FO feature-extraction
# pipeline (read raw signal → window by train event → compute octave bands).
# When implemented, provide the path to a .tdms / .h5 / .npy FO signal file
# and specify the target distance(s) below.
FO_FILE_PATH: Optional[str] = None  # e.g. r"C:\data\fo_signal_2023.tdms"
FO_SENSOR_DISTANCES_M: list = [5.0, 10.0, 15.0, 20.0]  # m

# ===========================================================================
# MODEL PATHS — fill in the build folder for each version
# Leave as "" to auto-discover the latest build in the models root folder.
# ===========================================================================

MODELS_ROOT = r"P:\11210978-erju-ai\holten_models"

# XGBoost v4 — direct sensor-level model
XGB_V4_BUILD_FOLDER = r""  # e.g. r"P:\...\xgb_v004_20260408_165448"

# XGBoost v6 — event-level c_i prediction with global n
XGB_V6_BUILD_FOLDER = r""  # e.g. r"P:\...\xgb_v006_20260509_152929"

# XGBoost v8 — two-stage residual-on-physics
XGB_V8_BUILD_FOLDER = r""  # e.g. r"P:\...\xgb_v008_20260509_154809"

# ===========================================================================
# DATA PATHS
# ===========================================================================

# Sensor-level Parquet v2 (all models need this for real measurements)
# Leave as "" to auto-discover the latest parquet_v002_* build.
V2_PARQUET = (
    r"P:\11210978-erju-ai\holten_parquet"
    r"\parquet_v002_20260408_151746\dataset.parquet"
)

# Parquet v4 root (for auto-discovery of latest build)
PARQUET_V4_ROOT = r"P:\11210978-erju-ai\holten_parquet"

# ===========================================================================
# END OF CONFIGURATION
# ===========================================================================


# ---------------------------------------------------------------------------
# Sensor → line mapping (parsed from site JSON)
# ---------------------------------------------------------------------------


def _build_sensor_line_map(site_json_path: str) -> dict[str, str]:
    """Return {sensor_id: line_letter} parsed from the site JSON.

    The measurement point names follow the pattern:
        Meetjournal_MP<n>_Holten_<side>_<dist>m_<LINE>
    The last underscore-separated token is the line letter.
    """
    with open(site_json_path) as f:
        site = json.load(f)

    sensor_id_map: dict = site["accelerometer"]["sensor_id_map"]
    result: dict[str, str] = {}
    for full_name, sensor_id in sensor_id_map.items():
        parts = full_name.split("_")
        line_letter = parts[-1]  # e.g. "C", "D", "B", "A", "E"
        # If a sensor maps to multiple lines (e.g. MP6 has B and D entries),
        # the last entry wins — but we store a set via a list to be safe.
        result[sensor_id] = line_letter
    return result


def _filter_sensors_by_line(
    df: pd.DataFrame, lines: list, sensor_line_map: dict
) -> pd.DataFrame:
    """Keep only rows whose sensor_id belongs to one of the requested lines."""
    mask = df["sensor_id"].map(sensor_line_map).isin(lines)
    filtered = df[mask].reset_index(drop=True)
    if len(filtered) == 0:
        available = sorted(set(sensor_line_map.values()))
        raise ValueError(
            f"No sensors found for lines {lines}.\n"
            f"Available lines at this site: {available}"
        )
    return filtered


# ---------------------------------------------------------------------------
# Auto-discovery helpers
# ---------------------------------------------------------------------------


def _latest_build(root: Path, pattern: str) -> Path:
    builds = sorted(root.glob(pattern), key=lambda p: p.name)
    if not builds:
        raise FileNotFoundError(f"No {pattern} builds found in {root}")
    return builds[-1]


def _resolve_v4_build() -> Path:
    if XGB_V4_BUILD_FOLDER:
        return Path(XGB_V4_BUILD_FOLDER)
    return _latest_build(Path(MODELS_ROOT), "xgb_v004_*")


def _resolve_v6_build() -> Path:
    if XGB_V6_BUILD_FOLDER:
        return Path(XGB_V6_BUILD_FOLDER)
    return _latest_build(Path(MODELS_ROOT), "xgb_v006_*")


def _resolve_v8_build() -> Path:
    if XGB_V8_BUILD_FOLDER:
        return Path(XGB_V8_BUILD_FOLDER)
    return _latest_build(Path(MODELS_ROOT), "xgb_v008_*")


def _latest_parquet_v4() -> Path:
    return _latest_build(Path(PARQUET_V4_ROOT), "parquet_v004_*")


# ---------------------------------------------------------------------------
# Feature engineering helpers (must match training-time transformations)
# ---------------------------------------------------------------------------


def _engineer_v4_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as during XGBoost v4 training."""
    df = df.copy()
    dist_col = (
        "effective_distance_to_active_track_m"
        if "effective_distance_to_active_track_m" in df.columns
        else "acc_distance_to_track_m"
    )
    d = df[dist_col].clip(lower=0.0)
    df["feat_log1p_distance"] = np.log1p(d)
    df["feat_inv_distance_sq"] = 1.0 / (d**2 + 1.0)
    return df


def _add_distance_physics_features(
    df: pd.DataFrame, n_global: float, r0: float
) -> pd.DataFrame:
    """Add physics-derived distance features (used in v8 stage-2)."""
    df = df.copy()
    dist_col = (
        "effective_distance_to_active_track_m"
        if "effective_distance_to_active_track_m" in df.columns
        else "acc_distance_to_track_m"
    )
    r = df[dist_col].clip(lower=0.1).values
    df["feat_log_r_ratio"] = np.log(r / r0)
    df["feat_inv_r"] = 1.0 / r
    df["feat_inv_sqrt_r"] = 1.0 / np.sqrt(r)
    df["feat_inv_r2"] = 1.0 / (r**2)
    df["feat_n_log_r_ratio"] = -n_global * np.log(r / r0)
    return df


def _align_to_model(X: pd.DataFrame, model: xgb.XGBRegressor) -> pd.DataFrame:
    """Select and reorder columns to match what the model was trained on."""
    expected = list(model.feature_names_in_)
    missing = [c for c in expected if c not in X.columns]
    if missing:
        raise ValueError(
            f"Feature matrix is missing columns expected by model: {missing}\n"
            f"Available: {list(X.columns)}"
        )
    return X[expected]


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------


def predict_v4(df_sensor: pd.DataFrame, build_dir: Path) -> np.ndarray:
    """XGBoost v4 — direct sensor-level log1p(PGV) prediction.

    Returns predicted PGV in mm/s (one value per sensor row, same order as df_sensor).
    """
    model = xgb.XGBRegressor()
    model.load_model(str(build_dir / "model_final.ubj"))

    X = _engineer_v4_features(df_sensor)
    X = _align_to_model(X, model)
    log1p_pred = model.predict(X)
    return np.expm1(log1p_pred)


def predict_v6(
    df_sensor: pd.DataFrame, df_event: pd.DataFrame, build_dir: Path, v4_dir: Path
) -> np.ndarray:
    """XGBoost v6 — predict c_i then reconstruct PGV via physics formula.

    Returns predicted PGV in mm/s per sensor row.
    """
    # Load model and physics parameters
    model = xgb.XGBRegressor()
    model.load_model(str(build_dir / "model_final.ubj"))

    with open(v4_dir / "global_fit_summary.json") as f:
        fit = json.load(f)
    n_global = fit["n_global"]
    r0 = fit["r0_m"]

    # Event-level feature matrix
    X_event = df_event.copy()
    # Drop columns that are not features (same logic as training)
    drop_cols = [
        "event_id",
        "site_id",
        "c_i",
        "n_i",
        "n_global",
        "n_sensors_used",
        "rmse_log_fit",
        "r2_fit",
        "quality_flag",
        "rmse_log",
        "r2",
        "fit_status",
        "train_type",
        "train_type_family",
        "sensor_id",
        "acc_distance_to_track_m",
        "target_pgv_z_mms",
        "acc_side_of_track",
        "effective_distance_to_active_track_m",
    ]
    X_event = X_event.drop(columns=[c for c in drop_cols if c in X_event.columns])
    X_event = X_event.select_dtypes(include=[np.number])
    X_event = _align_to_model(X_event, model)

    # Predict c_i (one scalar for the whole event)
    c_hat = float(model.predict(X_event)[0])

    # Reconstruct per-sensor PGV using effective distance to active track
    dist_col = (
        "effective_distance_to_active_track_m"
        if "effective_distance_to_active_track_m" in df_sensor.columns
        else "acc_distance_to_track_m"
    )
    r = df_sensor[dist_col].values.astype(float)
    log_pgv_pred = c_hat - n_global * np.log(np.maximum(r, 0.1) / r0)
    return np.exp(log_pgv_pred)


def predict_v8(
    df_sensor: pd.DataFrame, df_event: pd.DataFrame, build_dir: Path, variant: str
) -> np.ndarray:
    """XGBoost v8 — two-stage residual-on-physics.

    variant: "A" (direct log-PGV) or "B" (explicit residual).
    Returns predicted PGV in mm/s per sensor row.
    """
    assert variant in ("A", "B"), f"Unknown v8 variant: {variant}"

    # Load models
    s1_model = xgb.XGBRegressor()
    s1_model.load_model(str(build_dir / "model_stage1_final.ubj"))

    s2_model = xgb.XGBRegressor()
    s2_filename = f"model_stage2_var{variant}_final.ubj"
    s2_model.load_model(str(build_dir / s2_filename))

    # Load physics parameters
    with open(build_dir / "final_physics.json") as f:
        physics = json.load(f)
    n_global = physics["n_global"]
    r0 = physics["r0_m"]

    # ---------- Stage 1: event-level c_hat ----------
    drop_cols_s1 = [
        "event_id",
        "site_id",
        "c_i",
        "n_i",
        "n_global",
        "n_sensors_used",
        "rmse_log_fit",
        "r2_fit",
        "quality_flag",
        "rmse_log",
        "r2",
        "fit_status",
        "train_type",
        "train_type_family",
        "sensor_id",
        "acc_distance_to_track_m",
        "target_pgv_z_mms",
        "acc_side_of_track",
        "effective_distance_to_active_track_m",
    ]
    X_event = df_event.drop(columns=[c for c in drop_cols_s1 if c in df_event.columns])
    X_event = X_event.select_dtypes(include=[np.number])
    X_event = _align_to_model(X_event, s1_model)
    c_hat = float(s1_model.predict(X_event)[0])

    # ---------- Stage 2: sensor-level ----------
    dist_col = (
        "effective_distance_to_active_track_m"
        if "effective_distance_to_active_track_m" in df_sensor.columns
        else "acc_distance_to_track_m"
    )
    r = df_sensor[dist_col].clip(lower=0.1).values.astype(float)
    z_phys = c_hat - n_global * np.log(r / r0)

    # Build sensor feature matrix
    X_sensor = df_sensor.copy()

    # Encode acc_side_of_track if it's a string
    if "acc_side_of_track" in X_sensor.columns:
        if X_sensor["acc_side_of_track"].dtype == object:
            X_sensor["acc_side_of_track"] = (
                X_sensor["acc_side_of_track"].astype("category").cat.codes
            )

    X_sensor["c_hat"] = c_hat
    X_sensor["z_phys"] = z_phys
    X_sensor = _add_distance_physics_features(X_sensor, n_global, r0)
    X_sensor = X_sensor.drop(
        columns=[
            c
            for c in [
                "event_id",
                "site_id",
                "sensor_id",
                "target_pgv_z_mms",
                "train_type",
                "train_type_family",
            ]
            if c in X_sensor.columns
        ]
    )
    X_sensor = X_sensor.select_dtypes(include=[np.number])
    X_sensor = _align_to_model(X_sensor, s2_model)

    stage2_pred = s2_model.predict(X_sensor)

    if variant == "A":
        return np.exp(stage2_pred)
    else:  # Variant B: stage2 predicts residual ε
        return np.exp(z_phys + stage2_pred)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_table(
    df_sensor: pd.DataFrame, pgv_pred: np.ndarray, model_label: str
) -> None:
    rows = []
    for i, (_, row) in enumerate(df_sensor.iterrows()):
        real = float(row["target_pgv_z_mms"])
        pred = float(pgv_pred[i])
        err = pred - real
        pct = 100 * err / real if real > 0 else float("nan")
        rows.append(
            {
                "sensor_id": str(row.get("sensor_id", f"s{i}")),
                "line": str(row.get("sensor_line", "?")),
                "distance (m)": f"{row.get('effective_distance_to_active_track_m', row.get('acc_distance_to_track_m', float('nan'))):.1f}",
                "real (mm/s)": f"{real:.3f}",
                f"{model_label} pred (mm/s)": f"{pred:.3f}",
                "error (mm/s)": f"{err:+.3f}",
                "error (%)": f"{pct:+.1f}%",
            }
        )
    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))

    real_arr = df_sensor["target_pgv_z_mms"].values.astype(float)
    rmse = np.sqrt(np.mean((pgv_pred - real_arr) ** 2))
    mae = np.mean(np.abs(pgv_pred - real_arr))
    print(f"\n  RMSE = {rmse:.4f} mm/s  |  MAE = {mae:.4f} mm/s  (this event only)")


def _plot_results(
    df_sensor: pd.DataFrame, predictions: dict, event_id: str, out_path: Path
) -> None:
    """Plot real measurements and all model predictions vs distance."""
    r_real = (
        df_sensor["effective_distance_to_active_track_m"].values.astype(float)
        if "effective_distance_to_active_track_m" in df_sensor.columns
        else df_sensor["acc_distance_to_track_m"].values.astype(float)
    )
    pgv_real = df_sensor["target_pgv_z_mms"].values.astype(float)

    # Sort by distance for clean lines
    sort_idx = np.argsort(r_real)
    r_sorted = r_real[sort_idx]
    pgv_sorted = pgv_real[sort_idx]

    colors = {
        "xgb_v4": "steelblue",
        "xgb_v6": "darkorange",
        "xgb_v8_A": "forestgreen",
        "xgb_v8_B": "purple",
    }
    markers = {"xgb_v4": "o", "xgb_v6": "s", "xgb_v8_A": "^", "xgb_v8_B": "D"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (ax, use_log) in enumerate(zip(axes, [False, True])):
        # Real measurements
        ax.scatter(
            r_real,
            pgv_real,
            color="black",
            s=80,
            zorder=5,
            label="Real (accelerometer)",
            marker="x",
            linewidths=2,
        )

        for label, pgv_pred in predictions.items():
            pgv_pred_sorted = pgv_pred[sort_idx]
            c = colors.get(label, "grey")
            m = markers.get(label, "o")
            ax.plot(
                r_sorted,
                pgv_pred_sorted,
                color=c,
                lw=2,
                marker=m,
                markersize=6,
                label=label,
                alpha=0.85,
            )

        if use_log:
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title("Log-log scale")
        else:
            ax.set_title("Linear scale")

        ax.set_xlabel("Distance to track (m)")
        ax.set_ylabel("PGV (mm/s)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"PGV prediction vs real measurements\nevent_id: {event_id}", fontsize=12
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"\nPlot saved: {out_path}")

    # Try to show interactively (only works when a display is available)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import matplotlib

            matplotlib.use("TkAgg")
            import matplotlib.pyplot as _plt

            _plt.show()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:

    # ------------------------------------------------------------------
    # Input mode dispatch
    # ------------------------------------------------------------------
    if INPUT_MODE == "fo_file":
        raise NotImplementedError(
            "fo_file mode is not yet implemented.\n"
            "It requires running the full FO feature-extraction pipeline:\n"
            "  1. Read the raw FO signal file (TDMS / HDF5 / NPY)\n"
            "  2. Detect train event windows\n"
            "  3. Compute 1/3-octave band statistics (fo_oct_*_mean/max/std)\n"
            "  4. Compute time-domain stats (fo_td_*)\n"
            "  5. Pass the resulting feature row through the selected model\n"
            "See src/erju/ for the FO processing modules."
        )

    if INPUT_MODE != "event_id":
        raise ValueError(f"Unknown INPUT_MODE: {INPUT_MODE!r}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading v2 Parquet ...")
    _v2_path = Path(V2_PARQUET)
    if not _v2_path.exists():
        _v2_root = Path(PARQUET_V4_ROOT)
        _v2_builds = sorted(_v2_root.glob("parquet_v002_*"), key=lambda p: p.name)
        if not _v2_builds:
            raise FileNotFoundError("No parquet_v002_* builds found.")
        _v2_path = _v2_builds[-1] / "dataset.parquet"
        print(f"  (auto-discovered: {_v2_path})")
    df_v2 = pd.read_parquet(_v2_path)
    if "effective_distance_to_active_track_m" not in df_v2.columns:
        from src.utils.geometry_utils import apply_corrected_distances

        df_v2 = apply_corrected_distances(df_v2)
        print("  Distance correction applied from holten.json.")

    if PRINT_AVAILABLE_EVENTS:
        events = sorted(df_v2["event_id"].unique())
        print(f"\nAvailable event IDs ({len(events)} total):")
        for e in events:
            print(f"  {e}")
        return

    # ------------------------------------------------------------------
    # Select event
    # ------------------------------------------------------------------
    all_events = sorted(df_v2["event_id"].unique())
    if EVENT_ID is not None:
        if EVENT_ID not in set(all_events):
            raise ValueError(
                f"event_id {EVENT_ID!r} not found in dataset.\n"
                f"Set PRINT_AVAILABLE_EVENTS = True to list all valid IDs."
            )
        selected_event = EVENT_ID
    else:
        # Default: pick the event with the most sensors (most informative)
        counts = df_v2.groupby("event_id").size()
        selected_event = counts.idxmax()
        print(
            f"EVENT_ID not set — auto-selected: {selected_event!r} "
            f"({counts[selected_event]} sensor rows)"
        )

    df_sensor = df_v2[df_v2["event_id"] == selected_event].copy()
    df_sensor = df_sensor.dropna(subset=["acc_distance_to_track_m", "target_pgv_z_mms"])
    df_sensor = df_sensor.sort_values("acc_distance_to_track_m").reset_index(drop=True)

    # --- Line filter --------------------------------------------------------
    sensor_line_map = _build_sensor_line_map(SITE_JSON)

    # Annotate each row with its line letter (for display)
    df_sensor["sensor_line"] = df_sensor["sensor_id"].map(sensor_line_map).fillna("?")

    if SENSOR_LINES is not None:
        df_sensor = _filter_sensors_by_line(df_sensor, SENSOR_LINES, sensor_line_map)
        line_label = f"lines {SENSOR_LINES}"
    else:
        line_label = "all lines"

    # --- Side filter --------------------------------------------------------
    if SENSOR_SIDE is not None:
        mask_side = df_sensor["acc_side_of_track"].isin(SENSOR_SIDE)
        df_sensor = df_sensor[mask_side].reset_index(drop=True)
        side_label = f"side {SENSOR_SIDE}"
        if len(df_sensor) == 0:
            available_sides = sorted(df_v2["acc_side_of_track"].unique().tolist())
            raise ValueError(
                f"No sensors found for side(s) {SENSOR_SIDE} "
                f"(after line filter).\n"
                f"Available sides in dataset: {available_sides}"
            )
    else:
        side_label = "all sides"

    print(f"\nEvent      : {selected_event}")
    print(f"Lines used : {line_label}  |  Side: {side_label}")
    print(
        f"Sensors    : {len(df_sensor)}  "
        f"({', '.join(df_sensor['sensor_id'].tolist())})"
    )
    print(f"Distances  : {sorted(df_sensor['acc_distance_to_track_m'].unique())} m")
    print(f"Real PGVs  : {df_sensor['target_pgv_z_mms'].values.round(3)} mm/s")

    # For v6 / v8: load event-level row from v4 dataset
    df_event = None
    if MODEL in ("xgb_v6", "xgb_v8_A", "xgb_v8_B"):
        v4_parquet_dir = _latest_build(Path(PARQUET_V4_ROOT), "parquet_v004_*")
        df_v4 = pd.read_parquet(v4_parquet_dir / "dataset.parquet")
        matches = df_v4[df_v4["event_id"] == selected_event]
        if len(matches) == 0:
            raise ValueError(
                f"Event {selected_event!r} not found in v4 Parquet.\n"
                f"v4 Parquet only contains events processed in that build.\n"
                f"Try a different event_id from v4, or use MODEL='xgb_v4'."
            )
        df_event = matches.iloc[[0]]

    # ------------------------------------------------------------------
    # Run selected model(s)
    # ------------------------------------------------------------------
    print(f"\nRunning model: {MODEL}")
    predictions = {}

    if MODEL == "xgb_v4":
        build = _resolve_v4_build()
        print(f"Build: {build.name}")
        predictions["xgb_v4"] = predict_v4(df_sensor, build)

    elif MODEL == "xgb_v6":
        v4_parquet_dir = _latest_build(Path(PARQUET_V4_ROOT), "parquet_v004_*")
        build = _resolve_v6_build()
        print(f"Build: {build.name}")
        predictions["xgb_v6"] = predict_v6(df_sensor, df_event, build, v4_parquet_dir)

    elif MODEL == "xgb_v8_A":
        build = _resolve_v8_build()
        print(f"Build: {build.name}")
        predictions["xgb_v8_A"] = predict_v8(df_sensor, df_event, build, variant="A")

    elif MODEL == "xgb_v8_B":
        build = _resolve_v8_build()
        print(f"Build: {build.name}")
        predictions["xgb_v8_B"] = predict_v8(df_sensor, df_event, build, variant="B")

    elif MODEL == "all":
        # Run all models and overlay on the same plot
        print("Running all models ...")
        v4_build = _resolve_v4_build()
        predictions["xgb_v4"] = predict_v4(df_sensor, v4_build)

        v4_parquet_dir = _latest_build(Path(PARQUET_V4_ROOT), "parquet_v004_*")
        if df_event is None:
            df_v4 = pd.read_parquet(v4_parquet_dir / "dataset.parquet")
            matches = df_v4[df_v4["event_id"] == selected_event]
            if len(matches) > 0:
                df_event = matches.iloc[[0]]

        if df_event is not None:
            v6_build = _resolve_v6_build()
            predictions["xgb_v6"] = predict_v6(
                df_sensor, df_event, v6_build, v4_parquet_dir
            )

            v8_build = _resolve_v8_build()
            predictions["xgb_v8_A"] = predict_v8(
                df_sensor, df_event, v8_build, variant="A"
            )
            predictions["xgb_v8_B"] = predict_v8(
                df_sensor, df_event, v8_build, variant="B"
            )
        else:
            print("  (v6/v8 skipped: event not in v4 dataset)")

    else:
        raise ValueError(
            f"Unknown MODEL: {MODEL!r}\n"
            f"Valid options: 'xgb_v4', 'xgb_v6', 'xgb_v8_A', 'xgb_v8_B', 'all'"
        )

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    for model_label, pgv_pred in predictions.items():
        print(f"\n--- {model_label} ---")
        _print_table(df_sensor, pgv_pred, model_label)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    out_plot = Path("predict_pgv_output.png")
    _plot_results(df_sensor, predictions, selected_event, out_plot)


if __name__ == "__main__":
    main()
