"""XGBoost v003 demo — predict PGV_z at a chosen distance for a test event.

Usage:
  1. Edit the USER INPUT block below
  2. Run:  python src/ml/demo_predict_pgvz.py
  3. The result is printed to the console and a PNG is saved to the model plots folder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ml.xgb_utils import engineer_features, prepare_features

# ---------------------------------------------------------------------------
# Config — points to v003 artifacts
# ---------------------------------------------------------------------------
BUILD_DIR = Path(r"P:\11210978-erju-ai\holten_models\xgb_v003_20260406_194334")
PARQUET = Path(
    r"P:\11210978-erju-ai\holten_parquet\parquet_v001_20260406_031129\dataset.parquet"
)
OUTPUT_DIR = BUILD_DIR / "plots"

AVAILABLE_DISTANCES_M = [2.0, 4.0, 5.0, 8.0, 16.0, 25.0]
DIST_SENSOR_MAP = {
    2.0: ["MP3", "MP4"],
    4.0: ["MP5", "MP6", "MP7", "MP8", "MP9"],
    5.0: ["MP12", "MP13"],
    8.0: ["MP10"],
    16.0: ["MP1"],
    25.0: ["MP2"],
}

# Known geometry for each free-field sensor (used when SENSOR_ID is set)
SENSOR_GEOMETRY: dict[str, dict] = {
    "MP1": {"distance_m": 16.0, "side": -1},
    "MP2": {"distance_m": 25.0, "side": -1},
    "MP3": {"distance_m": 2.0, "side": 1},
    "MP4": {"distance_m": 2.0, "side": -1},
    "MP5": {"distance_m": 4.0, "side": 1},
    "MP6": {"distance_m": 4.0, "side": 1},
    "MP7": {"distance_m": 4.0, "side": -1},
    "MP8": {"distance_m": 4.0, "side": -1},
    "MP9": {"distance_m": 4.0, "side": -1},
    "MP10": {"distance_m": 8.0, "side": -1},
    "MP12": {"distance_m": 5.0, "side": -1},
    "MP13": {"distance_m": 5.0, "side": -1},
}

# ===========================================================================
# USER INPUT — edit these values, then run the script
# ===========================================================================

EVENT_NUMBER = 10  # integer 1–254  (held-out test events only)

DISTANCE_M = 5.0  # distance to track in metres
# available: 2.0 | 4.0 | 5.0 | 8.0 | 16.0 | 25.0

TRACK_SIDE = -1  # side of track: -1 = left, 0 = unknown, +1 = right

SENSOR_ID = None  # optional: e.g. "MP12"  — if set, overrides DISTANCE_M
# and TRACK_SIDE with that sensor's known geometry.
# Set to None to use DISTANCE_M + TRACK_SIDE manually.

# ===========================================================================


# ---------------------------------------------------------------------------
# Load artifacts once
# ---------------------------------------------------------------------------


def _load_artifacts():
    cfg = json.loads((BUILD_DIR / "config_snapshot.json").read_text())
    mfst = json.loads((BUILD_DIR / "split_manifest.json").read_text())

    model = xgb.XGBRegressor()
    model.load_model(str(BUILD_DIR / "model_final.ubj"))

    df_full = pd.read_parquet(PARQUET)
    exclude = cfg.get("exclude_sensor_ids", [])
    df_full = df_full[~df_full["sensor_id"].isin(exclude)]

    test_ids = set(mfst["test_event_ids"])
    test_df = df_full[df_full["event_id"].isin(test_ids)].copy()

    events_sorted = sorted(test_df["event_id"].unique())
    return cfg, model, test_df, events_sorted


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def _get_fo_feature_row(test_df: pd.DataFrame, event_id: str, cfg: dict) -> pd.Series:
    """Return one feature row for the event (FO features only; distance will be overwritten)."""
    feat_cfg = cfg.get("features", {})
    identifier_cols = feat_cfg.get(
        "identifier_cols", ["event_id", "site_id", "sensor_id"]
    )
    string_cols = feat_cfg.get("string_cols", ["train_type"])
    target_col = feat_cfg.get("target_col", "target_pgv_z_mms")

    event_rows = test_df[test_df["event_id"] == event_id].reset_index(drop=True)

    # Use the first sensor row as the FO feature template
    # (FO features are identical across all sensors for the same event)
    drop_cols = set(identifier_cols) | set(string_cols) | {target_col}
    X = event_rows.drop(columns=[c for c in drop_cols if c in event_rows.columns])
    return X.iloc[0].copy(), event_rows


def _predict_at_distance(
    feature_row: pd.Series,
    distance_m: float,
    track_side: int,
    cfg: dict,
    model: xgb.XGBRegressor,
) -> float:
    """Overwrite distance and side in the feature row and predict PGV_z (mm/s)."""
    row = feature_row.copy()
    row["acc_distance_to_track_m"] = distance_m
    row["acc_side_of_track"] = track_side

    X = pd.DataFrame([row])

    class _FECfg:
        pass

    fe = _FECfg()
    fe_dict = cfg.get("fe", {})
    fe.log_transform_target = fe_dict.get("log_transform_target", False)
    fe.add_geometry_features = fe_dict.get("add_geometry_features", False)

    X = engineer_features(X, fe)
    y_pred_model = model.predict(X)[0]

    if fe.log_transform_target:
        return float(np.expm1(y_pred_model))
    return float(y_pred_model)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot(
    event_id: str,
    event_rows: pd.DataFrame,
    chosen_dist: float,
    track_side: int,
    sensor_label: str,
    predicted: float,
    output_dir: Path,
) -> Path:

    side_str = {-1: "left", 0: "unknown", 1: "right"}.get(track_side, str(track_side))

    # Actual values per sensor, sorted by distance
    actual = (
        event_rows.groupby("acc_distance_to_track_m")["target_pgv_z_mms"]
        .mean()
        .reset_index()
        .sort_values("acc_distance_to_track_m")
    )
    all_dists = actual["acc_distance_to_track_m"].values
    all_actuals = actual["target_pgv_z_mms"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        all_dists,
        all_actuals,
        s=90,
        zorder=5,
        color="#4878CF",
        label="Actual PGV$_z$ (measured sensors)",
        edgecolors="white",
        linewidths=0.6,
    )
    ax.plot(all_dists, all_actuals, "--", color="#4878CF", lw=1.0, alpha=0.5)
    ax.scatter(
        chosen_dist,
        predicted,
        s=160,
        zorder=6,
        color="#D65F5F",
        marker="*",
        label=f"Model prediction @ {chosen_dist:.0f} m, side={side_str} → {predicted:.2f} mm/s",
        edgecolors="white",
        linewidths=0.6,
    )
    ax.axvline(chosen_dist, color="#D65F5F", lw=0.8, ls=":", alpha=0.6)

    mask = np.isclose(all_dists, chosen_dist)
    if mask.any():
        actual_at_dist = all_actuals[mask][0]
        ax.annotate(
            f"Actual = {actual_at_dist:.2f} mm/s",
            xy=(chosen_dist, actual_at_dist),
            xytext=(chosen_dist + 0.8, actual_at_dist + 0.2),
            fontsize=8,
            color="#4878CF",
            arrowprops=dict(arrowstyle="->", color="#4878CF", lw=0.8),
        )

    ax.set_xlabel("Distance to track (m)", fontsize=10)
    ax.set_ylabel("PGV$_z$ (mm/s)", fontsize=10)
    ax.set_title(
        f"PGV$_z$ attenuation — Event #{sensor_label} ({event_id})\n"
        f"Prediction: {chosen_dist:.0f} m, side={side_str} → {predicted:.2f} mm/s",
        fontsize=10,
    )
    ax.set_xlim(0, max(all_dists.max(), chosen_dist) * 1.1 + 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_event = event_id.replace(":", "-").replace("/", "-")
    side_tag = {-1: "L", 0: "U", 1: "R"}.get(track_side, str(track_side))
    out_path = (
        output_dir / f"demo_pgvz_{safe_event}_d{int(chosen_dist)}m_s{side_tag}.png"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main — run once with the USER INPUT variables above
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 60)
    print("  XGBoost v003 — PGV_z prediction demo")
    print("=" * 60)
    print("Loading model and test data ...")
    cfg, model, test_df, events = _load_artifacts()
    n_events = len(events)

    # --- Resolve inputs ---
    event_number = EVENT_NUMBER
    distance_m = DISTANCE_M
    track_side = TRACK_SIDE
    sensor_id = SENSOR_ID

    if sensor_id is not None:
        if sensor_id not in SENSOR_GEOMETRY:
            raise ValueError(
                f"SENSOR_ID '{sensor_id}' not recognised. "
                f"Valid options: {sorted(SENSOR_GEOMETRY)}"
            )
        distance_m = SENSOR_GEOMETRY[sensor_id]["distance_m"]
        track_side = SENSOR_GEOMETRY[sensor_id]["side"]
        print(
            f"  SENSOR_ID={sensor_id} → distance={distance_m} m, "
            f"side={track_side} (overrides DISTANCE_M and TRACK_SIDE)"
        )

    # --- Validate ---
    if not (1 <= event_number <= n_events):
        raise ValueError(f"EVENT_NUMBER must be 1–{n_events}, got {event_number}")
    if distance_m not in AVAILABLE_DISTANCES_M:
        raise ValueError(
            f"DISTANCE_M={distance_m} not in available distances: {AVAILABLE_DISTANCES_M}"
        )
    if track_side not in (-1, 0, 1):
        raise ValueError(f"TRACK_SIDE must be -1, 0 or 1, got {track_side}")

    event_id = events[event_number - 1]
    side_str = {-1: "left", 0: "unknown", 1: "right"}.get(track_side, str(track_side))
    sensor_label = sensor_id if sensor_id else f"#{event_number}"

    print(f"\n  Event   : #{event_number}  →  {event_id}")
    print(f"  Distance: {distance_m} m")
    print(f"  Side    : {track_side} ({side_str})")
    if sensor_id:
        print(f"  Sensor  : {sensor_id}")

    # --- Predict ---
    feature_row, event_rows = _get_fo_feature_row(test_df, event_id, cfg)
    predicted = _predict_at_distance(feature_row, distance_m, track_side, cfg, model)

    # --- Actual value(s) at chosen distance ---
    actual_rows = event_rows[
        np.isclose(event_rows["acc_distance_to_track_m"], distance_m)
    ]
    if sensor_id is not None and not actual_rows.empty:
        mask_sensor = actual_rows["sensor_id"] == sensor_id
        if mask_sensor.any():
            actual_rows = actual_rows[mask_sensor]

    print(f"\n  ─── Result ──────────────────────────────────────────")
    print(f"  Event          : {event_id}")
    print(f"  Distance       : {distance_m} m  |  Side: {track_side} ({side_str})")
    print(f"  Predicted PGVz : {predicted:.3f} mm/s")
    if not actual_rows.empty:
        for _, r in actual_rows.iterrows():
            err = predicted - r["target_pgv_z_mms"]
            print(
                f"  Actual PGVz    : {r['target_pgv_z_mms']:.3f} mm/s  "
                f"(sensor {r['sensor_id']},  error = {err:+.3f} mm/s)"
            )
    else:
        print(f"  Actual PGVz    : no sensor at {distance_m} m in the dataset")
    print(f"  ─────────────────────────────────────────────────────")

    # --- Plot ---
    out_path = _plot(
        event_id,
        event_rows,
        distance_m,
        track_side,
        sensor_label,
        predicted,
        OUTPUT_DIR,
    )
    print(f"\n  Plot saved: {out_path}")


if __name__ == "__main__":
    main()
