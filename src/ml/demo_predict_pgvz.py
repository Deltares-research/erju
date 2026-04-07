"""Interactive demo: predict PGV_z at a chosen distance for a test event.

Usage (from project root):
    python src/ml/demo_predict_pgvz.py

The user is prompted to pick:
  - An event number (1-254, from the v003 held-out test set)
  - A distance to track (from the 6 sensor distances available in the data)

The model predicts PGV_z (mm/s) at that distance and compares it to the
real measured values at every sensor for that event.
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
    cfg: dict,
    model: xgb.XGBRegressor,
) -> float:
    """Overwrite distance in the feature row and predict PGV_z (mm/s)."""
    row = feature_row.copy()
    row["acc_distance_to_track_m"] = distance_m

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
    predicted: float,
    output_dir: Path,
) -> Path:

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

    ax.scatter(all_dists, all_actuals, s=90, zorder=5, color="#4878CF",
               label="Actual PGV$_z$ (measured sensors)",
               edgecolors="white", linewidths=0.6)
    ax.plot(all_dists, all_actuals, "--", color="#4878CF", lw=1.0, alpha=0.5)
    ax.scatter(chosen_dist, predicted, s=160, zorder=6, color="#D65F5F", marker="*",
               label=f"Model prediction @ {chosen_dist:.0f} m → {predicted:.2f} mm/s",
               edgecolors="white", linewidths=0.6)
    ax.axvline(chosen_dist, color="#D65F5F", lw=0.8, ls=":", alpha=0.6)

    mask = np.isclose(all_dists, chosen_dist)
    if mask.any():
        actual_at_dist = all_actuals[mask][0]
        ax.annotate(
            f"Actual = {actual_at_dist:.2f} mm/s",
            xy=(chosen_dist, actual_at_dist),
            xytext=(chosen_dist + 0.8, actual_at_dist + 0.2),
            fontsize=8, color="#4878CF",
            arrowprops=dict(arrowstyle="->", color="#4878CF", lw=0.8),
        )

    ax.set_xlabel("Distance to track (m)", fontsize=10)
    ax.set_ylabel("PGV$_z$ (mm/s)", fontsize=10)
    ax.set_title(
        f"PGV$_z$ attenuation — Event: {event_id}\n"
        f"XGBoost v003 prediction at {chosen_dist:.0f} m = {predicted:.2f} mm/s",
        fontsize=10,
    )
    ax.set_xlim(0, max(all_dists.max(), chosen_dist) * 1.1 + 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_event = event_id.replace(":", "-").replace("/", "-")
    out_path = output_dir / f"demo_pgvz_{safe_event}_d{int(chosen_dist)}m.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
        # Accel Z
        t_acc = signals["acc_time"]
        z_acc = signals["acc_z"]
        units = "mm/s" if "velocity" in signals["acc_units"] else "g"
        ax_acc.plot(t_acc, z_acc, lw=0.7, color="#4878CF")
        ax_acc.set_xlabel("Time rel. to event t0 (s)", fontsize=9)
        ax_acc.set_ylabel(f"Velocity Z ({units})", fontsize=9)
        ax_acc.set_title(
            f"Accelerometer Z — {sensor_id} ({chosen_dist:.0f} m)", fontsize=9
        )
        ax_acc.grid(True, alpha=0.3)
        ax_acc.tick_params(labelsize=8)

        # FO centre channel (bandpass filtered)
        t_fo = signals["fo_time"]
        fo_sig = signals["fo_centre_bp"]
        ax_fo.plot(t_fo, fo_sig, lw=0.7, color="#6ACC65")
        ax_fo.set_xlabel("Time rel. to event t0 (s)", fontsize=9)
        ax_fo.set_ylabel("FO strain (ε, bandpass 1–100 Hz)", fontsize=9)
        ax_fo.set_title("FO — centre channel (bandpass filtered)", fontsize=9)
        ax_fo.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax_fo.grid(True, alpha=0.3)
        ax_fo.tick_params(labelsize=8)
    else:
        fig.text(
            0.5,
            0.02,
            "(Raw signal traces not available — NetCDF file not found)",
            ha="center",
            fontsize=8,
            color="grey",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_event = event_id.replace(":", "-").replace("/", "-")
    out_path = output_dir / f"demo_pgvz_{safe_event}_d{int(chosen_dist)}m.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 60)
    print("  XGBoost v003 — PGV_z prediction demo")
    print("=" * 60)
    print("Loading model and test data ...")
    cfg, model, test_df, events = _load_artifacts()
    n_events = len(events)

    print(f"\n  Test set: {n_events} events (held-out, never seen during training)")
    print(f"\n  Available distances: {AVAILABLE_DISTANCES_M} m")
    print("  Sensors per distance:")
    for d, sensors in DIST_SENSOR_MAP.items():
        print(f"    {d:5.1f} m  →  {', '.join(sensors)}")

    while True:
        print("\n" + "-" * 60)

        # --- Event selection ---
        print(f"\n  Enter event number [1–{n_events}]  (or 'q' to quit)")
        raw = input("  Event number: ").strip()
        if raw.lower() == "q":
            print("  Bye!")
            break
        try:
            event_num = int(raw)
            if not (1 <= event_num <= n_events):
                raise ValueError
        except ValueError:
            print(f"  Invalid — please enter a number between 1 and {n_events}.")
            continue

        event_id = events[event_num - 1]
        print(f"  Selected: event #{event_num}  →  {event_id}")

        # --- Distance selection ---
        dist_str = "  |  ".join([f"{d:.0f} m" for d in AVAILABLE_DISTANCES_M])
        print(f"\n  Available distances:  {dist_str}")
        raw_d = input("  Distance (m): ").strip()
        try:
            chosen_dist = float(raw_d)
            if chosen_dist not in AVAILABLE_DISTANCES_M:
                raise ValueError
        except ValueError:
            print(f"  Invalid — please choose from {AVAILABLE_DISTANCES_M}.")
            continue

        # --- Predict ---
        feature_row, event_rows = _get_fo_feature_row(test_df, event_id, cfg)
        predicted = _predict_at_distance(feature_row, chosen_dist, cfg, model)

        # --- Find actual at chosen distance (if sensor exists) ---
        actual_rows = event_rows[
            np.isclose(event_rows["acc_distance_to_track_m"], chosen_dist)
        ]

        print(f"\n  ─── Result ───────────────────────────────────────")
        print(f"  Event          : {event_id}")
        print(f"  Distance       : {chosen_dist:.1f} m")
        print(f"  Predicted PGVz : {predicted:.3f} mm/s")
        if not actual_rows.empty:
            for _, r in actual_rows.iterrows():
                print(
                    f"  Actual PGVz    : {r['target_pgv_z_mms']:.3f} mm/s"
                    f"  (sensor {r['sensor_id']},"
                    f"  error = {predicted - r['target_pgv_z_mms']:+.3f} mm/s)"
                )
        else:
            print(f"  Actual PGVz    : no sensor at this distance in the dataset")
        print(f"  ──────────────────────────────────────────────────")

        # --- Plot ---
        out_path = _plot(
            event_id, event_rows, chosen_dist, predicted, sensor_id, signals, OUTPUT_DIR
        )
        print(f"\n  Plot saved: {out_path}")

        again = input("\n  Try another event? [y/n]: ").strip().lower()
        if again != "y":
            print("  Bye!")
            break


if __name__ == "__main__":
    main()
