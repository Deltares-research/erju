"""Target and residual diagnostics for full dataset and best existing models.

Analyzes:
- PGV distribution by sensor_id, distance, train_type, speed, track_number, crop status
- Residual breakdown by these dimensions
- Top contributors to RMSE
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# ==================================================================================
# CONFIG
# ==================================================================================

PARQUET_PATH = Path(r"P:\11210978-erju-ai\holten_parquet\parquet_v002_20260509_180119\dataset.parquet")
CNN_V2_21CH_BUILD = Path(r"P:\11210978-erju-ai\holten_models\cnn_v002_ch21_modeB_20260625_225138")
OUTPUT_ROOT = Path(r"P:\11210978-erju-ai\holten_models\diagnostics_20260627")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target_pgv_z_mms"
SENSOR_COL = "sensor_id"
EVENT_COL = "event_id"

# ==================================================================================
# LOAD DATA
# ==================================================================================

print("=" * 70)
print("Target and Residual Diagnostics")
print("=" * 70)

df = pd.read_parquet(PARQUET_PATH)
print(f"\nFull dataset: {len(df):,} rows, {df['event_id'].nunique():,} events")

# Clean
df = df[(df[TARGET_COL] > 0) & (df[TARGET_COL].notna())].copy()
print(f"After cleaning: {len(df):,} rows")

# Add bins
df["pgv_bin"] = pd.cut(df[TARGET_COL], bins=[0, 1, 2, 3, 4, 100], labels=["0-1", "1-2", "2-3", "3-4", "4+"])
df["distance_bin"] = pd.cut(df["effective_distance_to_active_track_m"], 
                             bins=[0, 5, 10, 15, 20, 100], 
                             labels=["0-5m", "5-10m", "10-15m", "15-20m", "20+m"])
speed_bins = [0, 10, 20, 30, 40, 50, 100]
df["speed_bin"] = pd.cut(df["train_speed_kmh"], bins=speed_bins, 
                          labels=[f"{speed_bins[i]}-{speed_bins[i+1]}" for i in range(len(speed_bins)-1)])

# ==================================================================================
# TARGET DISTRIBUTIONS
# ==================================================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTIONS")
print("=" * 70)

print("\nPGV distribution by sensor_id:")
sensor_stats = df.groupby(SENSOR_COL)[TARGET_COL].agg(['count', 'mean', 'std', 'min', 'max'])
print(sensor_stats)

print("\nPGV distribution by distance bin:")
dist_stats = df.groupby("distance_bin")[TARGET_COL].agg(['count', 'mean', 'std', 'min', 'max'])
print(dist_stats)

print("\nPGV distribution by train_type:")
if "train_type" in df.columns:
    tt_stats = df.groupby("train_type")[TARGET_COL].agg(['count', 'mean', 'std', 'min', 'max'])
    print(tt_stats)

print("\nPGV distribution by speed bin:")
speed_stats = df.groupby("speed_bin")[TARGET_COL].agg(['count', 'mean', 'std', 'min', 'max'])
print(speed_stats)

print("\nPGV distribution by track_number:")
if "track_number" in df.columns:
    track_stats = df.groupby("track_number")[TARGET_COL].agg(['count', 'mean', 'std', 'min', 'max'])
    print(track_stats)

print("\nPGV distribution by crop status:")
if "was_cropped" in df.columns:
    crop_stats = df.groupby("was_cropped")[TARGET_COL].agg(['count', 'mean', 'std', 'min', 'max'])
    print(crop_stats)

# ==================================================================================
# LOAD PREDICTIONS (if CNN v2 21-ch build exists)
# ==================================================================================

if CNN_V2_21CH_BUILD.exists():
    print(f"\nLoading CNN v2 21-ch predictions from {CNN_V2_21CH_BUILD.name}")
    pred_file = CNN_V2_21CH_BUILD / "predictions.parquet"
    if pred_file.exists():
        df_pred = pd.read_parquet(pred_file)
        df = df.merge(df_pred[["event_id", "sensor_id", "log_pgv_predicted", "pgv_predicted"]], 
                      on=["event_id", "sensor_id"], how="left")
        
        # Compute residuals
        df_pred_rows = df[df["pgv_predicted"].notna()].copy()
        df_pred_rows["residual_pgv"] = df_pred_rows[TARGET_COL] - df_pred_rows["pgv_predicted"]
        df_pred_rows["residual_log_pgv"] = np.log(np.clip(df_pred_rows[TARGET_COL], 1e-6, None)) - df_pred_rows["log_pgv_predicted"]
        
        print(f"\nCNN v2 21-ch residual breakdown by sensor_id:")
        sensor_residuals = df_pred_rows.groupby(SENSOR_COL).agg({
            "residual_pgv": ["count", "mean", "std"],
            TARGET_COL: "mean"
        })
        print(sensor_residuals)
        
        print(f"\nCNN v2 21-ch residual breakdown by distance_bin:")
        dist_residuals = df_pred_rows.groupby("distance_bin").agg({
            "residual_pgv": ["count", "mean", "std"],
            TARGET_COL: "mean"
        })
        print(dist_residuals)
        
        print(f"\nCNN v2 21-ch residual breakdown by pgv_bin:")
        pgv_residuals = df_pred_rows.groupby("pgv_bin").agg({
            "residual_pgv": ["count", "mean", "std"],
            TARGET_COL: "mean"
        })
        print(pgv_residuals)
        
        print(f"\nTop 20 events contributing most to RMSE:")
        event_rmse = df_pred_rows.groupby(EVENT_COL).agg({
            "residual_pgv": lambda x: np.sqrt(np.mean(x**2)),
            "event_id": "count"
        }).rename(columns={"residual_pgv": "rmse", "event_id": "n_rows"})
        event_rmse = event_rmse.sort_values("rmse", ascending=False)
        print(event_rmse.head(20))
        
        print(f"\nTop 20 sensor_ids contributing most to RMSE:")
        sensor_rmse = df_pred_rows.groupby(SENSOR_COL).agg({
            "residual_pgv": lambda x: np.sqrt(np.mean(x**2)),
            "sensor_id": "count"
        }).rename(columns={"residual_pgv": "rmse", "sensor_id": "n_rows"})
        sensor_rmse = sensor_rmse.sort_values("rmse", ascending=False)
        print(sensor_rmse)
        
else:
    print(f"Warning: CNN v2 21-ch build not found at {CNN_V2_21CH_BUILD}")
    df_pred_rows = None

# ==================================================================================
# SAVE SUMMARY
# ==================================================================================

summary = {
    "total_rows": len(df),
    "total_events": df[EVENT_COL].nunique(),
    "pgv_mean": float(df[TARGET_COL].mean()),
    "pgv_std": float(df[TARGET_COL].std()),
    "pgv_min": float(df[TARGET_COL].min()),
    "pgv_max": float(df[TARGET_COL].max()),
    "sensors_unique": int(df[SENSOR_COL].nunique()),
    "has_cnn_predictions": df_pred_rows is not None,
}

(OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nDiagnostics saved to {OUTPUT_ROOT}")
