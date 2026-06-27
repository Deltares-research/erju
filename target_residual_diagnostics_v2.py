"""Target and residual diagnostics — corrected version."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==================================================================================
# CONFIG
# ==================================================================================

PARQUET_PATH = Path(r"P:\11210978-erju-ai\holten_parquet\parquet_v002_20260509_180119\dataset.parquet")
CNN_V2_21CH_PRED = Path(r"P:\11210978-erju-ai\holten_models\cnn_v002_ch21_modeB_20260625_225138\predictions.parquet")
OUTPUT_ROOT = Path(r"P:\11210978-erju-ai\holten_models\diagnostics_20260627")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target_pgv_z_mms"

# ==================================================================================
# LOAD AND MERGE
# ==================================================================================

print("=" * 70)
print("Target and Residual Diagnostics (CNN v2 21-ch)")
print("=" * 70)

df = pd.read_parquet(PARQUET_PATH)
df_pred = pd.read_parquet(CNN_V2_21CH_PRED)

# Merge to get residuals
df_analysis = df_pred.copy()

# Add bins
df_analysis["pgv_bin"] = pd.cut(df_analysis[TARGET_COL], 
                                 bins=[0, 1, 2, 3, 4, 100], 
                                 labels=["0-1", "1-2", "2-3", "3-4", "4+"])
df_analysis["distance_bin"] = pd.cut(df_analysis["effective_distance_to_active_track_m"], 
                                      bins=[0, 5, 10, 15, 20, 100], 
                                      labels=["0-5m", "5-10m", "10-15m", "15-20m", "20+m"])
df_analysis["pgv_residual"] = df_analysis[TARGET_COL] - df_analysis["pgv_pred_mms"]

print(f"\nAnalysis set: {len(df_analysis):,} rows from {df_analysis['event_id'].nunique():,} events")

# ==================================================================================
# RESIDUAL BY DIMENSION
# ==================================================================================

print("\n" + "=" * 70)
print("RESIDUAL BREAKDOWN")
print("=" * 70)

print("\nResidual by sensor_id (CNN v2 21-ch):")
sensor_res = df_analysis.groupby("sensor_id").agg({
    "pgv_residual": ["count", lambda x: np.sqrt(np.mean(x**2)), "mean", "std"],
    TARGET_COL: "mean"
}).round(4)
sensor_res.columns = ["n", "rmse", "mean_resid", "std_resid", "mean_pgv"]
print(sensor_res)

print("\nResidual by distance_bin (CNN v2 21-ch):")
dist_res = df_analysis.groupby("distance_bin", observed=True).agg({
    "pgv_residual": ["count", lambda x: np.sqrt(np.mean(x**2)), "mean", "std"],
    TARGET_COL: "mean"
}).round(4)
dist_res.columns = ["n", "rmse", "mean_resid", "std_resid", "mean_pgv"]
print(dist_res)

print("\nResidual by pgv_bin (CNN v2 21-ch):")
pgv_res = df_analysis.groupby("pgv_bin", observed=True).agg({
    "pgv_residual": ["count", lambda x: np.sqrt(np.mean(x**2)), "mean", "std"],
    TARGET_COL: ["mean", "std"]
}).round(4)
pgv_res.columns = ["n", "rmse", "mean_resid", "std_resid", "mean_pgv", "std_pgv"]
print(pgv_res)

# ==================================================================================
# TOP CONTRIBUTORS
# ==================================================================================

print("\n" + "=" * 70)
print("TOP CONTRIBUTORS TO RMSE")
print("=" * 70)

print("\nTop 20 events by RMSE:")
event_rmse = df_analysis.groupby("event_id").agg({
    "pgv_residual": lambda x: np.sqrt(np.mean(x**2)),
}).rename(columns={"pgv_residual": "rmse"}).sort_values("rmse", ascending=False)
print(event_rmse.head(20))

print("\nTop 12 sensors by RMSE:")
sensor_rmse = df_analysis.groupby("sensor_id").agg({
    "pgv_residual": lambda x: np.sqrt(np.mean(x**2)),
}).rename(columns={"pgv_residual": "rmse"}).sort_values("rmse", ascending=False)
print(sensor_rmse)

# ==================================================================================
# HIGH-PGV ANALYSIS
# ==================================================================================

print("\n" + "=" * 70)
print("HIGH-PGV ANALYSIS")
print("=" * 70)

df_high_pgv = df_analysis[df_analysis[TARGET_COL] >= 4.0]
df_low_pgv = df_analysis[df_analysis[TARGET_COL] < 2.0]

high_pgv_rmse = np.sqrt(np.mean(df_high_pgv["pgv_residual"]**2))
low_pgv_rmse = np.sqrt(np.mean(df_low_pgv["pgv_residual"]**2))

high_pgv_bias = df_high_pgv["pgv_residual"].mean()
low_pgv_bias = df_low_pgv["pgv_residual"].mean()

print(f"\nHigh-PGV (≥4 mm/s): {len(df_high_pgv)} rows")
print(f"  RMSE = {high_pgv_rmse:.4f} mm/s")
print(f"  Mean residual (bias) = {high_pgv_bias:.4f} mm/s (negative = underprediction)")

print(f"\nLow-PGV (<2 mm/s): {len(df_low_pgv)} rows")
print(f"  RMSE = {low_pgv_rmse:.4f} mm/s")
print(f"  Mean residual (bias) = {low_pgv_bias:.4f} mm/s")

# ==================================================================================
# CROPPED EVENTS ANALYSIS
# ==================================================================================

if "was_cropped" in df_analysis.columns:
    print("\n" + "=" * 70)
    print("CROPPED EVENTS ANALYSIS")
    print("=" * 70)
    
    for cropped_status in [False, True]:
        df_c = df_analysis[df_analysis["was_cropped"] == cropped_status]
        rmse_c = np.sqrt(np.mean(df_c["pgv_residual"]**2))
        bias_c = df_c["pgv_residual"].mean()
        mean_pgv_c = df_c[TARGET_COL].mean()
        
        status_str = "CROPPED" if cropped_status else "NOT-CROPPED"
        print(f"\n{status_str}: {len(df_c)} rows")
        print(f"  Mean PGV = {mean_pgv_c:.4f} mm/s")
        print(f"  RMSE = {rmse_c:.4f} mm/s")
        print(f"  Mean residual (bias) = {bias_c:.4f} mm/s")

print(f"\nDiagnostics complete. Output saved to {OUTPUT_ROOT}")
