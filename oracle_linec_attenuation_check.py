"""Oracle attenuation curve fitting on line-C side -1 clean subset.

Fits power-law models: log(PGV_i(r)) = c_i - n*log(r/r_0)

Subset: MP4, MP8, MP10, MP1, MP2 (side -1, line C)
Distances: 2.5m, 4m, 8m, 16m, 23m (verified from geometry)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==================================================================================
# CONFIG
# ==================================================================================

PARQUET_PATH = Path(r"P:\11210978-erju-ai\holten_parquet\parquet_v002_20260509_180119\dataset.parquet")
OUTPUT_ROOT = Path(r"P:\11210978-erju-ai\holten_models\oracle_linec_20260627")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Line-C side -1 sensors
TARGET_SENSORS = ["MP4", "MP8", "MP10", "MP1", "MP2"]
TARGET_DISTANCES = {
    "MP4": 2.5,
    "MP8": 4.0,
    "MP10": 8.0,
    "MP1": 16.0,
    "MP2": 23.0,
}
R0 = 10.0
TARGET_COL = "target_pgv_z_mms"
SENSOR_COL = "sensor_id"
EVENT_COL = "event_id"

# Splits (same as train_cnn_v2)
TEST_FRACTION = 0.15
TEST_SEED = 42
VAL_FRACTION = 0.15
VAL_SEED = 43

# ==================================================================================
# LOAD AND FILTER DATA
# ==================================================================================

print("=" * 70)
print("Oracle Attenuation Check — Line C Side -1")
print("=" * 70)

df = pd.read_parquet(PARQUET_PATH)
print(f"\nFull dataset: {len(df):,} rows, {df['event_id'].nunique():,} events")

# Filter to target sensors
df_linec = df[df[SENSOR_COL].isin(TARGET_SENSORS)].copy()
print(f"Line-C side -1 subset: {len(df_linec):,} rows, {df_linec['event_id'].nunique():,} events")

# Verify all sensors present
missing = set(TARGET_SENSORS) - set(df_linec[SENSOR_COL].unique())
if missing:
    print(f"WARNING: Missing sensors: {missing}")

# Clean targets
df_linec = df_linec[(df_linec[TARGET_COL] > 0) & (df_linec[TARGET_COL].notna())].copy()
print(f"After cleaning targets: {len(df_linec):,} rows, {df_linec['event_id'].nunique():,} events")

# Add distances
df_linec["distance_m"] = df_linec[SENSOR_COL].map(TARGET_DISTANCES)
df_linec = df_linec[df_linec["distance_m"].notna()].copy()

# ==================================================================================
# EVENT-LEVEL SPLITS
# ==================================================================================

events_all = df_linec[EVENT_COL].unique()
n_events = len(events_all)

# Test split
np.random.seed(TEST_SEED)
test_events = np.random.choice(events_all, size=int(n_events * TEST_FRACTION), replace=False)
train_val_events = np.setdiff1d(events_all, test_events)

# Val split (from train_val)
np.random.seed(VAL_SEED)
val_events = np.random.choice(train_val_events, size=int(len(train_val_events) * VAL_FRACTION), replace=False)
train_events = np.setdiff1d(train_val_events, val_events)

print(f"\nEvent splits: train {len(train_events)}, val {len(val_events)}, test {len(test_events)}")

df_linec["split"] = "test"
df_linec.loc[df_linec[EVENT_COL].isin(train_events), "split"] = "train"
df_linec.loc[df_linec[EVENT_COL].isin(val_events), "split"] = "val"

# ==================================================================================
# ORACLE MODELS
# ==================================================================================

def power_law(r, c, n):
    """log(PGV) = c - n*log(r/r0)"""
    return c - n * np.log(r / R0)

def fit_oracle_global_n(df_train):
    """Fit single n_global on training events."""
    r = df_train["distance_m"].to_numpy(np.float64)
    pgv = df_train[TARGET_COL].to_numpy(np.float64)
    log_pgv = np.log(np.clip(pgv, 1e-6, None))
    
    # Fit: log(PGV) = c - n*log(r/r0)
    # Linear regression: log(PGV) = c - n*log(r/r0)
    X = np.column_stack([np.ones_like(r), -np.log(r / R0)])
    coeffs = np.linalg.lstsq(X, log_pgv, rcond=None)[0]
    c_mean, n_global = coeffs
    
    return n_global, c_mean

def fit_oracle_event_specific_n(df_train):
    """Fit n per event; return n_dist (distribution)."""
    ns = []
    cs = []
    for event_id in df_train[EVENT_COL].unique():
        df_ev = df_train[df_train[EVENT_COL] == event_id]
        if len(df_ev) < 2:
            continue
        
        r = df_ev["distance_m"].to_numpy(np.float64)
        pgv = df_ev[TARGET_COL].to_numpy(np.float64)
        log_pgv = np.log(np.clip(pgv, 1e-6, None))
        
        # Fit per event
        X = np.column_stack([np.ones_like(r), -np.log(r / R0)])
        coeffs = np.linalg.lstsq(X, log_pgv, rcond=None)[0]
        c_ev, n_ev = coeffs
        
        ns.append(n_ev)
        cs.append(c_ev)
    
    return np.array(ns), np.array(cs)

# Fit models on training fold
df_train = df_linec[df_linec["split"] == "train"]
n_global, c_mean_train = fit_oracle_global_n(df_train)
n_per_event, c_per_event = fit_oracle_event_specific_n(df_train)

print(f"\nOracle parameters (fit on train):")
print(f"  n_global = {n_global:.4f}")
print(f"  c_mean (train, global-n) = {c_mean_train:.4f}")
print(f"  n_per_event: mean={n_per_event.mean():.4f}, std={n_per_event.std():.4f}")
print(f"  c_per_event: mean={c_per_event.mean():.4f}, std={c_per_event.std():.4f}")

# ==================================================================================
# PREDICT ON EACH SPLIT
# ==================================================================================

def predict_oracle(df, n_global, fit_on_train_df=None):
    """Predict log(PGV) using oracle model."""
    # Fit c_i per event on training data
    c_dict = {}
    if fit_on_train_df is not None:
        for event_id in fit_on_train_df[EVENT_COL].unique():
            df_ev = fit_on_train_df[fit_on_train_df[EVENT_COL] == event_id]
            r = df_ev["distance_m"].to_numpy(np.float64)
            pgv = df_ev[TARGET_COL].to_numpy(np.float64)
            log_pgv = np.log(np.clip(pgv, 1e-6, None))
            
            X = np.column_stack([np.ones_like(r), -np.log(r / R0)])
            coeffs = np.linalg.lstsq(X, log_pgv, rcond=None)[0]
            c_dict[event_id] = coeffs[0]
    
    # Predict on test/val events (use fitted c from train if available, else estimate)
    log_pgv_pred = []
    for idx, row in df.iterrows():
        event_id = row[EVENT_COL]
        r = row["distance_m"]
        
        if event_id in c_dict:
            c_i = c_dict[event_id]
        else:
            # Estimate c_i from all rows of this event
            df_ev = df[df[EVENT_COL] == event_id]
            if len(df_ev) > 1:
                r_ev = df_ev["distance_m"].to_numpy(np.float64)
                pgv_ev = df_ev[TARGET_COL].to_numpy(np.float64)
                log_pgv_ev = np.log(np.clip(pgv_ev, 1e-6, None))
                
                X_ev = np.column_stack([np.ones_like(r_ev), -np.log(r_ev / R0)])
                coeffs_ev = np.linalg.lstsq(X_ev, log_pgv_ev, rcond=None)[0]
                c_i = coeffs_ev[0]
            else:
                c_i = c_per_event.mean()  # fallback
        
        log_pgv_i = power_law(r, c_i, n_global)
        log_pgv_pred.append(log_pgv_i)
    
    return np.array(log_pgv_pred)

# Predictions
for split_name in ["train", "val", "test"]:
    df_split = df_linec[df_linec["split"] == split_name]
    log_pgv_true = np.log(np.clip(df_split[TARGET_COL].to_numpy(np.float64), 1e-6, None))
    log_pgv_pred = predict_oracle(df_split, n_global, fit_on_train_df=df_train)
    pgv_pred = np.exp(np.clip(log_pgv_pred, -30, 30))
    pgv_true = df_split[TARGET_COL].to_numpy(np.float64)
    
    rmse_pgv = np.sqrt(mean_squared_error(pgv_true, pgv_pred))
    mae_pgv = mean_absolute_error(pgv_true, pgv_pred)
    rmse_log = np.sqrt(mean_squared_error(log_pgv_true, log_pgv_pred))
    mae_log = mean_absolute_error(log_pgv_true, log_pgv_pred)
    r2_log = r2_score(log_pgv_true, log_pgv_pred)
    
    print(f"\n{split_name.upper()}: {len(df_split):,} rows")
    print(f"  RMSE(PGV)  = {rmse_pgv:.4f} mm/s")
    print(f"  MAE(PGV)   = {mae_pgv:.4f} mm/s")
    print(f"  RMSE(log)  = {rmse_log:.4f}")
    print(f"  MAE(log)   = {mae_log:.4f}")
    print(f"  R²(log)    = {r2_log:.4f}")

# ==================================================================================
# COMPARISON TABLE
# ==================================================================================

print("\n" + "=" * 70)
print("COMPARISON TABLE")
print("=" * 70)

results = []

# Naive mean baseline
df_test = df_linec[df_linec["split"] == "test"]
pgv_mean = df_train[TARGET_COL].mean()
pgv_test_true = df_test[TARGET_COL].to_numpy(np.float64)
pgv_test_pred = np.full_like(pgv_test_true, pgv_mean)
rmse_mean = np.sqrt(mean_squared_error(pgv_test_true, pgv_test_pred))
results.append(("Naive mean baseline", "all 5", len(df_test), rmse_mean, None))

# Oracle global-n
log_pgv_test_true = np.log(np.clip(pgv_test_true, 1e-6, None))
log_pgv_test_pred = predict_oracle(df_test, n_global, fit_on_train_df=df_train)
pgv_test_pred_oracle = np.exp(np.clip(log_pgv_test_pred, -30, 30))
rmse_oracle_global = np.sqrt(mean_squared_error(pgv_test_true, pgv_test_pred_oracle))
results.append(("Oracle global-n", "all 5", len(df_test), rmse_oracle_global, n_global))

# Oracle event-specific n
# TODO: implement event-specific prediction

print("\nTest set results:")
for name, sensors, n_rows, rmse, n_val in results:
    print(f"  {name:30s}  {sensors:10s}  {n_rows:5d} rows  RMSE={rmse:.4f} mm/s" + 
          (f"  n={n_val:.4f}" if n_val else ""))

# ==================================================================================
# SAVE RESULTS
# ==================================================================================

results_dict = {
    "subset": "line_c_side_minus1",
    "sensors": TARGET_SENSORS,
    "distances": TARGET_DISTANCES,
    "n_events_train": len(train_events),
    "n_events_val": len(val_events),
    "n_events_test": len(test_events),
    "n_rows_total": len(df_linec),
    "oracle_n_global": float(n_global),
    "oracle_c_mean_train": float(c_mean_train),
    "oracle_n_per_event_mean": float(n_per_event.mean()),
    "oracle_n_per_event_std": float(n_per_event.std()),
    "oracle_c_per_event_mean": float(c_per_event.mean()),
    "oracle_c_per_event_std": float(c_per_event.std()),
    "r0": R0,
}

(OUTPUT_ROOT / "results.json").write_text(json.dumps(results_dict, indent=2))
print(f"\nResults saved to {OUTPUT_ROOT / 'results.json'}")

print("\nOracle check complete. Proceed to diagnostic plots and comparisons.")
