"""Three-way oracle attenuation check on line-C side -1 subset.

Compares:
1. Track 1 only (r = distance_to_track_1_m)
2. Track 2 only (r = distance_to_track_2_m)
3. Both tracks (r = effective_distance_to_active_track_m)

This validates whether the attenuation curve is track-independent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==================================================================================
# CONFIG
# ==================================================================================

PARQUET_PATH = Path(r"P:\11210978-erju-ai\holten_parquet\parquet_v002_20260509_180119\dataset.parquet")
OUTPUT_ROOT = Path(r"P:\11210978-erju-ai\holten_models\oracle_linec_threeway_20260627")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_SENSORS = ["MP4", "MP8", "MP10", "MP1", "MP2"]
R0 = 10.0
TARGET_COL = "target_pgv_z_mms"
SENSOR_COL = "sensor_id"
EVENT_COL = "event_id"

# Splits
TEST_FRACTION = 0.15
TEST_SEED = 42
VAL_FRACTION = 0.15
VAL_SEED = 43

# ==================================================================================
# LOAD AND FILTER
# ==================================================================================

print("=" * 80)
print("THREE-WAY ORACLE ATTENUATION CHECK — Line C Side -1")
print("=" * 80)

df = pd.read_parquet(PARQUET_PATH)
print(f"\nFull dataset: {len(df):,} rows, {df['event_id'].nunique():,} events")

# Filter to line-C side -1
df_linec = df[df[SENSOR_COL].isin(TARGET_SENSORS)].copy()
print(f"Line-C subset: {len(df_linec):,} rows, {df_linec['event_id'].nunique():,} events")

# Clean targets
df_linec = df_linec[(df_linec[TARGET_COL] > 0) & (df_linec[TARGET_COL].notna())].copy()

# Track breakdown
n_track1 = (df_linec["track_number"] == 1).sum()
n_track2 = (df_linec["track_number"] == 2).sum()
n_unknown = df_linec["track_number"].isna().sum()

print(f"After cleaning: {len(df_linec):,} rows")
print(f"  Track 1 rows: {n_track1:,}")
print(f"  Track 2 rows: {n_track2:,}")
print(f"  Unknown track: {n_unknown:,}")

# ==================================================================================
# ORACLE HELPER FUNCTIONS
# ==================================================================================

def power_law(r, c, n):
    """log(PGV) = c - n*log(r/r0)"""
    return c - n * np.log(r / R0)

def fit_oracle_global_n(df_train, distance_col):
    """Fit single n_global on training events using specified distance column."""
    r = df_train[distance_col].to_numpy(np.float64)
    pgv = df_train[TARGET_COL].to_numpy(np.float64)
    log_pgv = np.log(np.clip(pgv, 1e-6, None))
    
    # Linear regression: log(PGV) = c - n*log(r/r0)
    X = np.column_stack([np.ones_like(r), -np.log(r / R0)])
    coeffs = np.linalg.lstsq(X, log_pgv, rcond=None)[0]
    c_mean, n_global = coeffs
    
    return n_global, c_mean

def predict_oracle_global(df, n_global, distance_col, df_train):
    """Predict using global-n model."""
    # Fit c_i per event on training data
    c_dict = {}
    for event_id in df_train[EVENT_COL].unique():
        df_ev = df_train[df_train[EVENT_COL] == event_id]
        r = df_ev[distance_col].to_numpy(np.float64)
        pgv = df_ev[TARGET_COL].to_numpy(np.float64)
        log_pgv = np.log(np.clip(pgv, 1e-6, None))
        
        if len(r) > 0:
            X = np.column_stack([np.ones_like(r), -np.log(r / R0)])
            coeffs = np.linalg.lstsq(X, log_pgv, rcond=None)[0]
            c_dict[event_id] = coeffs[0]
    
    # Predict
    log_pgv_pred = []
    for idx, row in df.iterrows():
        event_id = row[EVENT_COL]
        r = row[distance_col]
        
        if event_id in c_dict:
            c_i = c_dict[event_id]
        else:
            # Estimate from all rows of this event
            df_ev = df[df[EVENT_COL] == event_id]
            r_ev = df_ev[distance_col].to_numpy(np.float64)
            pgv_ev = df_ev[TARGET_COL].to_numpy(np.float64)
            log_pgv_ev = np.log(np.clip(pgv_ev, 1e-6, None))
            
            if len(r_ev) > 0:
                X_ev = np.column_stack([np.ones_like(r_ev), -np.log(r_ev / R0)])
                coeffs_ev = np.linalg.lstsq(X_ev, log_pgv_ev, rcond=None)[0]
                c_i = coeffs_ev[0]
            else:
                c_i = 0.0
        
        log_pgv_i = power_law(r, c_i, n_global)
        log_pgv_pred.append(log_pgv_i)
    
    return np.array(log_pgv_pred)

def compute_metrics(log_pgv_true, log_pgv_pred):
    """Compute all metrics."""
    pgv_pred = np.exp(np.clip(log_pgv_pred, -30, 30))
    pgv_true = np.exp(log_pgv_true)
    
    rmse_pgv = np.sqrt(mean_squared_error(pgv_true, pgv_pred))
    mae_pgv = mean_absolute_error(pgv_true, pgv_pred)
    rmse_log = np.sqrt(mean_squared_error(log_pgv_true, log_pgv_pred))
    mae_log = mean_absolute_error(log_pgv_true, log_pgv_pred)
    r2_log = r2_score(log_pgv_true, log_pgv_pred)
    
    return {
        "rmse_pgv": rmse_pgv,
        "mae_pgv": mae_pgv,
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "r2_log": r2_log,
    }

# ==================================================================================
# THREE CASES
# ==================================================================================

results_summary = []

for case_name, case_subset, distance_col, track_desc in [
    ("Track 1 only", df_linec[df_linec["track_number"] == 1], "acc_distance_to_track_m", "distance to track 1"),
    ("Track 2 only", df_linec[df_linec["track_number"] == 2], "acc_distance_to_track_2_m", "distance to track 2"),
    ("Both tracks (active)", df_linec[df_linec["track_number"].isin([1, 2])], "effective_distance_to_active_track_m", "distance to active track"),
]:
    
    print(f"\n" + "=" * 80)
    print(f"CASE: {case_name}")
    print("=" * 80)
    
    # Event splits
    events_all = case_subset[EVENT_COL].unique()
    n_events = len(events_all)
    n_rows = len(case_subset)
    
    np.random.seed(TEST_SEED)
    test_events = np.random.choice(events_all, size=int(n_events * TEST_FRACTION), replace=False)
    train_val_events = np.setdiff1d(events_all, test_events)
    
    np.random.seed(VAL_SEED)
    val_events = np.random.choice(train_val_events, size=int(len(train_val_events) * VAL_FRACTION), replace=False)
    train_events = np.setdiff1d(train_val_events, val_events)
    
    case_subset["split"] = "test"
    case_subset.loc[case_subset[EVENT_COL].isin(train_events), "split"] = "train"
    case_subset.loc[case_subset[EVENT_COL].isin(val_events), "split"] = "val"
    
    print(f"\nSubset: {n_rows:,} rows, {n_events:,} events")
    print(f"  Train: {len(train_events)} events, {len(case_subset[case_subset['split'] == 'train']):,} rows")
    print(f"  Val:   {len(val_events)} events, {len(case_subset[case_subset['split'] == 'val']):,} rows")
    print(f"  Test:  {len(test_events)} events, {len(case_subset[case_subset['split'] == 'test']):,} rows")
    print(f"Distance reference: {track_desc}")
    
    # Fit oracle
    df_train = case_subset[case_subset["split"] == "train"]
    n_global, c_mean = fit_oracle_global_n(df_train, distance_col)
    
    print(f"\nOracle parameters (fit on train):")
    print(f"  n_global = {n_global:.4f}")
    print(f"  c_mean (train) = {c_mean:.4f}")
    
    # Predict on each split
    case_metrics = {}
    for split_name in ["train", "val", "test"]:
        df_split = case_subset[case_subset["split"] == split_name]
        if len(df_split) == 0:
            continue
        
        log_pgv_true = np.log(np.clip(df_split[TARGET_COL].to_numpy(np.float64), 1e-6, None))
        log_pgv_pred = predict_oracle_global(df_split, n_global, distance_col, df_train)
        metrics = compute_metrics(log_pgv_true, log_pgv_pred)
        
        print(f"\n{split_name.upper():6s}: RMSE(PGV)={metrics['rmse_pgv']:.4f}  MAE(PGV)={metrics['mae_pgv']:.4f}  R²(log)={metrics['r2_log']:.4f}")
        case_metrics[split_name] = metrics
    
    # Store for comparison
    test_metrics = case_metrics.get("test", {})
    results_summary.append({
        "case": case_name,
        "track_desc": track_desc,
        "n_events": n_events,
        "n_rows": n_rows,
        "n_events_test": len(test_events),
        "n_rows_test": len(case_subset[case_subset['split'] == 'test']),
        "n_global": n_global,
        "rmse_pgv_test": test_metrics.get("rmse_pgv", np.nan),
        "rmse_log_test": test_metrics.get("rmse_log", np.nan),
        "r2_log_test": test_metrics.get("r2_log", np.nan),
    })

# ==================================================================================
# COMPARISON TABLE
# ==================================================================================

print(f"\n" + "=" * 80)
print("COMPARISON TABLE")
print("=" * 80)

comparison_df = pd.DataFrame(results_summary)
print("\n")
print(comparison_df.to_string(index=False))

# ==================================================================================
# SAVE RESULTS
# ==================================================================================

results_dict = {
    "audit_date": "2026-06-27",
    "subset": "line_c_side_minus1",
    "sensors": TARGET_SENSORS,
    "cases": [
        {
            "case": row["case"],
            "track_desc": row["track_desc"],
            "n_events": int(row["n_events"]),
            "n_rows": int(row["n_rows"]),
            "n_events_test": int(row["n_events_test"]),
            "n_rows_test": int(row["n_rows_test"]),
            "n_global": float(row["n_global"]),
            "rmse_pgv_test": float(row["rmse_pgv_test"]),
            "rmse_log_test": float(row["rmse_log_test"]),
            "r2_log_test": float(row["r2_log_test"]),
        }
        for _, row in comparison_df.iterrows()
    ]
}

(OUTPUT_ROOT / "three_way_results.json").write_text(json.dumps(results_dict, indent=2))
print(f"\nResults saved to {OUTPUT_ROOT / 'three_way_results.json'}")
