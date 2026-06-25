"""Inventory all parquet builds and model builds."""
import json
import pandas as pd
from pathlib import Path

parquet_root = Path(r"P:\11210978-erju-ai\holten_parquet")
models_root = Path(r"P:\11210978-erju-ai\holten_models")

# ── PARQUETS ──────────────────────────────────────────────────────────────────
print("=" * 72)
print("PARQUET BUILDS")
print("=" * 72)
for d in sorted(parquet_root.iterdir()):
    if not d.is_dir():
        continue
    pq = d / "dataset.parquet"
    if not pq.exists():
        print(f"{d.name}: (no dataset.parquet)")
        continue
    df = pd.read_parquet(pq)
    n_events = df["event_id"].nunique() if "event_id" in df.columns else "?"
    n_sensors = df["sensor_id"].nunique() if "sensor_id" in df.columns else "?"
    dist_cols = [c for c in df.columns if "distance" in c]
    has_eff = "effective_distance_to_active_track_m" in df.columns
    print(f"{d.name}")
    print(f"  Rows: {len(df):,}  Cols: {df.shape[1]}  Events: {n_events}  Sensors: {n_sensors}")
    print(f"  Distance cols: {dist_cols}")
    print(f"  Has effective_distance: {has_eff}")
    print()

# ── MODELS ────────────────────────────────────────────────────────────────────
print("=" * 72)
print("MODEL BUILDS")
print("=" * 72)
for d in sorted(models_root.iterdir()):
    if not d.is_dir():
        continue
    sj = d / "summary.json"
    if not sj.exists():
        print(f"{d.name}: (no summary.json)")
        continue
    s = json.loads(sj.read_text())
    # Extract what we can
    info = {}
    for key in ["test_metrics_mms", "test_sensor_metrics", "test", "test_rmse_mms",
                "n_global", "n_global_reference", "n_global_final", "input_parquet",
                "input_parquet_v4", "input_parquet_v4_s2"]:
        if key in s:
            info[key] = s[key]
    print(f"{d.name}")
    # Input parquet
    for k in ["input_parquet", "input_parquet_v4", "input_parquet_v4_s2"]:
        if k in s:
            print(f"  input: {Path(s[k]).parent.name}")
    # n_global
    for k in ["n_global", "n_global_reference", "n_global_final"]:
        if k in s:
            print(f"  {k}: {s[k]:.4f}")
    # Test RMSE
    if "test_metrics_mms" in s:
        m = s["test_metrics_mms"]
        print(f"  test RMSE: {m.get('rmse', m.get('rmse_mms','?')):.4f}  R²: {m.get('r2','?'):.4f}")
    elif "test_sensor_metrics" in s:
        m = s["test_sensor_metrics"]
        print(f"  test RMSE: {m.get('rmse_mms','?'):.4f}  R²: {m.get('r2_mms','?'):.4f}")
    elif "test" in s and isinstance(s["test"], dict):
        t = s["test"]
        if "varA" in t:
            print(f"  test RMSE varA: {t['varA']['rmse']:.4f}  varB: {t['varB']['rmse']:.4f}")
    elif "test_rmse_mms" in s:
        print(f"  test RMSE: {s['test_rmse_mms']:.4f}")
    # n_features
    for k in ["n_features", "feature_count"]:
        if k in s:
            print(f"  n_features: {s[k]}")
    print()
