"""Oracle attenuation check.

Tests the ceiling of the power-law attenuation representation by using the
*true* fitted curve parameters (c_i, n_i from the actual measured PGVs) to
reconstruct sensor-level PGV, then comparing against the v4 baseline.

Three scenarios evaluated:
  Oracle S1 — use true c_i  +  global n  (same structure as XGBoost v6)
  Oracle S2 — use true c_i  +  true n_i  (same structure as XGBoost v7)
  Mean pred — naive baseline: predict global mean PGV for every row

Key question:
  If oracle RMSE << v4 RMSE (1.79 mm/s) → the power-law representation is
      expressive enough; the gap is an ML prediction problem.
  If oracle RMSE ≈ v4 RMSE → the representation is too rigid; physics-prior
      residual correction is the right next step, not better curve prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.utils.geometry_utils import apply_corrected_distances

# ---------------------------------------------------------------------------
# Paths — auto-discover latest parquet_v004 build
# ---------------------------------------------------------------------------
PARQUET_ROOT = Path(r"P:\11210978-erju-ai\holten_parquet")
# Auto-discover latest v2 build
v2_builds = sorted(PARQUET_ROOT.glob("parquet_v002_*"), key=lambda p: p.name)
if not v2_builds:
    raise FileNotFoundError("No parquet_v002_* builds found.")
V2_PARQUET = v2_builds[-1] / "dataset.parquet"
print(f"Using Parquet v2 build: {v2_builds[-1].name}")

v4_builds = sorted(PARQUET_ROOT.glob("parquet_v004_*"), key=lambda p: p.name)
if not v4_builds:
    raise FileNotFoundError("No parquet_v004_* builds found.")
V4_DIR = v4_builds[-1]
print(f"Using Parquet v4 build: {V4_DIR.name}")

GLOBAL_FIT_JSON = V4_DIR / "global_fit_summary.json"
PER_EVENT_PARQUET = V4_DIR / "attenuation_per_event.parquet"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("\nLoading sensor-level v2 data ...")
df_v2 = pd.read_parquet(
    V2_PARQUET,
    columns=[
        "event_id",
        "sensor_id",
        "track_number",
        "acc_distance_to_track_m",
        "target_pgv_z_mms",
    ],
)
# Apply geometry correction (also adds effective_distance_to_active_track_m)
df_v2 = apply_corrected_distances(df_v2)
df_v2 = df_v2.dropna(
    subset=["effective_distance_to_active_track_m", "target_pgv_z_mms"]
)
df_v2 = df_v2[df_v2["target_pgv_z_mms"] > 0]
df_v2 = df_v2[df_v2["effective_distance_to_active_track_m"] > 0]
print(
    f"  Sensor rows (valid): {len(df_v2):,}  |  Events: {df_v2['event_id'].nunique():,}"
)

print("\nLoading per-event attenuation parameters ...")
df_params = pd.read_parquet(PER_EVENT_PARQUET)
print(f"  Events: {len(df_params):,}  |  Cols: {list(df_params.columns)}")

with open(GLOBAL_FIT_JSON) as f:
    fit_summary = json.load(f)
n_global = fit_summary["n_global"]
r0 = fit_summary["r0_m"]
print(f"\n  n_global = {n_global:.4f}  |  r0 = {r0} m")

# ---------------------------------------------------------------------------
# Join v2 sensor rows with per-event curve parameters
# ---------------------------------------------------------------------------
df = df_v2.merge(
    df_params[["event_id", "c_i", "n_i", "quality_flag"]],
    on="event_id",
    how="inner",
)
print(f"\nAfter join: {len(df):,} sensor rows  |  {df['event_id'].nunique():,} events")

# ---------------------------------------------------------------------------
# Reconstruct PGV from oracle parameters
# ---------------------------------------------------------------------------
r = df["effective_distance_to_active_track_m"].values
c_i = df["c_i"].values
n_i = df["n_i"].values
pgv_true = df["target_pgv_z_mms"].values
log_pgv_true = np.log(pgv_true)

log_r_ratio = np.log(r / r0)

# Oracle Scenario 1: true c_i + global n
log_pgv_s1 = c_i - n_global * log_r_ratio
pgv_s1 = np.exp(log_pgv_s1)

# Oracle Scenario 2: true c_i + true n_i
log_pgv_s2 = c_i - n_i * log_r_ratio
pgv_s2 = np.exp(log_pgv_s2)

# Naive mean baseline
pgv_mean_pred = np.full_like(pgv_true, pgv_true.mean())


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------
def metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    log_true = np.log(np.maximum(y_true, 1e-9))
    log_pred = np.log(np.maximum(y_pred, 1e-9))
    rmse_log = np.sqrt(np.mean((log_true - log_pred) ** 2))

    smape = 100 * np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)
    )

    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"  RMSE      = {rmse:.4f} mm/s")
    print(f"  RMSE(log) = {rmse_log:.4f}")
    print(f"  MAE       = {mae:.4f} mm/s")
    print(f"  R²        = {r2:.4f}")
    print(f"  sMAPE     = {smape:.2f}%")
    return dict(label=label, rmse=rmse, rmse_log=rmse_log, mae=mae, r2=r2, smape=smape)


results = []
results.append(metrics(pgv_true, pgv_mean_pred, "Naive mean baseline"))
results.append(metrics(pgv_true, pgv_s1, "Oracle S1: true c_i + global n"))
results.append(metrics(pgv_true, pgv_s2, "Oracle S2: true c_i + true n_i"))

V4_RMSE = 1.79
print(f"\n{'=' * 55}")
print(f"  XGBoost v4 benchmark (sensor-level direct) = {V4_RMSE} mm/s")
print(f"  Oracle S1 gap vs v4 : {results[1]['rmse'] - V4_RMSE:+.4f} mm/s")
print(f"  Oracle S2 gap vs v4 : {results[2]['rmse'] - V4_RMSE:+.4f} mm/s")
print(f"{'=' * 55}")

# ---------------------------------------------------------------------------
# What fraction of error is representation vs ML?
# ---------------------------------------------------------------------------
print("\n--- Decomposition ---")
print(f"  Representation ceiling (Oracle S1 RMSE)    = {results[1]['rmse']:.4f} mm/s")
print(
    f"  ML gap (v6 test RMSE - Oracle S1 RMSE)     = {2.0650 - results[1]['rmse']:+.4f} mm/s"
)
print(f"  Representation ceiling (Oracle S2 RMSE)    = {results[2]['rmse']:.4f} mm/s")
print(
    f"  ML gap (v7 test RMSE - Oracle S2 RMSE)     = {2.0661 - results[2]['rmse']:+.4f} mm/s"
)

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(".")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = [
    "Naive Mean",
    "Oracle S1 (true c_i, global n)",
    "Oracle S2 (true c_i, true n_i)",
]
preds = [pgv_mean_pred, pgv_s1, pgv_s2]

for ax, title, y_pred, res in zip(axes, titles, preds, results):
    ax.scatter(pgv_true, y_pred, s=4, alpha=0.3, color="steelblue")
    lim = [min(pgv_true.min(), y_pred.min()), max(pgv_true.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Measured PGV (mm/s)")
    ax.set_ylabel("Reconstructed PGV (mm/s)")
    ax.set_title(f"{title}\nRMSE={res['rmse']:.3f} mm/s  R²={res['r2']:.3f}")

plt.tight_layout()
out_path = OUTPUT_DIR / "oracle_scatter.png"
plt.savefig(out_path, dpi=120)
plt.close()
print(f"\nScatter plot saved: {out_path}")

# ---------------------------------------------------------------------------
# Residuals vs distance
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, title, y_pred in zip(axes, ["Oracle S1", "Oracle S2"], [pgv_s1, pgv_s2]):
    resid = np.log(pgv_true) - np.log(y_pred)
    ax.scatter(r, resid, s=4, alpha=0.3, color="steelblue")
    ax.axhline(0, color="red", lw=1, ls="--")
    ax.set_xlabel("Sensor distance (m)")
    ax.set_ylabel("log(PGV_true) - log(PGV_pred)")
    ax.set_title(f"{title} — residuals vs distance")

plt.tight_layout()
out_path2 = OUTPUT_DIR / "oracle_residuals_vs_distance.png"
plt.savefig(out_path2, dpi=120)
plt.close()
print(f"Residual plot saved: {out_path2}")

# ---------------------------------------------------------------------------
# Residuals vs n_i (is the spread in n_i meaningful?)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.hist(df_params["n_i"], bins=60, color="steelblue", edgecolor="white")
ax.axvline(n_global, color="red", lw=2, ls="--", label=f"n_global={n_global:.3f}")
ax.set_xlabel("n_i (per-event exponent)")
ax.set_ylabel("Count")
ax.set_title("Distribution of per-event n_i")
ax.legend()

ax = axes[1]
# residual of S1 vs actual n_i (shows whether knowing n_i would help)
resid_s1_log = np.log(pgv_true) - log_pgv_s1
ax.scatter(df.loc[df.index, "n_i"], resid_s1_log, s=4, alpha=0.3, color="steelblue")
ax.axhline(0, color="red", lw=1, ls="--")
ax.set_xlabel("True n_i")
ax.set_ylabel("S1 residual (log scale)")
ax.set_title("S1 residuals vs true n_i\n(slope ≠ 0 means n_i adds information)")

plt.tight_layout()
out_path3 = OUTPUT_DIR / "oracle_ni_analysis.png"
plt.savefig(out_path3, dpi=120)
plt.close()
print(f"n_i analysis plot saved: {out_path3}")

print("\nDone.")
