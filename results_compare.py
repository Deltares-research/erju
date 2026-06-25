"""One-shot script to print before/after results table."""

import json
from pathlib import Path

models_root = Path(r"P:\11210978-erju-ai\holten_models")


def load(name):
    return json.loads((models_root / name / "summary.json").read_text())


# ── XGBoost v4 ────────────────────────────────────────────────────────────────
v4_old = load("xgb_v004_20260408_165448")
v4_new = load("xgb_v004_20260509_191209")

# ── XGBoost v6 ────────────────────────────────────────────────────────────────
v6_old = load("xgb_v006_20260509_152929")
v6_new = load("xgb_v006_20260509_220122")

# ── XGBoost v7 ────────────────────────────────────────────────────────────────
v7_old = load("xgb_v007_20260509_153011")
v7_new = load("xgb_v007_20260509_220559")

# ── XGBoost v8 ────────────────────────────────────────────────────────────────
v8_old = load("xgb_v008_20260509_154809")
v8_new = load("xgb_v008_20260509_221501")

# ── MLPs ──────────────────────────────────────────────────────────────────────
mlp_old = {
    i: load(f"mlp_v00{i}_20260408_18{s}")
    for i, s in zip([1, 2, 3, 4, 5], ["0508", "1957", "2326", "3030", "3514"])
}
mlp_new = {
    1: load("mlp_v001_20260509_221607"),
    2: load("mlp_v002_20260510_000754"),
    3: load("mlp_v003_20260510_000837"),
    4: load("mlp_v004_20260510_000954"),
    5: load("mlp_v005_20260510_001129"),
}

# ── Oracle (from oracle_attenuation_check output) ─────────────────────────────
# Old oracle (wrong distances): S1=1.72, S2=1.68  (from repo memory)
# New oracle (correct):         S1=1.8828, S2=1.4951
oracle_old = {"s1_rmse": 1.72, "s2_rmse": 1.68}
oracle_new = {"s1_rmse": 1.8828, "s2_rmse": 1.4951}

# n_global
n_old = v4_old  # use v6 old which has n_global
n_old_val = v6_old["n_global"]
n_new_val = v6_new["n_global"]

print("=" * 76)
print(
    f"{'Model':<22} {'Before (wrong dist)':>20} {'After (correct dist)':>20} {'Delta':>10}"
)
print(f"{'':22} {'Test RMSE (mm/s)':>20} {'Test RMSE (mm/s)':>20} {'(mm/s)':>10}")
print("=" * 76)

# XGBoost
xgb4_old_rmse = v4_old["test_metrics_mms"]["rmse"]
xgb4_new_rmse = v4_new["test_metrics_mms"]["rmse"]
print(
    f"{'XGBoost v4 (direct)':<22} {xgb4_old_rmse:>20.4f} {xgb4_new_rmse:>20.4f} {xgb4_new_rmse-xgb4_old_rmse:>+10.4f}"
)

xgb6_old_rmse = v6_old["test_sensor_metrics"]["rmse_mms"]
xgb6_new_rmse = v6_new["test_sensor_metrics"]["rmse_mms"]
print(
    f"{'XGBoost v6 (global n)':<22} {xgb6_old_rmse:>20.4f} {xgb6_new_rmse:>20.4f} {xgb6_new_rmse-xgb6_old_rmse:>+10.4f}"
)

xgb7_old_rmse = v7_old["test_sensor_metrics"]["rmse_mms"]
xgb7_new_rmse = v7_new["test_sensor_metrics"]["rmse_mms"]
print(
    f"{'XGBoost v7 (per-event n)':<22} {xgb7_old_rmse:>20.4f} {xgb7_new_rmse:>20.4f} {xgb7_new_rmse-xgb7_old_rmse:>+10.4f}"
)

xgb8a_old = v8_old["test"]["varA"]["rmse"]
xgb8a_new = v8_new["test"]["varA"]["rmse"]
print(
    f"{'XGBoost v8-A (residual)':<22} {xgb8a_old:>20.4f} {xgb8a_new:>20.4f} {xgb8a_new-xgb8a_old:>+10.4f}"
)

xgb8b_old = v8_old["test"]["varB"]["rmse"]
xgb8b_new = v8_new["test"]["varB"]["rmse"]
print(
    f"{'XGBoost v8-B (residual)':<22} {xgb8b_old:>20.4f} {xgb8b_new:>20.4f} {xgb8b_new-xgb8b_old:>+10.4f}"
)

print("-" * 76)

# MLPs
for i in [1, 2, 3, 4, 5]:
    old_rmse = mlp_old[i]["test_rmse_mms"]
    new_rmse = mlp_new[i]["test_rmse_mms"]
    print(
        f"{'MLP v' + str(i):<22} {old_rmse:>20.4f} {new_rmse:>20.4f} {new_rmse-old_rmse:>+10.4f}"
    )

print("-" * 76)

# Oracle
print(
    f"{'Oracle S1 ceiling':<22} {oracle_old['s1_rmse']:>20.4f} {oracle_new['s1_rmse']:>20.4f} {oracle_new['s1_rmse']-oracle_old['s1_rmse']:>+10.4f}"
)
print(
    f"{'Oracle S2 ceiling':<22} {oracle_old['s2_rmse']:>20.4f} {oracle_new['s2_rmse']:>20.4f} {oracle_new['s2_rmse']-oracle_old['s2_rmse']:>+10.4f}"
)

print("=" * 76)
print(
    f"\nAttenuation exponent n_global:  {n_old_val:.4f}  →  {n_new_val:.4f}  (Δ {n_new_val-n_old_val:+.4f})"
)
print()
print("Note: lower Oracle S1 ceiling before-fix is misleading — it was computed")
print("with wrong distances, artificially compressing the residuals.")
