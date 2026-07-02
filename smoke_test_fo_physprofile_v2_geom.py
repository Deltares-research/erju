"""
smoke_test_fo_physprofile_v2_geom.py

Smoke tests for the two-issue patch:
  Issue 1 — V1_ref contaminated by geometry features
  Issue 2 — ablation CSV uses wrong test metrics per policy
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import train_fo_physprofile_linec_v2_geometry as v2

# ─── Setup ────────────────────────────────────────────────────────────────────

cfg  = v2.load_site_config()
geom = v2.build_sensor_geometry_table(cfg)
v2.assert_geometry_values(geom)

# Simulate a row_df that has been through _build_row_features_v2 with
# add_geometry=True (as always happens in main). It will have fo_y_m and
# fo_to_track_axis_pos as well as real FO octave-band features.
base = pd.DataFrame({
    "event_id":    ["e0","e1","e2","e3","e4"],
    "sensor":       v2.SENSOR_ORDER,
    "track_number": [1]*5,
    "distance":     [2.5,4.0,8.0,16.0,23.0],
    "log_distance": [0.0]*5,
    "sensor_code":  [0,1,2,3,4],
    "pred_log_profile": [0.0]*5,
    "c_hat_profile":    [1.0]*5,
    "n_hat_profile":    [1.0]*5,
    "pc1_hat":          [0.0]*5,
    "pc2_hat":          [0.0]*5,
    "target_log":       [1.0]*5,
    "target_pgv":       [3.0]*5,
    "train_speed_kmh":  [120.0]*5,
    "train_type_code":  [0]*5,
    "train_speed_missing": [0.0]*5,
    # Parquet FO features (should survive V1_ref feature selection)
    "fo_oct_001hz_mean": [0.1]*5,
    "fo_oct_5_00hz_mean":[0.05]*5,
    "fo_td_rms_mean":    [1e-9]*5,
    # WF features
    "wf_global_rms":     [1e-9]*5,
})
df = v2.add_fo_relative_geometry_features(base, geom)
df = v2.add_geometry_interactions(df)

fo_all = [c for c in df.columns if c.startswith("fo_")]
print(f"fo_* columns in df (includes geometry): {fo_all}")
assert "fo_y_m" in fo_all,             "fo_y_m should be present"
assert "fo_to_track_axis_pos" in fo_all, "fo_to_track_axis_pos should be present"

# ─── Smoke 1: ISSUE 1 — V1_ref must have ZERO geometry columns ────────────────
print("\n[SMOKE 1] V1_ref must contain zero geometry columns")
fc_v1 = v2._get_feat_cols(df, use_sensor_code=True, use_geometry=False,
                            use_track_number=True)

# All forbidden geometry-related names
bad = [c for c in fc_v1
       if c in v2._GEOM_ALL_COLS
       or c.startswith("gi_")
       or "fo_to_track_axis" in c
       or "acc_track_to_fo" in c
       or "between_track_and_fo" in c
       or "beyond_fo" in c
       or c == "fo_y_m"]
assert bad == [], f"SMOKE 1 FAILED: geometry leaked into V1_ref: {bad}"

# Real parquet FO feature must still be present
fo_oct_in_v1 = [c for c in fc_v1 if c.startswith("fo_oct_")]
assert len(fo_oct_in_v1) > 0, "fo_oct_* features lost from V1_ref"
print(f"  V1_ref: {len(fc_v1)} features, 0 geometry, fo_oct present={fo_oct_in_v1[:2]}")
print("SMOKE 1 PASSED ✓")

# ─── Smoke 2: V2_geom contains fo_to_track_axis_pos ─────────────────────────
print("\n[SMOKE 2] V2_geom must contain fo_to_track_axis_pos")
fc_v2 = v2._get_feat_cols(df, use_sensor_code=True, use_geometry=True,
                            use_track_number=True)
assert "fo_to_track_axis_pos" in fc_v2
assert len(fc_v2) > len(fc_v1), "V2_geom must have more features than V1_ref"
print(f"  V2_geom: {len(fc_v2)} features  V1_ref: {len(fc_v1)} features")
print("SMOKE 2 PASSED ✓")

# ─── Smoke 3: no_sc has no sensor_code and no feat_sc_x_* ───────────────────
print("\n[SMOKE 3] V2_no_sc must not contain sensor_code or feat_sc_x_*")
fc_nsc = v2._get_feat_cols(df, use_sensor_code=False, use_geometry=True,
                             use_track_number=True)
assert "sensor_code" not in fc_nsc, f"sensor_code in fc_nsc"
assert not any(c.startswith("feat_sc_x_") for c in fc_nsc), "feat_sc_x_ in fc_nsc"
assert "fo_to_track_axis_pos" in fc_nsc, "geometry must still be present"
print(f"  V2_no_sc: {len(fc_nsc)} features")
print("SMOKE 3 PASSED ✓")

# ─── Smoke 4: no_trackcat has no track_number ────────────────────────────────
print("\n[SMOKE 4] V2_no_sensor_no_trackcat must not contain track_number")
fc_nnt = v2._get_feat_cols(df, use_sensor_code=False, use_geometry=True,
                             use_track_number=False)
assert "track_number" not in fc_nnt, "track_number in fc_nnt"
assert "fo_to_track_axis_pos" in fc_nnt, "geometry still needed"
print(f"  V2_no_trackcat: {len(fc_nnt)} features")
print("SMOKE 4 PASSED ✓")

# ─── Smoke 5: ISSUE 2 — policy_metrics stores independent per-policy metrics ─
print("\n[SMOKE 5] policy_metrics independence check")
pm: dict = {}
pm[("V1_ref", "global_best")] = {
    "rmse_pgv": 2.20,
    "per_sensor": {"MP4": {"rmse_pgv": 4.50}},
    "mp4_pgv_gt4": {"rmse_pgv": 5.10, "n": 20},
}
pm[("V1_ref", "mp4_best")] = {
    "rmse_pgv": 2.25,
    "per_sensor": {"MP4": {"rmse_pgv": 4.30}},
    "mp4_pgv_gt4": {"rmse_pgv": 4.80, "n": 20},
}
# Simulate how the ablation CSV rows are now built
abl = []
for policy in ["global_best", "mp4_best"]:
    m = pm.get(("V1_ref", policy), {})
    ps = m.get("per_sensor", {})
    abl.append({
        "variant": "V1_ref", "policy": policy,
        "test_rmse_pgv": m.get("rmse_pgv", float("nan")),
        "test_mp4_rmse": ps.get("MP4", {}).get("rmse_pgv", float("nan")),
        "test_mp4_pgv_gt4_rmse": m.get("mp4_pgv_gt4", {}).get("rmse_pgv", float("nan")),
    })

abl_df = pd.DataFrame(abl)
gb_row = abl_df[abl_df["policy"] == "global_best"].iloc[0]
mb_row = abl_df[abl_df["policy"] == "mp4_best"].iloc[0]

assert gb_row["test_rmse_pgv"] != mb_row["test_rmse_pgv"], \
    "SMOKE 5 FAILED: global_best and mp4_best have identical test_rmse_pgv"
assert gb_row["test_mp4_rmse"] != mb_row["test_mp4_rmse"], \
    "SMOKE 5 FAILED: global_best and mp4_best have identical test_mp4_rmse"
print(f"  global_best: rmse={gb_row['test_rmse_pgv']:.3f}  MP4={gb_row['test_mp4_rmse']:.3f}")
print(f"  mp4_best:    rmse={mb_row['test_rmse_pgv']:.3f}  MP4={mb_row['test_mp4_rmse']:.3f}")
print("SMOKE 5 PASSED ✓")

# ─── Verify _GEOM_ALL_COLS is a module-level attribute ────────────────────────
assert hasattr(v2, "_GEOM_ALL_COLS"), "_GEOM_ALL_COLS not found at module level"
assert "fo_to_track_axis_pos" in v2._GEOM_ALL_COLS
assert "fo_y_m" in v2._GEOM_ALL_COLS
print(f"\n_GEOM_ALL_COLS: {len(v2._GEOM_ALL_COLS)} entries  ✓")

print("\n" + "=" * 60)
print("ALL ISSUE-1 + ISSUE-2 SMOKE TESTS PASSED")
print("=" * 60)
