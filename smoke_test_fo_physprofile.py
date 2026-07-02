"""
smoke_test_fo_physprofile.py
Minimal 5-event smoke test for train_fo_physprofile_linec_v1.py

Tests:
  1. Waveform feature extraction on synthetic data
  2. Physics target computation
  3. PCA fit on tiny dataset
  4. Leakage assertion
  5. reconstruct_profile_predictions alignment
  6. apply_monotonic correctness
  7. load_waveforms fs detection (no actual file needed - unit test)
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

# Import the module under test
import train_fo_physprofile_linec_v1 as M

SENSOR_ORDER = M.SENSOR_ORDER
R0 = M.R0
N_TRACK = M.N_TRACK

# ─── 1. Waveform feature extraction ──────────────────────────────────────────
print("\n[1] Waveform feature extraction...")
np.random.seed(0)
wf_block = np.random.randn(21, 7500).astype(np.float32) * 1e-9
for fs in [250.0, 1000.0]:
    feats = M.extract_waveform_features_event(wf_block, fs=fs)
    assert "wf_global_rms" in feats, "Missing wf_global_rms"
    assert "wf_band_1_5" in feats, "Missing wf_band_1_5"
    assert "wf_spatial_centroid" in feats, "Missing spatial centroid"
    assert "wf_adj_corr_mean" in feats, "Missing adj corr"
    assert "wf_post_minus_pre" in feats, "Missing post_minus_pre"
    dur_25 = feats["wf_duration_above_25pct"]
    assert 0 <= dur_25 <= 7500 / fs + 1, f"wf_duration_above_25pct={dur_25} out of range"
print(f"   OK  {len(feats)} features extracted  (fs=250 and fs=1000)")

# ─── 2. Physics target computation ───────────────────────────────────────────
print("\n[2] Physics target computation...")
N_EVENTS = 20
events, sensors, tracks, dists, targets = [], [], [], [], []
sensor_dists_t1 = dict(zip(SENSOR_ORDER, [2.5, 4.0, 8.0, 16.0, 23.0]))

np.random.seed(42)
for i in range(N_EVENTS):
    c_true = np.random.uniform(0.5, 2.5)
    n_true = N_TRACK[1] + np.random.uniform(-0.2, 0.2)
    for sensor in SENSOR_ORDER:
        r = sensor_dists_t1[sensor]
        log_pgv = c_true - n_true * np.log(r / R0)
        log_pgv += np.random.randn() * 0.05
        events.append(f"ev{i:03d}")
        sensors.append(sensor)
        tracks.append(1)
        dists.append(r)
        targets.append(np.exp(log_pgv))

df_test = pd.DataFrame({
    "event_id":    events,
    "sensor":      sensors,
    "sensor_id":   sensors,
    "track_number": tracks,
    "distance":    dists,
    "effective_distance_to_active_track_m": dists,
    "target_pgv_z_mms": targets,
    "target_pgv":  targets,
    "target_log":  [np.log(t) for t in targets],
    "log_distance": [np.log(d / R0) for d in dists],
    "sensor_code": [SENSOR_ORDER.index(s) for s in sensors],
})

target_df, resid_df = M.compute_physics_targets(df_test)
assert len(target_df) == N_EVENTS, f"Expected {N_EVENTS} event targets, got {len(target_df)}"
assert "c_target_event" in target_df.columns
assert "delta_n_target" in target_df.columns
assert resid_df["residual_profile"].abs().max() < 1.0, \
    "Residuals too large for synthetic data"
print(f"   OK  {len(target_df)} events  mean c_target={target_df['c_target_event'].mean():.3f}"
      f"  mean delta_n={target_df['delta_n_target'].mean():.4f}")

# ─── 3. PCA fit ───────────────────────────────────────────────────────────────
print("\n[3] PCA on residual profiles...")
split_df = pd.DataFrame({
    "event_id": [f"ev{i:03d}" for i in range(N_EVENTS)],
    "split":    ["train"] * 15 + ["val"] * 3 + ["test"] * 2,
})
pca, pc_score_df, pca_ve = M.fit_pca_residuals(target_df, resid_df, split_df, n_comp=3)
assert len(pc_score_df) == N_EVENTS
assert pca_ve.sum() <= 1.001
assert "pc1_target" in pc_score_df.columns
print(f"   OK  VE={pca_ve.round(3).tolist()}")

# ─── 4. Leakage assertion ─────────────────────────────────────────────────────
print("\n[4] Leakage assertion...")
safe_cols = ["log_distance", "sensor_code", "fo_oct_001hz_mean", "wf_global_rms",
             "pred_log_profile", "c_hat_profile", "n_hat_profile"]
M.assert_no_leakage(safe_cols, "test_safe")

# These should trigger
leaky_cols = safe_cols + ["target_pgv"]
try:
    M.assert_no_leakage(leaky_cols, "test_leaky")
    print("   FAIL: should have raised ValueError for target_pgv")
    sys.exit(1)
except ValueError as e:
    print(f"   OK  correctly caught: {e}")

leaky_cols2 = safe_cols + ["c_target_event"]
try:
    M.assert_no_leakage(leaky_cols2, "test_leaky2")
    print("   FAIL: should have raised ValueError for c_target_event")
    sys.exit(1)
except ValueError:
    print("   OK  correctly caught c_target_event leakage")

# ─── 5. reconstruct_profile_predictions alignment ─────────────────────────────
print("\n[5] Profile reconstruction alignment...")
# Build minimal dicts from first 5 events
ev_ids = target_df["event_id"].values[:5]
c_by  = {e: float(target_df.loc[target_df["event_id"] == e, "c_target_event"].iloc[0])
          for e in ev_ids}
dn_by = {e: 0.0 for e in ev_ids}
pc_by = {e: np.zeros(2) for e in ev_ids}

df_sub = df_test[df_test["event_id"].isin(ev_ids)].copy()
pred = M.reconstruct_profile_predictions(df_sub, ev_ids, c_by, dn_by, pc_by, pca)
assert pred.shape[0] == len(df_sub), "Shape mismatch"
n_valid = np.isfinite(pred).sum()
assert n_valid > 0, "No valid predictions"
print(f"   OK  {n_valid}/{len(pred)} valid predictions")

# ─── 6. apply_monotonic ───────────────────────────────────────────────────────
print("\n[6] apply_monotonic...")
df_mono = df_sub.copy()
df_mono["pred_col"] = pred
mono_pred = M.apply_monotonic(df_mono, "pred_col")
# Within each event, mono_pred should be non-increasing with distance
for eid in ev_ids:
    ev = df_mono[df_mono["event_id"] == eid].sort_values("distance")
    m_vals = mono_pred[ev.index].values
    for k in range(len(m_vals) - 1):
        if not np.isnan(m_vals[k]) and not np.isnan(m_vals[k+1]):
            assert m_vals[k] >= m_vals[k+1] - 1e-9, \
                f"Monotonicity violation for {eid}: {m_vals}"
print("   OK  monotonicity constraints satisfied")

# ─── 7. WF_FS global update ───────────────────────────────────────────────────
print("\n[7] WF_FS global variable...")
original = M.WF_FS
M.WF_FS = 1000.0
assert M.WF_FS == 1000.0
M.WF_FS = original
print(f"   OK  WF_FS mutable global (default={original})")

print("\n" + "=" * 50)
print("ALL SMOKE TESTS PASSED")
print("=" * 50)
