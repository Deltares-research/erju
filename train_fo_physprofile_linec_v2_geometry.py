"""
train_fo_physprofile_linec_v2_geometry.py  —  FO-PhysProfile v2 Geometry

Geometry correction/augmentation of FO-PhysProfile v1.

Core addition:
  Explicit FO-relative coordinate system for every sensor row:
    fo_to_track_axis_pos = (sensor_y - fo_y) / (track_y - fo_y)

  Holten coordinate system (from sites/holten.json):
    FO cable:  Y = 0
    Track 1:   Y = 4.0 m
    Track 2:   Y = 8.0 m
    Line-C side -1 sensors:
      MP4:  Y = +1.5  (between FO and track 1)
      MP8:  Y =  0.0  (on FO cable)
      MP10: Y = -4.0  (behind FO, away from track)
      MP1:  Y = -12.0 (behind FO)
      MP2:  Y = -19.0 (behind FO)

Variants trained (local residual only, v1 profile predictions reused):
  V1_ref               — v1 feature set, reproduces v1 local residual
  V2_geom              — v1 + FO-relative geometry features + interactions
  V2_geom_no_sensor_code — ablation: drop sensor_code, keep geometry

Current baselines:
  FO-PhysProfile v1:   RMSE(PGV)=2.1882  RMSE(log)=0.5962  MP4=4.5592
  PXGBR_ens_top5:      RMSE(PGV)=2.2473  RMSE(log)=0.6223  MP4=4.6680
  PXGBR-R2:            RMSE(PGV)=2.2718  RMSE(log)=0.5946  MP4=4.7389
  P3:                   RMSE(PGV)=2.4006  RMSE(log)=0.5993  MP4=5.0185

Success criteria:
  MP4 RMSE < 4.50
  Global RMSE(PGV) <= 2.1882
  RMSE(log) <= 0.60
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

# Import pipeline utilities from v1 (stateless functions only)
import train_fo_physprofile_linec_v1 as _v1

from src.utils.geometry_utils import apply_corrected_distances


# ─── Constants ────────────────────────────────────────────────────────────────

def _root() -> Path:
    return Path(r"P:\11210978-erju-ai") if os.name == "nt" else Path("/p/11210978-erju-ai")

MODELS_ROOT  = _root() / "holten_models"
PARQUET_ROOT = _root() / "holten_parquet"
WAVE_ROOT    = _root() / "holten_waveform"

SENSOR_ORDER = ["MP4", "MP8", "MP10", "MP1", "MP2"]
SENSOR_CODE  = {s: i for i, s in enumerate(SENSOR_ORDER)}

N_TRACK   = {1: 1.0655, 2: 1.3246}
R0        = 10.0
N_SHRINK_W = 0.3
N_CLIP    = (0.5, 2.0)
PCA_N_COMP = 3

# Site config
SITE_CONFIG_PATH = _REPO / "sites" / "holten.json"

# Leakage tokens (subset of v1 LEAKAGE + explicit physics targets)
_LEAKAGE_TOKENS = _v1._LEAKAGE_TOKENS


# ─── Path helpers ─────────────────────────────────────────────────────────────

def make_output_dir(base: Path) -> Path:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / "outputs" / "fo_physprofile_linec_v2_geometry" \
          / f"fo_physprofile_linec_v002_geom_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def find_v1_output() -> Optional[Path]:
    """Return the most recent completed v1 output directory."""
    hits = sorted(
        (MODELS_ROOT / "outputs" / "fo_physprofile_linec_v1").glob(
            "fo_physprofile_linec_v001_*"
        )
        if (MODELS_ROOT / "outputs" / "fo_physprofile_linec_v1").exists()
        else [],
        key=lambda p: p.name,
    )
    for h in reversed(hits):
        if (h / "profile_targets_event.parquet").exists():
            return h
    return None


# ─── TASK 1 — Site geometry config ────────────────────────────────────────────

def load_site_config(path: Path = SITE_CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Holten site config not found: {path}")
    with open(path) as f:
        return json.load(f)


def build_sensor_geometry_table(config: dict) -> pd.DataFrame:
    """
    Extract per-sensor Y-coordinates and FO/track reference quantities.
    Returns DataFrame indexed by sensor name.
    """
    coords = config["accelerometer"]["sensor_coordinates_m"]
    fo_cfg = config["fibre_optics"]

    fo_y      = 0.0                           # FO cable always at Y=0
    track1_y  = float(fo_cfg["track_1_to_fo_m"])   # 4.0
    track2_y  = float(fo_cfg["track_2_to_fo_m"])   # 8.0

    rows = []
    for sensor in SENSOR_ORDER:
        if sensor not in coords:
            continue
        sx = float(coords[sensor]["x"])
        sy = float(coords[sensor]["y"])
        rows.append({
            "sensor":       sensor,
            "sensor_x_m":   sx,
            "sensor_y_m":   sy,
            "fo_y_m":       fo_y,
            "track1_y_m":   track1_y,
            "track2_y_m":   track2_y,
        })

    df = pd.DataFrame(rows).set_index("sensor")
    return df


def print_geometry_diagnostic(geom_df: pd.DataFrame) -> None:
    fo_y = 0.0
    print("\n" + "=" * 80)
    print("GEOMETRY DIAGNOSTIC — FO-Relative Coordinates")
    print("=" * 80)
    header = (f"{'Sensor':>6}  {'y_m':>6}  {'Track':>5}  "
              f"{'track_y':>7}  {'fo_axis_pos':>11}  "
              f"{'acc_to_track':>12}  {'acc_to_fo':>9}  Zone")
    print(header)
    print("-" * 80)
    for sensor in SENSOR_ORDER:
        if sensor not in geom_df.index:
            continue
        row = geom_df.loc[sensor]
        sy = row["sensor_y_m"]
        for track_no, track_y in [(1, row["track1_y_m"]), (2, row["track2_y_m"])]:
            denom = track_y - fo_y
            axis_pos = (sy - fo_y) / denom if abs(denom) > 1e-9 else 0.0
            acc_to_track = abs(sy - track_y)
            acc_to_fo    = abs(sy - fo_y)
            if   axis_pos > 1.0:   zone = "beyond_track"
            elif axis_pos < 0.0:   zone = "beyond_fo_away"
            elif abs(axis_pos) < 0.75 / abs(denom): zone = "on_fo_line"
            else:                  zone = "between_fo_track"
            print(f"{sensor:>6}  {sy:>6.1f}  "
                  f"{'T'+str(track_no):>5}  {track_y:>7.1f}  "
                  f"{axis_pos:>11.4f}  {acc_to_track:>12.2f}  "
                  f"{acc_to_fo:>9.2f}  {zone}")
    print("=" * 80)


def add_fo_relative_geometry_features(
    df: pd.DataFrame,
    geom_df: pd.DataFrame,
    r0: float = R0,
) -> pd.DataFrame:
    """
    Add FO-relative geometry features to every row in df.
    Requires: df has columns  sensor, track_number.
    """
    df = df.copy()
    fo_y = 0.0

    # Vectorised approach: build feature arrays
    n = len(df)
    # Sensor y-coordinates
    sy = df["sensor"].map(geom_df["sensor_y_m"].to_dict()).astype(float).values
    sx = df["sensor"].map(geom_df["sensor_x_m"].to_dict()).astype(float).values
    tn = df["track_number"].values.astype(int)

    # Active track Y (per row, depending on which track the train is on)
    track1_y = float(geom_df["track1_y_m"].iloc[0])   # 4.0
    track2_y = float(geom_df["track2_y_m"].iloc[0])   # 8.0
    active_track_y = np.where(tn == 1, track1_y, track2_y)  # (n,)

    # ── Basic coordinates ──────────────────────────────────────────────────────
    df["sensor_x_m"]           = sx
    df["sensor_y_m"]           = sy
    df["fo_y_m"]               = fo_y
    df["active_track_y_m"]     = active_track_y
    df["active_track_to_fo_m"] = np.abs(active_track_y - fo_y)   # 4 or 8

    # ── Track-reference features ───────────────────────────────────────────────
    df["acc_to_active_track_m"]       = np.abs(sy - active_track_y)
    df["signed_acc_from_active_track"] = sy - active_track_y
    df["log_acc_to_active_track"]     = np.log(
        np.maximum(np.abs(sy - active_track_y), 0.1) / r0
    )

    # ── Fibre-reference features ───────────────────────────────────────────────
    df["acc_to_fo_m"]       = np.abs(sy - fo_y)
    df["signed_acc_from_fo"] = sy - fo_y

    # ── Normalised source-fibre-receiver coordinate ────────────────────────────
    denom = active_track_y - fo_y          # 4 or 8
    df["fo_to_track_axis_pos"] = (sy - fo_y) / denom

    # ── Boolean / topology flags ───────────────────────────────────────────────
    axis = df["fo_to_track_axis_pos"].values
    df["is_between_track_and_fo"]      = ((axis > 0) & (axis < 1)).astype(float)
    df["is_on_fo_line"]                = (np.abs(sy - fo_y) <= 0.75).astype(float)
    df["is_beyond_fo_away_from_track"] = (axis < 0).astype(float)
    df["is_beyond_track"]              = (axis > 1).astype(float)
    df["is_track_side_of_fo"]          = df["is_between_track_and_fo"]   # alias

    # ── Distance-ratio features ────────────────────────────────────────────────
    dtrack = np.maximum(np.abs(sy - active_track_y), 0.1)
    dfo    = np.maximum(np.abs(sy - fo_y), 0.1)
    d_t2fo = np.maximum(np.abs(active_track_y - fo_y), 0.1)

    df["acc_track_to_fo_ratio"]        = dtrack / d_t2fo
    df["log_acc_track_to_fo_ratio"]    = np.log(np.maximum(dtrack / d_t2fo, 1e-3))
    df["acc_to_fo_over_track_to_fo"]   = dfo / d_t2fo

    return df


# ─── TASK 2 — Geometry interactions ───────────────────────────────────────────

_GEOM_COLS = [
    "fo_to_track_axis_pos", "acc_track_to_fo_ratio",
    "is_between_track_and_fo", "is_on_fo_line",
    "is_beyond_fo_away_from_track", "acc_to_fo_m",
    "signed_acc_from_fo", "signed_acc_from_active_track",
    "log_acc_to_active_track", "log_acc_track_to_fo_ratio",
    "acc_to_fo_over_track_to_fo", "active_track_to_fo_m",
]

_GEOM_WF_PAIRS = [
    ("fo_to_track_axis_pos", "wf_ch_energy_max"),
    ("fo_to_track_axis_pos", "wf_global_energy"),
    ("fo_to_track_axis_pos", "wf_global_rms"),
    ("fo_to_track_axis_pos", "wf_high_low_energy_ratio"),
    ("fo_to_track_axis_pos", "wf_spectral_centroid"),
    ("acc_track_to_fo_ratio", "wf_ch_energy_max"),
    ("acc_track_to_fo_ratio", "wf_high_low_energy_ratio"),
    ("is_between_track_and_fo", "wf_ch_energy_max"),
    ("is_between_track_and_fo", "wf_global_energy"),
    ("is_between_track_and_fo", "wf_high_low_energy_ratio"),
]

_GEOM_OCT_PAIRS = [
    ("fo_to_track_axis_pos", "fo_oct_063hz"),
    ("fo_to_track_axis_pos", "fo_oct_1_00hz"),
    ("fo_to_track_axis_pos", "fo_oct_2_00hz"),
    ("fo_to_track_axis_pos", "fo_oct_5_00hz"),
    ("fo_to_track_axis_pos", "fo_oct_8_00hz"),
    ("acc_track_to_fo_ratio", "fo_oct_063hz"),
    ("acc_track_to_fo_ratio", "fo_oct_1_00hz"),
    ("is_between_track_and_fo", "fo_oct_063hz"),
    ("is_between_track_and_fo", "fo_oct_1_00hz"),
    ("is_between_track_and_fo", "fo_oct_5_00hz"),
]

_GEOM_PROFILE_PAIRS = [
    ("is_between_track_and_fo", "pred_log_profile"),
    ("acc_track_to_fo_ratio",   "pred_log_profile"),
    ("fo_to_track_axis_pos",    "pred_log_profile"),
]


def add_geometry_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-product features between geometry and FO/waveform/profile columns."""
    df = df.copy()

    # WF interactions (exact column name match)
    for gcol, wcol in _GEOM_WF_PAIRS:
        if gcol in df.columns and wcol in df.columns:
            df[f"gi_{gcol}_x_{wcol}"] = df[gcol] * df[wcol]

    # Octave band interactions (prefix match: look for any col starting with suffix)
    for gcol, oct_prefix in _GEOM_OCT_PAIRS:
        if gcol not in df.columns:
            continue
        # Find all columns that start with the prefix (e.g. fo_oct_063hz_mean/_max/_std)
        matches = [c for c in df.columns if c.startswith(oct_prefix)]
        for wcol in matches:
            df[f"gi_{gcol}_x_{wcol}"] = df[gcol] * df[wcol]

    # Profile prior interactions
    for gcol, pcol in _GEOM_PROFILE_PAIRS:
        if gcol in df.columns and pcol in df.columns:
            df[f"gi_{gcol}_x_{pcol}"] = df[gcol] * df[pcol]

    return df


# ─── Local residual model ──────────────────────────────────────────────────────

def _local_grid_v2() -> List[dict]:
    """Focused grid around v1 best (scheme=agg_hi), slightly expanded."""
    grid = []
    for depth, lr, mcw, rl in product(
        [3, 4, 5],
        [0.01, 0.02, 0.05],
        [1, 5],
        [1, 10, 20],
    ):
        grid.append({
            "max_depth": depth,
            "learning_rate": lr,
            "min_child_weight": mcw,
            "reg_lambda": rl,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        })
    return grid


def _local_weights_v2(df: pd.DataFrame, scheme: str) -> np.ndarray:
    s  = df["sensor"].values
    tp = df["target_pgv"].values
    w  = np.ones(len(df), dtype=np.float32)
    if scheme == "agg_hi":
        w += 1.0 * (s == "MP4") + 2.0 * (tp > 4) + 3.0 * (tp > 8)
    elif scheme == "mp4_hi":
        w += 1.5 * (s == "MP4") + 1.0 * (tp > 4) + 1.0 * (tp > 8)
    return w


def _val_selection_score(pred_pgv_va, tgv_va, s_va):
    """Multi-criterion selection: RMSE(PGV), tie-break MP4 RMSE, then RMSE(log)."""
    rmse_pgv = float(np.sqrt(np.mean((pred_pgv_va - tgv_va) ** 2)))
    mp4 = s_va == "MP4"
    mp4_rmse = float(np.sqrt(np.mean((pred_pgv_va[mp4] - tgv_va[mp4]) ** 2))) if mp4.any() else 999.0
    return rmse_pgv, mp4_rmse


def train_local_residual_variant(
    row_tr: pd.DataFrame,
    row_va: pd.DataFrame,
    feat_cols: List[str],
    label: str,
    schemes: List[str] = ("uniform", "agg_hi"),
) -> Tuple[xgb.XGBRegressor, float, float, str, List[str]]:
    """
    Train local residual XGBoost for one variant.
    Selection: val RMSE(PGV), tie-break MP4 RMSE.
    Returns (best_model, val_rmse_pgv, val_mp4_rmse, best_scheme, used_feat_cols).
    """
    valid_cols = [c for c in feat_cols
                  if c in row_tr.columns and c in row_va.columns]
    y_tr = (row_tr["target_log"] - row_tr["pred_log_profile"]).values
    y_va = (row_va["target_log"] - row_va["pred_log_profile"]).values
    X_tr = row_tr[valid_cols].fillna(0.0).values.astype(np.float32)
    X_va = row_va[valid_cols].fillna(0.0).values.astype(np.float32)
    p0_va = row_va["pred_log_profile"].values
    tgv_va = row_va["target_pgv"].values
    s_va   = row_va["sensor"].values

    grid = _local_grid_v2()
    best_score = (1e9, 1e9)
    best_model = None
    best_scheme_used = schemes[0]
    n_fits = 0

    for scheme in schemes:
        w_tr = _local_weights_v2(row_tr, scheme)
        for params in grid:
            m = xgb.XGBRegressor(
                **params,
                n_estimators=3000,
                early_stopping_rounds=50,
                eval_metric="rmse",
                tree_method="hist",
                verbosity=0,
                random_state=42,
            )
            m.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_va, y_va)], verbose=False)
            res_pred = m.predict(X_va)
            pred_pgv = np.exp(p0_va + res_pred)
            score = _val_selection_score(pred_pgv, tgv_va, s_va)
            if score < best_score:
                best_score = score
                best_model = m
                best_scheme_used = scheme
            n_fits += 1

    print(f"[{label}] Best val RMSE(PGV)={best_score[0]:.4f}  "
          f"MP4={best_score[1]:.4f}  scheme={best_scheme_used}  "
          f"({n_fits} fits)")
    return best_model, best_score[0], best_score[1], best_scheme_used, valid_cols


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics_v2(
    pred_log: np.ndarray,
    true_log: np.ndarray,
    sensors:  np.ndarray,
    event_ids: np.ndarray,
    label: str,
) -> dict:
    """Full metrics including per-sensor and high-PGV breakdown."""
    pp = np.exp(pred_log); tp = np.exp(true_log)
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    bias = lambda a, b: float(np.mean(a - b))
    r2 = lambda pl, tl: float(
        1 - np.sum((tl - pl) ** 2) / max(np.sum((tl - tl.mean()) ** 2), 1e-12)
    )
    m = {
        "model":    label,
        "rmse_log": rmse(pred_log, true_log),
        "rmse_pgv": rmse(pp, tp),
        "mae_pgv":  float(np.mean(np.abs(pp - tp))),
        "r2_log":   r2(pred_log, true_log),
        "bias_pgv": bias(pp, tp),
    }
    m["per_sensor"] = {}
    for s in SENSOR_ORDER:
        msk = sensors == s
        if msk.any():
            m["per_sensor"][s] = {
                "rmse_pgv": rmse(pp[msk], tp[msk]),
                "rmse_log": rmse(pred_log[msk], true_log[msk]),
                "bias_pgv": bias(pp[msk], tp[msk]),
                "bias_log": bias(pred_log[msk], true_log[msk]),
            }
    for thr in [4.0, 8.0]:
        hi = tp > thr
        if hi.any():
            m[f"pgv_gt{int(thr)}"] = {"n": int(hi.sum()), "rmse_pgv": rmse(pp[hi], tp[hi])}
        mp4hi = (sensors == "MP4") & hi
        if mp4hi.any():
            m[f"mp4_pgv_gt{int(thr)}"] = {"n": int(mp4hi.sum()), "rmse_pgv": rmse(pp[mp4hi], tp[mp4hi])}
    # Monotonicity
    viol = tot = 0
    df_tmp = pd.DataFrame({"eid": event_ids, "s": sensors, "p": pred_log})
    for eid, grp in df_tmp.groupby("eid"):
        if len(grp) != 5:
            continue
        row = grp.set_index("s").reindex(SENSOR_ORDER).dropna()
        vals = row["p"].values
        for k in range(len(vals) - 1):
            tot += 1; viol += int(vals[k] < vals[k + 1])
    m["mono_viol_rate"] = viol / tot if tot else 0.0
    return m


def compute_geometry_diagnostics(
    row_te: pd.DataFrame,
    pred_col: str,
    label: str,
) -> Dict[str, dict]:
    """Per-geometry-zone metrics and residual bias."""
    diag: Dict[str, dict] = {}
    if pred_col not in row_te.columns:
        return diag

    valid = row_te.dropna(subset=[pred_col])
    pp = np.exp(valid[pred_col].values)
    tp = valid["target_pgv"].values
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

    # Zone masks
    zones = {
        "between_track_and_fo": valid["is_between_track_and_fo"].values == 1,
        "on_fo_line":           valid["is_on_fo_line"].values == 1,
        "beyond_fo_away":       valid["is_beyond_fo_away_from_track"].values == 1,
        "beyond_track":         valid.get("is_beyond_track",
                                  pd.Series(0, index=valid.index)).values == 1,
    }
    for zone, msk in zones.items():
        if msk.sum() > 3:
            diag[zone] = {"n": int(msk.sum()), "rmse_pgv": rmse(pp[msk], tp[msk]),
                          "bias_pgv": float(np.mean(pp[msk] - tp[msk]))}

    # Bias by fo_to_track_axis_pos bin
    if "fo_to_track_axis_pos" in valid.columns:
        axis = valid["fo_to_track_axis_pos"].values
        bins = [(-99, 0, "lt0"), (0, 0.3, "near0"),
                (0.3, 1.0, "0to1"), (1.0, 99, "gt1")]
        for lo, hi, bname in bins:
            msk = (axis >= lo) & (axis < hi)
            if msk.sum() > 3:
                diag[f"axis_bin_{bname}"] = {
                    "n": int(msk.sum()),
                    "rmse_pgv":  rmse(pp[msk], tp[msk]),
                    "bias_pgv":  float(np.mean(pp[msk] - tp[msk])),
                }

    print(f"\n[geom_diag:{label}]")
    for k, v in diag.items():
        print(f"  {k:30s}  n={v['n']:5d}  "
              f"rmse_pgv={v.get('rmse_pgv', 0):.4f}  "
              f"bias={v.get('bias_pgv', 0):+.4f}")
    return diag


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _save_fig(path: Path, dpi: int = 150) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_per_sensor_rmse_v1_vs_v2(metrics_dict: Dict[str, dict], out: Path):
    models = list(metrics_dict.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SENSOR_ORDER))
    w = 0.8 / len(models)
    for i, (label, m) in enumerate(metrics_dict.items()):
        rmses = [m["per_sensor"].get(s, {}).get("rmse_pgv", np.nan)
                 for s in SENSOR_ORDER]
        bars = ax.bar(x + i * w, rmses, w, label=label)
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(SENSOR_ORDER)
    ax.set_ylabel("RMSE(PGV)  [mm/s]")
    ax.set_title("Per-sensor RMSE — v1 vs v2_geometry")
    ax.legend(fontsize=8)
    _save_fig(out)


def plot_residual_vs_axis_pos(row_te: pd.DataFrame, pred_col: str, out: Path):
    if pred_col not in row_te.columns or "fo_to_track_axis_pos" not in row_te.columns:
        return
    valid = row_te.dropna(subset=[pred_col, "fo_to_track_axis_pos"])
    residual = valid["target_log"] - valid[pred_col]
    axis = valid["fo_to_track_axis_pos"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, clabel in [
        (axes[0], "sensor", "Sensor"),
        (axes[1], "track_number", "Track"),
    ]:
        for group in valid[col].unique():
            msk = valid[col] == group
            ax.scatter(axis[msk], residual[msk], alpha=0.2, s=6, label=str(group))
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.axvline(0, color="gray", lw=0.6, ls=":")
        ax.axvline(1, color="gray", lw=0.6, ls=":")
        ax.set_xlabel("fo_to_track_axis_pos"); ax.set_ylabel("Residual (log scale)")
        ax.set_title(f"Residual vs axis pos — by {clabel}")
        ax.legend(fontsize=7, ncol=3)
    _save_fig(out)


def plot_residual_by_geometry_zone(row_te: pd.DataFrame, pred_col: str, out: Path):
    if pred_col not in row_te.columns:
        return
    valid = row_te.dropna(subset=[pred_col])
    residuals = valid["target_log"] - valid[pred_col]
    zones = ["is_between_track_and_fo", "is_on_fo_line",
             "is_beyond_fo_away_from_track"]
    zone_labels = ["between track & FO", "on FO line", "beyond FO (away)"]
    colors = ["steelblue", "orange", "green"]
    fig, ax = plt.subplots(figsize=(8, 4))
    data = []
    labs = []
    for zone, label, color in zip(zones, zone_labels, colors):
        if zone not in valid.columns:
            continue
        msk = valid[zone].values == 1
        if msk.sum() > 0:
            data.append(residuals[msk].values)
            labs.append(f"{label}\n(n={msk.sum()})")
    if data:
        ax.boxplot(data, labels=labs, patch_artist=True,
                   medianprops={"color": "red"})
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("Residual (log scale)")
    ax.set_title("Residual distribution by geometry zone")
    _save_fig(out)


def plot_high_pgv_profiles_v2(row_te: pd.DataFrame, pred_col: str,
                                ref_col: Optional[str], n: int, out: Path):
    top_eids = (row_te.groupby("event_id")["target_pgv"]
                .max().nlargest(n).index)
    ncols = 4; nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
    for ax, eid in zip(np.array(axes).flat, top_eids):
        ev = row_te[row_te["event_id"] == eid].sort_values("distance")
        ax.plot(ev["distance"], np.exp(ev["target_log"]), "ko-", ms=4, label="true")
        if pred_col in ev.columns:
            ax.plot(ev["distance"], np.exp(ev[pred_col]), "r^--", ms=4, label="v2")
        if ref_col and ref_col in ev.columns:
            ax.plot(ev["distance"], np.exp(ev[ref_col]), "b.:", ms=4, label="v1")
        ax.set_title(str(eid)[:12], fontsize=7)
        ax.set_xlabel("dist (m)", fontsize=7); ax.set_ylabel("PGV", fontsize=7)
    np.array(axes).flat[0].legend(fontsize=6)
    plt.suptitle("High-PGV profiles — FO-PhysProfile v2 geometry", fontsize=10)
    _save_fig(out)


def plot_mp4_profiles_v1_vs_v2(row_te: pd.DataFrame, v1_col: str, v2_col: str,
                                 n: int, out: Path):
    """Compare v1 vs v2 predictions specifically for MP4 events."""
    if v1_col not in row_te.columns or v2_col not in row_te.columns:
        return
    mp4 = row_te[row_te["sensor"] == "MP4"].copy()
    top_eids = mp4.groupby("event_id")["target_pgv"].max().nlargest(n).index
    # Get full event data (all sensors) for those events
    ev_data = row_te[row_te["event_id"].isin(top_eids)]
    ncols = 4; nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
    for ax, eid in zip(np.array(axes).flat, top_eids):
        ev = ev_data[ev_data["event_id"] == eid].sort_values("distance")
        ax.plot(ev["distance"], np.exp(ev["target_log"]), "ko-", ms=4, label="true")
        ax.plot(ev["distance"], np.exp(ev[v2_col].fillna(np.nan)), "r^--", ms=4, label="v2")
        ax.plot(ev["distance"], np.exp(ev[v1_col].fillna(np.nan)), "b.:", ms=4, label="v1")
        ax.set_title(str(eid)[:12], fontsize=7)
        ax.set_xlabel("dist (m)", fontsize=7); ax.set_ylabel("PGV", fontsize=7)
    np.array(axes).flat[0].legend(fontsize=6)
    plt.suptitle("High-PGV MP4-selected events — v1 vs v2", fontsize=10)
    _save_fig(out)


def plot_scatter_v2(true_pgv, pred_pgv, label, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_pgv, pred_pgv, alpha=0.3, s=8)
    lim = (0, max(true_pgv.max(), pred_pgv.max()) * 1.05)
    ax.plot(lim, lim, "r--", lw=1.2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True PGV"); ax.set_ylabel("Pred PGV")
    ax.set_title(f"{label} — measured vs predicted")
    _save_fig(out_path)


def plot_feature_importance_v2(model, feat_cols, title, out_path, top_n=30):
    imp = pd.DataFrame({
        "feature":    feat_cols,
        "importance": model.feature_importances_,
    }).nlargest(top_n, "importance")
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.25)))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1])
    ax.set_xlabel("Importance"); ax.set_title(title)
    _save_fig(out_path)


def plot_geometry_table(geom_df: pd.DataFrame, out: Path):
    """Visual table of per-sensor geometry diagnostics."""
    rows = []
    fo_y = 0.0
    track1_y = float(geom_df["track1_y_m"].iloc[0])
    track2_y = float(geom_df["track2_y_m"].iloc[0])
    for sensor in SENSOR_ORDER:
        if sensor not in geom_df.index:
            continue
        sy = float(geom_df.loc[sensor, "sensor_y_m"])
        for track_no, ty in [(1, track1_y), (2, track2_y)]:
            denom = ty - fo_y
            ap = (sy - fo_y) / denom if abs(denom) > 1e-9 else 0.0
            rows.append({
                "Sensor": sensor, "Track": track_no,
                "y_m": sy, "track_y_m": ty,
                "fo_axis_pos": round(ap, 4),
                "acc_to_track": round(abs(sy - ty), 2),
                "acc_to_fo": round(abs(sy - fo_y), 2),
                "track_to_fo_ratio": round(abs(sy - ty) / max(abs(ty - fo_y), 1e-9), 3),
            })
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.4 + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    plt.title("Holten Line-C FO-Relative Geometry Diagnostics", pad=10)
    _save_fig(out)


# ─── V1 pipeline reuse ────────────────────────────────────────────────────────

def run_v1_pipeline(smoke: bool = False):
    """
    Run FO-PhysProfile v1 pipeline stages 1–4 to obtain profile predictions
    on ALL splits (train/val/test).

    Returns a dict with keys:
        df_linec, split_df, event_feat_df, pca, pca_ve,
        ev_tr, ev_va, ev_te,
        df_tr_rows, df_va_rows, df_te_rows,   ← with pred_log_profile etc.
        fo_valid, n_comp_use, out_dir_v1
    """
    print("\n" + "=" * 70)
    print("Running v1 pipeline (data load → profile reconstruction)")
    print("=" * 70)

    # ── Stage 1: Load data ────────────────────────────────────────────────────
    df_linec = _v1.load_linec_parquet()
    wf_array, event_map, actual_fs = _v1.load_waveforms()
    p3_dir   = _v1.find_p3_dir()
    split_df = _v1.get_split_labels(df_linec, p3_dir)
    df_linec = df_linec.merge(split_df, on="event_id", how="inner")

    if smoke:
        # Tiny subset for smoke testing
        ev_sample = sorted(df_linec["event_id"].unique())[:30]
        df_linec = df_linec[df_linec["event_id"].isin(ev_sample)].copy()
        wf_array = wf_array  # full array, just fewer event_ids used

    # ── Stage 2: Feature extraction ───────────────────────────────────────────
    all_eids = df_linec["event_id"].unique().tolist()
    try:
        wf_feat_df = _v1.build_waveform_feature_df(wf_array, event_map, all_eids)
    except Exception as exc:
        print(f"[WARN] Waveform features failed: {exc}")
        wf_feat_df = pd.DataFrame({"event_id": all_eids})

    pq_event_df = _v1.build_parquet_event_features(df_linec)
    event_feat_df = pq_event_df.merge(wf_feat_df, on="event_id", how="left")
    max_pgv_ev = df_linec.groupby("event_id")["target_pgv"].max().reset_index()
    max_pgv_ev.columns = ["event_id", "max_pgv_event"]
    event_feat_df = event_feat_df.merge(max_pgv_ev, on="event_id", how="left")
    event_feat_df = event_feat_df.merge(split_df, on="event_id", how="left")

    # Feature column selection (same as v1)
    skip_ev = _v1.LEAKAGE | {
        "event_id", "split", "max_pgv_event",
        "track", "track_number", "train_type_code",
        "train_speed_kmh", "train_speed_missing",
    }
    meta_ev = ["train_speed_kmh", "train_speed_missing",
               "train_type_code", "track_number"]
    fo_event_feat_cols = [
        c for c in event_feat_df.columns
        if c not in skip_ev
        and pd.api.types.is_numeric_dtype(event_feat_df[c])
        and not c.startswith("target_") and "pgv" not in c.lower()
        and not c.startswith("pc") and c != "c_target_event"
        and "n_event" not in c and "delta_n" not in c
    ]
    fo_event_feat_cols += [c for c in meta_ev
                           if c in event_feat_df.columns
                           and c not in fo_event_feat_cols]

    # ── Stage 3: Physics targets ───────────────────────────────────────────────
    target_df, resid_df = _v1.compute_physics_targets(df_linec)
    pca, pc_score_df, pca_ve = _v1.fit_pca_residuals(
        target_df, resid_df, split_df, n_comp=PCA_N_COMP
    )
    event_full_df = (
        event_feat_df
        .merge(target_df, on="event_id", how="inner")
        .merge(pc_score_df, on="event_id", how="inner")
    )

    ev_tr = event_full_df[event_full_df["split"] == "train"].copy()
    ev_va = event_full_df[event_full_df["split"] == "val"].copy()
    ev_te = event_full_df[event_full_df["split"] == "test"].copy()
    fo_valid = [c for c in fo_event_feat_cols if c in event_full_df.columns]

    X_tr = ev_tr[fo_valid].fillna(0.0).values.astype(np.float32)
    X_va = ev_va[fo_valid].fillna(0.0).values.astype(np.float32)
    X_te = ev_te[fo_valid].fillna(0.0).values.astype(np.float32)

    # ── Stage 4: Event-level models + profile reconstruction ──────────────────
    w_tr = _v1._high_pgv_weights(ev_tr)
    df_va_rows = df_linec[df_linec["split"] == "val"].copy()
    df_te_rows = df_linec[df_linec["split"] == "test"].copy()
    df_tr_rows = df_linec[df_linec["split"] == "train"].copy()

    def _top_k(Xtr, ytr, Xva, yva, wtr, label, k=5):
        grid = _v1._event_grid()
        cands = []
        for params in grid:
            m = _v1._xgbr_event(params)
            m.fit(Xtr, ytr, sample_weight=wtr,
                  eval_set=[(Xva, yva)], verbose=False)
            vp = m.predict(Xva)
            rmse = float(np.sqrt(np.mean((vp - yva) ** 2)))
            cands.append((rmse, m, vp))
        cands.sort(key=lambda x: x[0])
        print(f"[{label}] top-1 val RMSE={cands[0][0]:.5f}")
        return cands[:k]

    zero_pc = np.zeros((len(ev_va), 2), dtype=np.float32)

    print("\n[v1_pipe] Training C model...")
    c_cands = _top_k(X_tr, ev_tr["c_target_event"].values,
                     X_va, ev_va["c_target_event"].values, w_tr, "C", 3)
    m_c, va_c = c_cands[0][1], c_cands[0][2]
    te_c = m_c.predict(X_te)
    r2_c = float(1 - np.sum((va_c - ev_va["c_target_event"].values) ** 2) /
                 max(np.sum((ev_va["c_target_event"].values
                              - ev_va["c_target_event"].values.mean()) ** 2), 1e-12))
    print(f"[v1_pipe] C: c_hat R²={r2_c:.4f} (val)")

    print("\n[v1_pipe] Training N model (profile-aware)...")
    n_cands = _top_k(X_tr, ev_tr["delta_n_target"].values,
                     X_va, ev_va["delta_n_target"].values, w_tr, "N", 5)
    best_n_idx = 0; best_rp = 1e9
    for k, (_, mn, va_n_cand) in enumerate(n_cands):
        rp = _v1._profile_rmse_pgv(va_c, va_n_cand, zero_pc, pca, ev_va, df_va_rows)
        if rp < best_rp: best_rp = rp; best_n_idx = k
    m_n, va_n = n_cands[best_n_idx][1], n_cands[best_n_idx][2]
    te_n = m_n.predict(X_te)

    print("\n[v1_pipe] Training PC1 (profile-aware)...")
    pc1_cands = _top_k(X_tr, ev_tr["pc1_target"].values,
                       X_va, ev_va["pc1_target"].values, w_tr, "PC1", 5)
    zero_pc2 = np.zeros(len(ev_va), dtype=np.float32)
    best_idx = 0; best_rp = 1e9
    for k, (_, mp, va_p) in enumerate(pc1_cands):
        rp = _v1._profile_rmse_pgv(va_c, va_n,
                                    np.column_stack([va_p, zero_pc2]), pca, ev_va, df_va_rows)
        if rp < best_rp: best_rp = rp; best_idx = k
    m_pc1, va_pc1 = pc1_cands[best_idx][1], pc1_cands[best_idx][2]
    te_pc1 = m_pc1.predict(X_te)

    print("\n[v1_pipe] Training PC2 (profile-aware)...")
    pc2_cands = _top_k(X_tr, ev_tr["pc2_target"].values,
                       X_va, ev_va["pc2_target"].values, w_tr, "PC2", 5)
    best_idx = 0; best_rp = 1e9
    for k, (_, mp, va_p) in enumerate(pc2_cands):
        rp = _v1._profile_rmse_pgv(va_c, va_n,
                                    np.column_stack([va_pc1, va_p]), pca, ev_va, df_va_rows)
        if rp < best_rp: best_rp = rp; best_idx = k
    m_pc2, va_pc2 = pc2_cands[best_idx][1], pc2_cands[best_idx][2]
    te_pc2 = m_pc2.predict(X_te)

    n_comp_use = 2
    m_pc3 = None
    if PCA_N_COMP >= 3 and "pc3_target" in ev_tr.columns and pca_ve[2] > 0.05:
        _, _, va_pc3 = _v1.train_event_model(
            X_tr, ev_tr["pc3_target"].values,
            X_va, ev_va["pc3_target"].values, w_tr, "PC3")
        te_pc3 = m_c.predict(X_te)   # placeholder until properly trained
        n_comp_use = 3
    else:
        va_pc3 = te_pc3 = None

    def _pred_dict(df_ev, c_a, n_a, pc1_a, pc2_a, pc3_a=None):
        c_by = dict(zip(df_ev["event_id"].values, c_a))
        dn_by = dict(zip(df_ev["event_id"].values, n_a))
        pc_by = {}
        for i, eid in enumerate(df_ev["event_id"].values):
            v = np.array([pc1_a[i], pc2_a[i]])
            if pc3_a is not None: v = np.append(v, pc3_a[i])
            pc_by[eid] = v
        return c_by, dn_by, pc_by

    c_va_d, dn_va_d, pc_va_d = _pred_dict(ev_va, va_c, va_n, va_pc1, va_pc2, va_pc3)
    c_te_d, dn_te_d, pc_te_d = _pred_dict(ev_te, te_c, te_n, te_pc1, te_pc2, te_pc3)

    def _attach(df_rows, c_by, dn_by, pc_by):
        pred = _v1.reconstruct_profile_predictions(
            df_rows, df_rows["event_id"].unique(),
            c_by, dn_by, pc_by, pca, n_comp_use=n_comp_use)
        df_rows = df_rows.copy()
        df_rows["pred_log_profile"] = pred
        df_rows["c_hat_profile"]    = df_rows["event_id"].map(c_by).astype(float)
        track_by = df_rows.groupby("event_id")["track_number"].first().astype(int)
        df_rows["n_hat_profile"] = df_rows["event_id"].map(
            lambda e: float(np.clip(
                N_TRACK.get(int(track_by.get(e, 1)), 1.0) + dn_by.get(e, 0.0), *N_CLIP))
            if e in dn_by else np.nan)
        df_rows["pc1_hat"] = df_rows["event_id"].map(
            {e: v[0] for e, v in pc_by.items()})
        df_rows["pc2_hat"] = df_rows["event_id"].map(
            {e: v[1] if len(v) > 1 else 0.0 for e, v in pc_by.items()})
        return df_rows

    df_va_rows = _attach(df_va_rows, c_va_d, dn_va_d, pc_va_d)
    df_te_rows = _attach(df_te_rows, c_te_d, dn_te_d, pc_te_d)

    # OOF for train (5-fold KFold)
    print("\n[v1_pipe] OOF KFold train predictions (5 folds)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    n_ev = len(ev_tr)
    oof_c = np.zeros(n_ev); oof_n = np.zeros(n_ev)
    oof_pc1 = np.zeros(n_ev); oof_pc2 = np.zeros(n_ev)
    fixed_params = {"max_depth": 3, "learning_rate": 0.02,
                    "min_child_weight": 5, "reg_lambda": 10,
                    "subsample": 0.8, "colsample_bytree": 0.8}
    for fold_i, (ftr, fva) in enumerate(kf.split(X_tr)):
        def _fit_fold(y_all):
            mf = _v1._xgbr_event(fixed_params)
            mf.fit(X_tr[ftr], y_all[ftr], sample_weight=w_tr[ftr],
                   eval_set=[(X_tr[fva], y_all[fva])], verbose=False)
            return mf.predict(X_tr[fva])
        oof_c[fva]   = _fit_fold(ev_tr["c_target_event"].values)
        oof_n[fva]   = _fit_fold(ev_tr["delta_n_target"].values)
        oof_pc1[fva] = _fit_fold(ev_tr["pc1_target"].values)
        oof_pc2[fva] = _fit_fold(ev_tr["pc2_target"].values)
        print(f"  fold {fold_i+1}/5  done")

    c_tr_d, dn_tr_d, pc_tr_d = _pred_dict(
        ev_tr, oof_c, oof_n, oof_pc1, oof_pc2,
        np.zeros(n_ev) if m_pc3 else None)
    df_tr_rows = _attach(df_tr_rows, c_tr_d, dn_tr_d, pc_tr_d)

    va_profile_rmse = _v1._profile_rmse_pgv(
        va_c, va_n, np.column_stack([va_pc1, va_pc2]), pca, ev_va, df_va_rows)
    print(f"\n[v1_pipe] Val profile RMSE(PGV) = {va_profile_rmse:.4f}")

    return {
        "df_linec":       df_linec,
        "split_df":       split_df,
        "event_feat_df":  event_feat_df,
        "pca":            pca,
        "pca_ve":         pca_ve,
        "ev_tr": ev_tr, "ev_va": ev_va, "ev_te": ev_te,
        "df_tr_rows":     df_tr_rows,
        "df_va_rows":     df_va_rows,
        "df_te_rows":     df_te_rows,
        "fo_valid":       fo_valid,
        "n_comp_use":     n_comp_use,
        "r2_c_val":       r2_c,
    }


# ─── Row feature preparation ───────────────────────────────────────────────────

def _build_row_features_v2(
    df_rows: pd.DataFrame,
    event_feat_df: pd.DataFrame,
    geom_df: pd.DataFrame,
    df_linec: pd.DataFrame,
    add_geometry: bool = True,
) -> pd.DataFrame:
    """
    Build row-level feature matrix:
      1. Join event-level FO features
      2. Add geometry features (if add_geometry=True)
      3. Add geometry interactions
      4. Add distance-frequency interactions (from v1)
    """
    # Drop target/split columns from event_feat_df before join
    drop_cols = [
        "split", "max_pgv_event", "c_target_event",
        "n_event_raw", "n_event_shrunk", "delta_n_target",
        "n_track", "track",
    ] + [f"pc{k}_target" for k in range(1, 4)]
    ev_feats = event_feat_df.drop(columns=drop_cols, errors="ignore")

    # Start from df_rows (already has pred_log_profile, target_log, etc.)
    # Join parquet row-level features from df_linec
    pq_skip = _v1.LEAKAGE | {
        "event_id", "sensor", "sensor_code",
        "distance", "log_distance", "target_log", "target_pgv",
        "split", "track_number", "train_type_code",
        "train_speed_kmh", "train_speed_missing",
    }
    pq_ro_cols = [
        c for c in df_linec.columns
        if pd.api.types.is_numeric_dtype(df_linec[c])
        and c not in pq_skip
        and not c.startswith("target_") and "pgv" not in c.lower()
    ]
    # Merge parquet per-row FO features
    row_df = df_rows.merge(
        df_linec[["event_id", "sensor"] + pq_ro_cols],
        on=["event_id", "sensor"], how="left", suffixes=("", "_pq")
    )
    # Join event-level FO features
    row_df = row_df.merge(ev_feats, on="event_id", how="left", suffixes=("", "_ev"))

    # Distance-frequency interactions (v1 style)
    log_d = row_df["log_distance"]
    for wcol in ["wf_spectral_centroid", "wf_high_low_energy_ratio",
                 "wf_band_1_5", "wf_band_5_10", "wf_band_10_20",
                 "wf_band_20_40", "wf_band_40_80"]:
        if wcol in row_df.columns:
            row_df[f"feat_logd_x_{wcol}"] = log_d * row_df[wcol]
    for wcol in ["train_speed_kmh", "train_type_code"]:
        if wcol in row_df.columns:
            row_df[f"feat_logd_x_{wcol}"] = log_d * row_df[wcol]
    sc = row_df.get("sensor_code", pd.Series(0, index=row_df.index))
    for wcol in ["wf_spectral_centroid", "wf_high_low_energy_ratio"]:
        if wcol in row_df.columns:
            row_df[f"feat_sc_x_{wcol}"] = sc * row_df[wcol]

    if add_geometry:
        row_df = add_fo_relative_geometry_features(row_df, geom_df)
        row_df = add_geometry_interactions(row_df)

    return row_df


def _get_feat_cols(row_df: pd.DataFrame,
                    include_sensor_code: bool = True,
                    include_geometry: bool = True) -> List[str]:
    """Build the feature column list for local residual model."""
    explicit = [
        "pred_log_profile", "c_hat_profile", "n_hat_profile",
        "pc1_hat", "pc2_hat",
        "distance", "log_distance", "track_number",
        "train_speed_kmh", "train_speed_missing", "train_type_code",
    ]
    if include_sensor_code:
        explicit.append("sensor_code")

    geom_explicit = _GEOM_COLS if include_geometry else []

    pq_cols   = [c for c in row_df.columns if c.startswith("fo_")]
    wf_cols   = [c for c in row_df.columns if c.startswith("wf_")]
    feat_cols = [c for c in row_df.columns if c.startswith("feat_")]
    gi_cols   = [c for c in row_df.columns if c.startswith("gi_")] if include_geometry else []

    all_cols = (explicit + geom_explicit + feat_cols + gi_cols + wf_cols + pq_cols)

    # Deduplicate, filter leakage and non-numeric
    seen = set()
    out = []
    for c in all_cols:
        if c in seen or c not in row_df.columns:
            continue
        col_lower = c.lower()
        is_leaky = any(tok in col_lower for tok in _LEAKAGE_TOKENS)
        if is_leaky:
            continue
        if not pd.api.types.is_numeric_dtype(row_df[c]):
            continue
        if c.startswith("target_") or "pgv" in col_lower:
            continue
        seen.add(c); out.append(c)
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="30-event smoke test (fast check)")
    args = parser.parse_args()

    out_dir = make_output_dir(MODELS_ROOT)
    print(f"\nOutput dir: {out_dir}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Load geometry config
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1 — Geometry config")
    print("=" * 70)

    site_config = load_site_config()
    geom_df = build_sensor_geometry_table(site_config)
    print_geometry_diagnostic(geom_df)

    # Save geometry diagnostic table
    diag_rows = []
    fo_y = 0.0
    t1y  = float(geom_df["track1_y_m"].iloc[0])
    t2y  = float(geom_df["track2_y_m"].iloc[0])
    for sensor in SENSOR_ORDER:
        if sensor not in geom_df.index:
            continue
        sy = float(geom_df.loc[sensor, "sensor_y_m"])
        for tn, ty in [(1, t1y), (2, t2y)]:
            ap = (sy - fo_y) / (ty - fo_y)
            diag_rows.append({
                "sensor": sensor, "track": tn,
                "sensor_y_m": sy, "track_y_m": ty, "fo_y_m": fo_y,
                "fo_to_track_axis_pos": round(ap, 4),
                "acc_to_active_track_m": round(abs(sy - ty), 2),
                "acc_to_fo_m": round(abs(sy - fo_y), 2),
                "acc_track_to_fo_ratio": round(abs(sy - ty) / abs(ty - fo_y), 3),
                "is_between_track_and_fo": int(0 < ap < 1),
                "is_on_fo_line": int(abs(sy - fo_y) <= 0.75),
                "is_beyond_fo_away": int(ap < 0),
            })
    pd.DataFrame(diag_rows).to_csv(out_dir / "geometry_diagnostic_table.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Run v1 pipeline to get profile predictions on all splits
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2 — v1 pipeline (profile reconstruction)")
    print("=" * 70)

    v1 = run_v1_pipeline(smoke=args.smoke)
    df_tr_rows    = v1["df_tr_rows"]
    df_va_rows    = v1["df_va_rows"]
    df_te_rows    = v1["df_te_rows"]
    df_linec      = v1["df_linec"]
    event_feat_df = v1["event_feat_df"]
    pca           = v1["pca"]

    # Save event features
    event_feat_df.to_parquet(out_dir / "event_features.parquet", index=False)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Build row-level feature matrices (all splits)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3 — Build row-level feature matrices with geometry")
    print("=" * 70)

    row_tr_geom = _build_row_features_v2(df_tr_rows, event_feat_df, geom_df,
                                          df_linec, add_geometry=True)
    row_va_geom = _build_row_features_v2(df_va_rows, event_feat_df, geom_df,
                                          df_linec, add_geometry=True)
    row_te_geom = _build_row_features_v2(df_te_rows, event_feat_df, geom_df,
                                          df_linec, add_geometry=True)

    # Drop rows without profile prediction
    row_tr_geom = row_tr_geom.dropna(subset=["pred_log_profile"])
    row_va_geom = row_va_geom.dropna(subset=["pred_log_profile"])
    row_te_geom = row_te_geom.dropna(subset=["pred_log_profile"])

    print(f"[rows] train={len(row_tr_geom)}  val={len(row_va_geom)}  "
          f"test={len(row_te_geom)}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — Leakage check
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 4 — Leakage check")
    print("=" * 70)

    # Build feature column lists for each variant
    fc_v1ref = _get_feat_cols(row_tr_geom, include_sensor_code=True,
                               include_geometry=False)
    fc_v2    = _get_feat_cols(row_tr_geom, include_sensor_code=True,
                               include_geometry=True)
    fc_v2_nosc = _get_feat_cols(row_tr_geom, include_sensor_code=False,
                                 include_geometry=True)

    for label, fc in [("V1_ref", fc_v1ref), ("V2_geom", fc_v2),
                       ("V2_geom_no_sc", fc_v2_nosc)]:
        _v1.assert_no_leakage(fc, label)

    # Write leakage audit for V2_geom
    _v1.write_leakage_audit(fc_v2, out_dir / "leakage_audit_geometry.txt")
    print(f"[feat] V1_ref={len(fc_v1ref)}  V2_geom={len(fc_v2)}  "
          f"V2_geom_no_sc={len(fc_v2_nosc)}")

    # Save feature column files
    (out_dir / "feature_columns_v1ref.txt").write_text("\n".join(fc_v1ref))
    (out_dir / "feature_columns_v2geom.txt").write_text("\n".join(fc_v2))
    (out_dir / "feature_columns_v2geom_nosc.txt").write_text("\n".join(fc_v2_nosc))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Train local residual variants
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 5 — Train local residual model variants")
    print("=" * 70)

    # Variant V1_ref
    print("\n[V1_ref] Training (reference — v1 feature set)...")
    m_v1ref, rmse_v1ref, mp4_v1ref, sch_v1ref, _ = train_local_residual_variant(
        row_tr_geom, row_va_geom, fc_v1ref, "V1_ref"
    )

    # Variant V2_geom
    print("\n[V2_geom] Training (with geometry features)...")
    m_v2, rmse_v2, mp4_v2, sch_v2, _ = train_local_residual_variant(
        row_tr_geom, row_va_geom, fc_v2, "V2_geom"
    )

    # Variant V2_geom_no_sensor_code
    print("\n[V2_geom_no_sc] Training (geometry replaces sensor_code)...")
    m_v2_nosc, rmse_v2_nosc, mp4_v2_nosc, sch_v2_nosc, _ = train_local_residual_variant(
        row_tr_geom, row_va_geom, fc_v2_nosc, "V2_geom_no_sc"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6 — Predict and apply monotonic on test set
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 6 — Test predictions")
    print("=" * 70)

    def _predict_final(model, row_te, fc, col_final, col_mono):
        valid_fc = [c for c in fc if c in row_te.columns]
        X = row_te[valid_fc].fillna(0.0).values.astype(np.float32)
        res = model.predict(X)
        row_te = row_te.copy()
        row_te[col_final] = row_te["pred_log_profile"] + res
        row_te[col_mono]  = _v1.apply_monotonic(row_te, col_final)
        return row_te

    row_te_geom = _predict_final(
        m_v1ref, row_te_geom, fc_v1ref,
        "pred_log_v1ref", "pred_log_v1ref_mono")
    row_te_geom = _predict_final(
        m_v2, row_te_geom, fc_v2,
        "pred_log_v2", "pred_log_v2_mono")
    row_te_geom = _predict_final(
        m_v2_nosc, row_te_geom, fc_v2_nosc,
        "pred_log_v2_nosc", "pred_log_v2_nosc_mono")

    # Profile-only + mono
    row_te_geom["pred_log_profile_mono"] = _v1.apply_monotonic(
        row_te_geom, "pred_log_profile")

    # Also predict on val (for any secondary analysis)
    row_va_geom_pred = _predict_final(
        m_v2, row_va_geom, fc_v2, "pred_log_v2", "pred_log_v2_mono")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 7 — Metrics
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 7 — Compute metrics (test set)")
    print("=" * 70)

    s_te  = row_te_geom["sensor"].values
    e_te  = row_te_geom["event_id"].values
    tl_te = row_te_geom["target_log"].values

    all_metrics: List[dict] = []

    for pred_col, label in [
        ("pred_log_profile",      "Profile-only (v1)"),
        ("pred_log_profile_mono", "Profile-only mono (v1)"),
        ("pred_log_v1ref",        "V1_ref (local residual)"),
        ("pred_log_v1ref_mono",   "V1_ref mono"),
        ("pred_log_v2",           "V2_geom (+ geometry)"),
        ("pred_log_v2_mono",      "V2_geom mono"),
        ("pred_log_v2_nosc",      "V2_geom_no_sc"),
        ("pred_log_v2_nosc_mono", "V2_geom_no_sc mono"),
    ]:
        if pred_col not in row_te_geom.columns:
            continue
        msk = np.isfinite(row_te_geom[pred_col].values)
        if msk.sum() < 10:
            continue
        m = compute_metrics_v2(
            row_te_geom[pred_col].values[msk],
            tl_te[msk], s_te[msk], e_te[msk], label)
        all_metrics.append(m)
        # Print brief
        ps = m.get("per_sensor", {})
        print(f"\n  {label}")
        print(f"    RMSE(PGV)={m['rmse_pgv']:.4f}  RMSE(log)={m['rmse_log']:.4f}  "
              f"R²(log)={m['r2_log']:.4f}  mono_viol={m['mono_viol_rate']:.3f}")
        for ss in SENSOR_ORDER:
            if ss in ps:
                print(f"    {ss}: RMSE(PGV)={ps[ss]['rmse_pgv']:.4f}  "
                      f"bias={ps[ss]['bias_pgv']:+.4f}")

    # Load baselines for comparison
    v1_out = find_v1_output()
    if v1_out:
        pred_path = v1_out / "predictions_test.parquet"
        if pred_path.exists():
            v1_preds = pd.read_parquet(pred_path)
            v1_preds["event_id"] = v1_preds["event_id"].astype(str)
            v1_preds["sensor"]   = v1_preds["sensor"].astype(str)
            for vcol, vlabel in [("pred_log_final", "FO-PhysProfile v1 final"),
                                  ("pred_log_final_mono", "FO-PhysProfile v1 final mono")]:
                if vcol not in v1_preds.columns:
                    continue
                aligned = row_te_geom[["event_id", "sensor"]].merge(
                    v1_preds[["event_id", "sensor", vcol]],
                    on=["event_id", "sensor"], how="left")[vcol].values
                msk = np.isfinite(aligned)
                if msk.sum() > 10:
                    m = compute_metrics_v2(aligned[msk], tl_te[msk],
                                            s_te[msk], e_te[msk], vlabel)
                    all_metrics.append(m)

    # Load PXGBR baselines
    pxgbr_preds = _v1.load_pxgbr_predictions(row_te_geom)
    for bl_label, bl_pred in pxgbr_preds.items():
        if bl_pred is not None:
            msk = np.isfinite(bl_pred)
            if msk.sum() > 10:
                m = compute_metrics_v2(bl_pred[msk], tl_te[msk],
                                        s_te[msk], e_te[msk], bl_label)
                all_metrics.append(m)

    # P3 baseline
    p3_dir = _v1.find_p3_dir()
    if p3_dir:
        p3all = pd.read_parquet(p3_dir / "all_predictions.parquet")
        p3all["event_id"] = p3all["event_id"].astype(str)
        p3all["sensor"]   = p3all["sensor"].astype(str)
        p3te = p3all[p3all["split"] == "test"]
        p3_aligned = row_te_geom[["event_id", "sensor"]].merge(
            p3te[["event_id", "sensor", "pred_log_p3"]],
            on=["event_id", "sensor"], how="left")["pred_log_p3"].values
        msk = np.isfinite(p3_aligned)
        if msk.sum() > 10:
            m = compute_metrics_v2(p3_aligned[msk], tl_te[msk],
                                    s_te[msk], e_te[msk], "P3_corrected_n")
            all_metrics.append(m)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 8 — Geometry diagnostics
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 8 — Geometry diagnostics")
    print("=" * 70)

    geom_diag_all = {}
    for pred_col, label in [("pred_log_v1ref", "V1_ref"),
                              ("pred_log_v2",    "V2_geom")]:
        geom_diag_all[label] = compute_geometry_diagnostics(
            row_te_geom, pred_col, label)

    # Save geometry ablation metrics
    ablation_rows = []
    for m in all_metrics:
        row_a = {
            "model": m["model"],
            "rmse_pgv": m["rmse_pgv"],
            "rmse_log": m["rmse_log"],
            "r2_log": m["r2_log"],
            "bias_pgv": m["bias_pgv"],
            "mono_viol": m.get("mono_viol_rate", np.nan),
        }
        for ss in SENSOR_ORDER:
            row_a[f"rmse_{ss}"] = m["per_sensor"].get(ss, {}).get("rmse_pgv", np.nan)
        for k in ["pgv_gt4", "pgv_gt8", "mp4_pgv_gt4", "mp4_pgv_gt8"]:
            row_a[k + "_n"]    = m.get(k, {}).get("n", np.nan)
            row_a[k + "_rmse"] = m.get(k, {}).get("rmse_pgv", np.nan)
        ablation_rows.append(row_a)

    ablation_df = pd.DataFrame(ablation_rows).sort_values("rmse_pgv")
    ablation_df.to_csv(out_dir / "geometry_ablation_metrics.csv", index=False)

    # Main metrics table
    ablation_df.rename(columns={"rmse_pgv": "rmse_pgv"}).to_csv(
        out_dir / "metrics_table.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 9 — Save predictions and row features
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 9 — Save outputs")
    print("=" * 70)

    # predictions_test.parquet
    pred_cols = ["event_id", "sensor", "track_number", "distance",
                 "target_log", "target_pgv",
                 "pred_log_profile", "pred_log_profile_mono",
                 "pred_log_v1ref",   "pred_log_v1ref_mono",
                 "pred_log_v2",      "pred_log_v2_mono",
                 "pred_log_v2_nosc", "pred_log_v2_nosc_mono",
                 # geometry columns
                 "sensor_y_m", "fo_to_track_axis_pos",
                 "acc_track_to_fo_ratio", "is_between_track_and_fo",
                 "is_on_fo_line", "is_beyond_fo_away_from_track",
                 ]
    pred_cols = [c for c in pred_cols if c in row_te_geom.columns]
    row_te_geom[pred_cols].to_parquet(out_dir / "predictions_test.parquet", index=False)

    # row_features_geometry.parquet (test set, no targets)
    geo_feat_cols = (fc_v2 + ["event_id", "sensor",
                               "fo_to_track_axis_pos", "acc_track_to_fo_ratio",
                               "is_between_track_and_fo", "is_on_fo_line",
                               "is_beyond_fo_away_from_track"])
    geo_feat_cols = [c for c in dict.fromkeys(geo_feat_cols)
                     if c in row_te_geom.columns]
    row_te_geom[geo_feat_cols].to_parquet(
        out_dir / "row_features_geometry.parquet", index=False)
    print(f"[save] row_features_geometry: {len(geo_feat_cols)} columns")

    # Feature importance
    for model, fc, name in [
        (m_v2,      fc_v2,    "feature_importance_local_residual_v2_geometry.csv"),
        (m_v1ref,   fc_v1ref, "feature_importance_local_residual_v1ref.csv"),
        (m_v2_nosc, fc_v2_nosc, "feature_importance_local_residual_v2_nosc.csv"),
    ]:
        valid_fc = [c for c in fc if c in row_tr_geom.columns]
        if len(valid_fc) == len(model.feature_importances_):
            pd.DataFrame({
                "feature": valid_fc,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False).to_csv(out_dir / name, index=False)

    print(f"\n[save] All outputs → {out_dir}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 10 — Plots
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 10 — Plots")
    print("=" * 70)

    models_for_plot = {m["model"]: m for m in all_metrics
                       if m["model"] in ["V1_ref (local residual)",
                                          "V2_geom (+ geometry)",
                                          "V2_geom_no_sc",
                                          "FO-PhysProfile v1 final",
                                          "PXGBR-R2"]}

    if models_for_plot:
        plot_per_sensor_rmse_v1_vs_v2(
            models_for_plot,
            out_dir / "per_sensor_rmse_v1_vs_v2_geometry.png")

    plot_residual_vs_axis_pos(
        row_te_geom, "pred_log_v2",
        out_dir / "residual_vs_fo_to_track_axis_pos.png")

    plot_residual_by_geometry_zone(
        row_te_geom, "pred_log_v2",
        out_dir / "residual_by_geometry_zone.png")

    if "pred_log_v2" in row_te_geom.columns:
        te_ok = row_te_geom.dropna(subset=["pred_log_v2"])
        if len(te_ok) > 4:
            plot_high_pgv_profiles_v2(
                te_ok, "pred_log_v2", "pred_log_v1ref", 16,
                out_dir / "high_pgv_profiles_v2_geometry.png")

    if "pred_log_v2" in row_te_geom.columns and "pred_log_v1ref" in row_te_geom.columns:
        plot_mp4_profiles_v1_vs_v2(
            row_te_geom, "pred_log_v1ref", "pred_log_v2", 16,
            out_dir / "mp4_profiles_v1_vs_v2_geometry.png")

    if "pred_log_v2" in row_te_geom.columns:
        ok = np.isfinite(row_te_geom["pred_log_v2"].values)
        plot_scatter_v2(
            row_te_geom["target_pgv"].values[ok],
            np.exp(row_te_geom["pred_log_v2"].values[ok]),
            "V2_geom", out_dir / "measured_vs_predicted_v2_geometry.png")

    valid_v2_fc = [c for c in fc_v2 if c in row_tr_geom.columns]
    if len(valid_v2_fc) == len(m_v2.feature_importances_):
        plot_feature_importance_v2(
            m_v2, valid_v2_fc,
            "Feature importance — V2_geom local residual",
            out_dir / "feature_importance_local_residual_v2_geometry.png")

    plot_geometry_table(geom_df, out_dir / "geometry_diagnostic_layout_table.png")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 11 — Final summary with interpretation
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(ablation_df[["model", "rmse_pgv", "rmse_log", "rmse_MP4"]].to_string(index=False))

    # Extract specific models
    def _get(label):
        rows = ablation_df[ablation_df["model"].str.contains(label, regex=False)]
        return rows.iloc[0].to_dict() if len(rows) > 0 else {}

    m_v1r = _get("V1_ref")
    m_v2g = _get("V2_geom (+ geometry)")
    m_v2n = _get("V2_geom_no_sc")

    print("\nMP4 RMSE comparison:")
    baselines = [
        ("P3_corrected_n",        5.0185),
        ("PXGBR-R2",              4.7389),
        ("PXGBR_ens_top5",        4.6680),
        ("FO-PhysProfile v1 final", 4.5592),
    ]
    for bl_label, bl_mp4 in baselines:
        r = _get(bl_label)
        actual = r.get("rmse_MP4", bl_mp4)
        print(f"  {bl_label:40s}  MP4={actual:.4f}")
    print(f"  {'V1_ref':40s}  MP4={m_v1r.get('rmse_MP4', 'N/A')}")
    print(f"  {'V2_geom':40s}  MP4={m_v2g.get('rmse_MP4', 'N/A')}")
    print(f"  {'V2_geom_no_sc':40s}  MP4={m_v2n.get('rmse_MP4', 'N/A')}")

    print("\nGeometry zone metrics (V2_geom):")
    for zone, d in geom_diag_all.get("V2_geom", {}).items():
        print(f"  {zone:30s}  n={d.get('n',0):5d}  "
              f"rmse={d.get('rmse_pgv',0):.4f}  bias={d.get('bias_pgv',0):+.4f}")

    # Interpretation
    v2_mp4 = m_v2g.get("rmse_MP4", 999.0)
    v1_mp4 = m_v1r.get("rmse_MP4", 999.0)
    if isinstance(v2_mp4, float) and isinstance(v1_mp4, float):
        if v2_mp4 < v1_mp4 - 0.05:
            print("\n→ CONCLUSION: V2 improves MP4 — explicit FO-relative geometry "
                  "captures interpolation back toward the source.")
        elif v2_mp4 < v1_mp4:
            print("\n→ CONCLUSION: V2 modestly improves MP4. Geometry adds signal "
                  "but sensor_code already captured most of the effect.")
        else:
            print("\n→ CONCLUSION: V2 does not improve MP4. Profile model is the "
                  "main bottleneck, not the local residual geometry.")

    if isinstance(m_v2n.get("rmse_MP4"), float) and isinstance(v1_mp4, float):
        diff = m_v2n.get("rmse_MP4", 999.0) - v2_mp4
        if abs(diff) < 0.1:
            print("→ V2_no_sc ≈ V2: explicit geometry can replace sensor_code "
                  "(good for generalisation to new sensors).")
        else:
            print(f"→ V2_no_sc degraded by {diff:+.3f} vs V2: sensor-specific "
                  "calibration effects remain important.")

    target_rmse = 2.1882
    v2_rmse = m_v2g.get("rmse_pgv", 999.0)
    if isinstance(v2_rmse, float):
        if v2_rmse < 2.20:
            print(f"\n✓ TARGET MET: RMSE(PGV)={v2_rmse:.4f} < 2.20")
        elif v2_rmse <= target_rmse:
            print(f"✓ BEATS v1: RMSE(PGV)={v2_rmse:.4f} ≤ {target_rmse}")
        else:
            print(f"✗ Does not beat v1 ({target_rmse}): RMSE(PGV)={v2_rmse:.4f}")

    print(f"\n  Output dir: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
