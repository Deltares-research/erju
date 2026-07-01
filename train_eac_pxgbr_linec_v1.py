"""
train_eac_pxgbr_linec_v1.py
EAC-PXGB: Event-Amplitude Calibrated Physics-Boosted XGBoost

Three-stage pipeline:
  Stage 0:  Event-level high-PGV gate       (classifier)
  Stage 1:  Event-level amplitude correction (delta_c regressor)
  Stage 2:  Row-level local residual         (PXGB residual regressor)

Final prediction:
  pred_log_final = pred_log_p3 + delta_c_hat + local_residual_hat

Where:
  delta_c_hat     corrects c_hat event-by-event
  local_residual  corrects sensor-specific deviations from the corrected curve

Expected targets:
  RMSE(PGV) < 2.20 (vs P3: 2.40, vs PXGBR-R2: 2.27)
  MP4 RMSE  < 4.70 (vs P3: 5.02, vs PXGBR-R2: 4.74)

Usage:
    python train_eac_pxgbr_linec_v1.py
    python train_eac_pxgbr_linec_v1.py --p3_dir /path/to/p3_savepred_dir
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import average_precision_score

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))


def _get_data_root() -> Path:
    return Path(r"P:\11210978-erju-ai") if os.name == "nt" else Path("/p/11210978-erju-ai")


MODELS_ROOT  = _get_data_root() / "holten_models"
PARQUET_ROOT = _get_data_root() / "holten_parquet"
R0           = 10.0
SENSOR_ORDER = ["MP4", "MP8", "MP10", "MP1", "MP2"]

# Columns that must never appear as inference features
LEAKAGE_COLS = {
    "target_pgv_z_mms", "target_log", "target_pgv",
    "c_target",           # derived from true label
    "delta_c_target",     # derived from true label
    "high_pgv_event",     # derived from true label
    "high_mp4_event",     # derived from true label
    "residual_log",       # derived from true label
    "local_residual",     # derived from true label
    # identifiers
    "event_id", "split", "sensor", "sensor_id", "site_id",
    "train_type",         # string
    "acc_side_of_track",  # constant
}


# ===========================================================================
# DISCOVERY
# ===========================================================================

def find_p3_dir(override: Optional[str] = None) -> Path:
    if override:
        return Path(override)
    hits = sorted(MODELS_ROOT.glob("cnn_curveprior_p3_savepred_linec_v001_vP3_*"),
                  key=lambda p: p.name)
    if not hits:
        raise FileNotFoundError(
            "No P3 savepred directory found. "
            "Run: python train_cnn_curveprior_linec_v1_savepred.py "
            "--variant P3 --n_mode fit_corrected --lambda_residual 0.05"
        )
    return hits[-1]


def find_parquet_v2() -> Path:
    hits = sorted(PARQUET_ROOT.glob("parquet_v002_*"), key=lambda p: p.name)
    if not hits:
        raise FileNotFoundError("No parquet_v002_* builds found")
    return hits[-1] / "dataset.parquet"


def find_pxgbr_dir() -> Optional[Path]:
    hits = sorted(MODELS_ROOT.glob("pxgbr_linec_v001_*"), key=lambda p: p.name)
    return hits[-1] if hits else None


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_p3(p3_dir: Path) -> pd.DataFrame:
    p = p3_dir / "all_predictions.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run savepred script first")
    df = pd.read_parquet(p)
    print(f"P3 predictions: {len(df):,} rows | splits: {df.split.value_counts().to_dict()}")
    return df


def load_fo(parquet_v2: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_v2)
    df = df[df["sensor_id"].isin(SENSOR_ORDER)].copy()
    df = df.dropna(subset=["target_pgv_z_mms", "track_number"])
    df = df[(df["target_pgv_z_mms"] > 0) & (df["track_number"].isin([1, 2]))]
    df["event_id"] = df["event_id"].astype(str)
    df["sensor"]   = df["sensor_id"]
    print(f"FO features: {len(df):,} rows, {df.columns.tolist()[:5]}...")
    return df


# ===========================================================================
# FEATURE IDENTIFICATION
# ===========================================================================

def fo_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return all non-leakage numeric FO feature columns."""
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and c not in LEAKAGE_COLS
            and not c.startswith("target_")
            and "pgv" not in c.lower()
            and c not in ("track_number", "train_speed_kmh", "train_type_code",
                          "acc_distance_to_track_m", "acc_distance_to_track_2_m",
                          "effective_distance_to_active_track_m")]


# ===========================================================================
# EVENT-LEVEL TABLE
# ===========================================================================

def build_event_table(p3_df: pd.DataFrame, fo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one-row-per-event table for Stages 0 and 1.

    P3 features  : c_hat_p3, epsilon_p3 stats, n_used, track
    Metadata     : train_speed_kmh, train_speed_missing, train_type_code
    FO aggregates: mean/std/min/max over 5 sensors per event
    Targets      : c_target_event, delta_c_target (leakage — excluded from features)
    Labels       : high_pgv_event, high_mp4_event (leakage — excluded from features)
    """
    print("\nBuilding event-level table...")

    # FO feature column names
    fo_cols = fo_feature_cols(fo_df)

    # Join P3 + FO on event_id + sensor
    merged = p3_df.merge(fo_df[["event_id", "sensor"] + fo_cols +
                                ["train_speed_kmh", "train_type_code"]],
                         on=["event_id", "sensor"], how="inner")
    print(f"  After join: {len(merged):,} rows")

    rows = []
    for event_id, grp in merged.groupby("event_id"):
        if len(grp) != 5:
            continue

        split     = grp["split"].iloc[0]
        track     = int(grp["track"].iloc[0])
        c_hat_p3  = float(grp["c_hat_p3"].iloc[0])
        n_used    = float(grp["n_used"].iloc[0])

        # c_target per sensor = target_log + n_used * log(distance / r0)
        c_tgt_per_sensor = grp["c_target"].values
        c_target_event   = float(c_tgt_per_sensor.mean())
        delta_c_target   = c_target_event - c_hat_p3

        # P3 epsilon stats (5 sensors)
        eps = grp["epsilon_p3"].values
        p3_feats = {
            "c_hat_p3":       c_hat_p3,
            "epsilon_p3_mean": float(eps.mean()),
            "epsilon_p3_std":  float(eps.std()),
            "epsilon_p3_max":  float(eps.max()),
            "epsilon_p3_min":  float(eps.min()),
            "epsilon_p3_p95":  float(np.percentile(eps, 95)),
            "n_used":          n_used,
            "track":           float(track),
        }

        # Metadata (event-level — same for all sensor rows)
        speed_raw = grp["train_speed_kmh"].iloc[0]
        meta_feats = {
            "train_speed_kmh":     float(speed_raw) if pd.notna(speed_raw) else 0.0,
            "train_speed_missing": float(pd.isna(speed_raw)),
            "train_type_code":     float(grp["train_type_code"].iloc[0])
                                   if "train_type_code" in grp.columns else -1.0,
        }

        # FO aggregates over 5 sensors
        fo_feats = {}
        for col in fo_cols:
            v = grp[col].values.astype(float)
            fo_feats[f"{col}__mean"] = float(np.mean(v))
            fo_feats[f"{col}__std"]  = float(np.std(v))
            fo_feats[f"{col}__min"]  = float(np.min(v))
            fo_feats[f"{col}__max"]  = float(np.max(v))

        # Labels (not features!)
        max_pgv    = float(grp["target_pgv"].max())
        mp4_pgv    = float(grp[grp["sensor"] == "MP4"]["target_pgv"].values[0])
        high_pgv   = int(max_pgv > 8.0)
        high_mp4   = int(mp4_pgv > 8.0)

        row = {
            "event_id":        event_id,
            "split":           split,
            # targets (leakage — excluded from features)
            "c_target_event":  c_target_event,
            "delta_c_target":  delta_c_target,
            "high_pgv_event":  high_pgv,
            "high_mp4_event":  high_mp4,
            "max_pgv":         max_pgv,
            "mp4_pgv":         mp4_pgv,
            **p3_feats,
            **meta_feats,
            **fo_feats,
        }
        rows.append(row)

    ev_df = pd.DataFrame(rows)
    print(f"  Event table: {len(ev_df):,} events")
    print(f"  high_pgv_event (>8 mm/s): {ev_df['high_pgv_event'].sum()} / {len(ev_df)}")
    return ev_df


def event_feature_cols(ev_df: pd.DataFrame) -> List[str]:
    """Return inference-safe event-level feature columns."""
    excluded = LEAKAGE_COLS | {
        "split", "event_id",
        "c_target_event", "delta_c_target",
        "high_pgv_event", "high_mp4_event",
        "max_pgv", "mp4_pgv",
        "p_high_event",  # added later — handled separately
    }
    return [c for c in ev_df.columns
            if c not in excluded
            and pd.api.types.is_numeric_dtype(ev_df[c])]


# ===========================================================================
# ROW-LEVEL TABLE (for Stage 2)
# ===========================================================================

def build_row_table(p3_df: pd.DataFrame, fo_df: pd.DataFrame,
                    ev_df: pd.DataFrame) -> pd.DataFrame:
    """Join P3 predictions with FO features for Stage 2 row-level model."""
    fo_cols = fo_feature_cols(fo_df)
    merged  = p3_df.merge(
        fo_df[["event_id", "sensor"] + fo_cols +
              ["train_speed_kmh", "train_type_code"]],
        on=["event_id", "sensor"], how="inner"
    )
    # Add event-level features (c_hat, p_high_event, etc.) via merge
    ev_small = ev_df[["event_id", "p_high_event",
                       "epsilon_p3_mean", "epsilon_p3_std",
                       "epsilon_p3_max", "epsilon_p3_min",
                       "epsilon_p3_p95",
                       "high_pgv_event", "high_mp4_event"]].copy()
    merged = merged.merge(ev_small, on="event_id", how="left")

    # Physics-derived row features
    merged["log_distance"]  = np.log(merged["distance"] / R0)
    merged["sensor_code"]   = merged["sensor"].map(
        {s: i for i, s in enumerate(SENSOR_ORDER)}
    ).astype(float)
    speed = merged["train_speed_kmh"]
    merged["train_speed_missing"] = speed.isna().astype(float)
    merged["train_speed_kmh"]     = speed.fillna(0.0)
    return merged


def row_feature_cols(row_df: pd.DataFrame, extra_cols: List[str] = None) -> List[str]:
    """Return inference-safe row-level feature columns."""
    excluded = LEAKAGE_COLS | {
        "split", "event_id", "sensor",
        "target_log", "target_pgv",
        "pred_log_eac_c",  # added in Stage 2, included explicitly
        "local_residual",
        "high_pgv_event", "high_mp4_event",
        "c_target",
    }
    base = [c for c in row_df.columns
            if c not in excluded
            and pd.api.types.is_numeric_dtype(row_df[c])]
    if extra_cols:
        base = list(dict.fromkeys(base + extra_cols))
    return base


# ===========================================================================
# XGBOOST HELPERS
# ===========================================================================

def _xgb_regressor(params: Dict, n_est: int = 2000, esr: int = 50) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        **params,
        n_estimators=n_est,
        early_stopping_rounds=esr,
        eval_metric="rmse",
        tree_method="hist",
        verbosity=0,
    )


def _xgb_classifier(params: Dict, n_est: int = 2000, esr: int = 50) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **params,
        n_estimators=n_est,
        early_stopping_rounds=esr,
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
        use_label_encoder=False,
    )


BASE_PARAMS = {
    "max_depth":        3,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_lambda":       5,
    "min_child_weight": 5,
    "random_state":     42,
}


# ===========================================================================
# METRICS
# ===========================================================================

def compute_metrics(pred_log: np.ndarray, target_log: np.ndarray,
                    sensors: np.ndarray, event_ids: np.ndarray,
                    label: str) -> Dict:
    pred_pgv   = np.exp(pred_log)
    target_pgv = np.exp(target_log)

    def _rmse(a, b):  return float(np.sqrt(np.mean((a - b) ** 2)))
    def _mae(a, b):   return float(np.mean(np.abs(a - b)))
    def _bias(a, b):  return float(np.mean(a - b))
    def _r2(a, b):
        ss = np.sum((b - a) ** 2); st = np.sum((b - b.mean()) ** 2)
        return float(1.0 - ss / st) if st > 0 else 0.0

    m: Dict = {
        "model":    label,
        "rmse_log": _rmse(pred_log, target_log),
        "rmse_pgv": _rmse(pred_pgv, target_pgv),
        "mae_pgv":  _mae(pred_pgv, target_pgv),
        "r2_log":   _r2(pred_log, target_log),
        "bias_pgv": _bias(pred_pgv, target_pgv),
    }

    m["per_sensor"] = {}
    for s in SENSOR_ORDER:
        mask = sensors == s
        if mask.any():
            m["per_sensor"][s] = {
                "rmse_log": _rmse(pred_log[mask], target_log[mask]),
                "rmse_pgv": _rmse(pred_pgv[mask], target_pgv[mask]),
                "bias_pgv": _bias(pred_pgv[mask], target_pgv[mask]),
                "bias_log": _bias(pred_log[mask], target_log[mask]),
            }

    for thr in [4.0, 8.0]:
        hi = target_pgv > thr
        if hi.any():
            m[f"pgv_gt{int(thr)}"] = {
                "n": int(hi.sum()),
                "rmse_pgv": _rmse(pred_pgv[hi], target_pgv[hi]),
                "bias_pgv": _bias(pred_pgv[hi], target_pgv[hi]),
            }

    mp4 = sensors == "MP4"
    for thr in [4.0, 8.0]:
        hi = mp4 & (target_pgv > thr)
        if hi.any():
            m[f"mp4_pgv_gt{int(thr)}"] = {
                "n": int(hi.sum()),
                "rmse_pgv": _rmse(pred_pgv[hi], target_pgv[hi]),
            }

    # Monotonicity
    violations, total = 0, 0
    for eid in np.unique(event_ids)[:300]:
        ev_m = event_ids == eid
        if ev_m.sum() != 5:
            continue
        # Sort by sensor order (near → far): assumes all 5 sensors present
        ev_sub = pd.DataFrame({
            "sensor": sensors[ev_m], "pred_log": pred_log[ev_m]
        }).set_index("sensor")
        pv = [ev_sub.loc[s, "pred_log"] for s in SENSOR_ORDER if s in ev_sub.index]
        for i in range(len(pv) - 1):
            total += 1
            if pv[i] < pv[i + 1]:
                violations += 1
    m["mono_viol_rate"] = violations / total if total > 0 else 0.0

    return m


def apply_monotonic(df: pd.DataFrame, pred_col: str) -> pd.Series:
    result = df[pred_col].copy()
    for _, grp in df.groupby("event_id"):
        idx    = grp.sort_values("distance").index
        values = grp.loc[idx, pred_col].values
        mono   = np.minimum.accumulate(values)
        result.loc[idx] = mono
    return result


# ===========================================================================
# PLOTS
# ===========================================================================

def _loglog_scatter(ax, target_pgv, pred_pgv, label, colour):
    ax.scatter(target_pgv, pred_pgv, s=4, alpha=0.3, color=colour, rasterized=True)
    lim = max(target_pgv.max(), pred_pgv.max()) * 1.05
    ax.plot([0.05, lim], [0.05, lim], "r--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    rmse = float(np.sqrt(np.mean((pred_pgv - target_pgv) ** 2)))
    ax.set_title(f"{label}\nRMSE(PGV)={rmse:.3f}"); ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("Measured PGV"); ax.set_ylabel("Predicted PGV")


def save_all_plots(test_df: pd.DataFrame, metrics: Dict[str, Dict],
                   gate_metrics: Dict, delta_c_df: Optional[pd.DataFrame],
                   fi_gate: pd.DataFrame, fi_delta_c: pd.DataFrame, fi_local: pd.DataFrame,
                   p_high_test: np.ndarray,
                   output_dir: Path) -> None:
    colours = {"P3": "#1f77b4", "PXGBR-R2": "#ff7f0e",
               "EAC-C2": "#9467bd", "EAC-C2+R2": "#2ca02c",
               "EAC-C2+R2_mono": "#17becf"}

    # 1. Scatter comparison
    models_to_plot = [k for k in ["P3", "PXGBR-R2", "EAC-C2+R2"] if k in metrics]
    n = len(models_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    cols_map = {
        "P3":       "pred_pgv_p3",
        "PXGBR-R2": "pred_pgv_pxgbr" if "pred_pgv_pxgbr" in test_df.columns else None,
        "EAC-C2+R2":"pred_pgv_eac_r2",
        "EAC-C2+R2_mono": "pred_pgv_eac_r2_mono",
    }
    for ax, m_name in zip(axes[0], models_to_plot):
        col = cols_map.get(m_name)
        if col and col in test_df.columns:
            _loglog_scatter(ax, test_df["target_pgv"].values,
                            test_df[col].values, m_name, colours.get(m_name, "grey"))
    fig.suptitle("Measured vs Predicted — P3 / PXGBR-R2 / EAC-C2+R2")
    fig.tight_layout()
    fig.savefig(output_dir / "measured_vs_predicted_P3_vs_PXGBR_vs_EAC.png",
                dpi=120, bbox_inches="tight"); plt.close(fig)

    # 2. Per-sensor RMSE bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(SENSOR_ORDER)); w = 0.25
    for metric_key, ylabel, ax_i in [("rmse_pgv", "RMSE(PGV)", 0), ("rmse_log", "RMSE(log)", 1)]:
        for j, (m_name, colour) in enumerate(
            [(k, colours.get(k, "grey")) for k in metrics if k in colours]
        ):
            vals = [metrics[m_name]["per_sensor"].get(s, {}).get(metric_key, np.nan)
                    for s in SENSOR_ORDER]
            offset = (j - len(metrics) / 2 + 0.5) * w
            axes[ax_i].bar(x + offset, vals, w, label=m_name, color=colour, alpha=0.8)
        axes[ax_i].set_xticks(x); axes[ax_i].set_xticklabels(SENSOR_ORDER)
        axes[ax_i].set_ylabel(ylabel); axes[ax_i].legend(fontsize=6)
        axes[ax_i].grid(True, axis="y", alpha=0.4)
    fig.suptitle("Per-Sensor Error Comparison"); fig.tight_layout()
    fig.savefig(output_dir / "per_sensor_rmse_P3_vs_PXGBR_vs_EAC.png",
                dpi=120, bbox_inches="tight"); plt.close(fig)

    # 3. High-PGV profile examples (EAC-C2+R2 vs P3)
    if "pred_log_eac_r2" in test_df.columns:
        hi_evs = list(test_df.groupby("event_id")["target_pgv"].max()
                               .sort_values(ascending=False).head(6).index)
        fig, axes = plt.subplots(2, 3, figsize=(13, 8)); axes = axes.flatten()
        for ax_i, ev in enumerate(hi_evs[:6]):
            ev_df = test_df[test_df["event_id"] == ev].sort_values("distance")
            if ev_df.empty: continue
            d = ev_df["distance"].values; t = ev_df["target_pgv"].values
            axes[ax_i].semilogy(d, t, "ko-", ms=5, label="Measured")
            axes[ax_i].semilogy(d, ev_df["pred_pgv_p3"].values, "b^--", ms=4, lw=1, label="P3")
            if "pred_pgv_eac_r2" in ev_df.columns:
                axes[ax_i].semilogy(d, ev_df["pred_pgv_eac_r2"].values, "g^--", ms=4, lw=1.2, label="EAC-R2")
            axes[ax_i].set_title(f"Event {str(ev)[-12:-4]}", fontsize=8)
            axes[ax_i].set_xlabel("d (m)"); axes[ax_i].grid(True, which="both", alpha=0.3)
            if ax_i == 0: axes[ax_i].legend(fontsize=7)
        fig.suptitle("High-PGV Events — P3 vs EAC-C2+R2"); fig.tight_layout()
        fig.savefig(output_dir / "high_pgv_profiles_P3_vs_EAC.png", dpi=120); plt.close(fig)

    # 4. Failure profiles
    if "pred_log_eac_r2" in test_df.columns:
        fail_evs = list(
            test_df.assign(sq=(lambda d: (d["pred_pgv_eac_r2"] - d["target_pgv"]) ** 2))
                   .groupby("event_id")["sq"].mean()
                   .apply(np.sqrt).sort_values(ascending=False).head(6).index
        )
        fig, axes = plt.subplots(2, 3, figsize=(13, 8)); axes = axes.flatten()
        for ax_i, ev in enumerate(fail_evs[:6]):
            ev_df = test_df[test_df["event_id"] == ev].sort_values("distance")
            if ev_df.empty: continue
            axes[ax_i].semilogy(ev_df["distance"].values, ev_df["target_pgv"].values,
                                "ko-", ms=5, label="Measured")
            axes[ax_i].semilogy(ev_df["distance"].values, ev_df["pred_pgv_p3"].values,
                                "b^--", ms=4, lw=1, label="P3")
            if "pred_pgv_eac_r2" in ev_df.columns:
                axes[ax_i].semilogy(ev_df["distance"].values, ev_df["pred_pgv_eac_r2"].values,
                                    "r^--", ms=4, lw=1.2, label="EAC-R2")
            axes[ax_i].set_title(f"{str(ev)[-12:-4]}", fontsize=8)
            axes[ax_i].set_xlabel("d (m)"); axes[ax_i].grid(True, which="both", alpha=0.3)
            if ax_i == 0: axes[ax_i].legend(fontsize=7)
        fig.suptitle("Failure Cases — P3 vs EAC-C2+R2"); fig.tight_layout()
        fig.savefig(output_dir / "failure_profiles_P3_vs_EAC.png", dpi=120); plt.close(fig)

    # 5. delta_c target vs predicted
    if delta_c_df is not None and "delta_c_pred" in delta_c_df.columns:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(delta_c_df["delta_c_target"], delta_c_df["delta_c_pred"],
                   s=6, alpha=0.4, rasterized=True)
        lim = max(abs(delta_c_df["delta_c_target"]).max(),
                  abs(delta_c_df["delta_c_pred"]).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
        ax.set_xlabel("True Δc"); ax.set_ylabel("Predicted Δc")
        rmse_dc = float(np.sqrt(np.mean(
            (delta_c_df["delta_c_pred"] - delta_c_df["delta_c_target"]) ** 2
        )))
        ax.set_title(f"Stage 1: Δc correction\nRMSE={rmse_dc:.4f}")
        ax.grid(True, alpha=0.3); fig.tight_layout()
        fig.savefig(output_dir / "delta_c_target_vs_predicted.png", dpi=120); plt.close(fig)

    # 6. c_target vs c_hat before/after calibration
    if delta_c_df is not None and "c_hat_calibrated" in delta_c_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, (c_col, title) in zip(axes, [
            ("c_hat_p3",      "Before calibration: c_hat_p3"),
            ("c_hat_calibrated", "After calibration: c_hat_calibrated"),
        ]):
            if c_col not in delta_c_df.columns: continue
            ax.scatter(delta_c_df["c_target_event"],
                       delta_c_df[c_col], s=6, alpha=0.4, rasterized=True)
            lim = max(abs(delta_c_df["c_target_event"]).max(),
                      abs(delta_c_df[c_col]).max()) * 1.05
            ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
            corr = float(np.corrcoef(delta_c_df["c_target_event"],
                                     delta_c_df[c_col])[0, 1])
            ax.set_title(f"{title}\nCorr={corr:.3f}")
            ax.set_xlabel("c_target"); ax.set_ylabel(c_col)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "c_target_vs_c_hat_before_after_calibration.png",
                    dpi=120); plt.close(fig)

    # 7-8. Feature importance plots
    for fi_df, fname, title in [
        (fi_delta_c,  "feature_importance_delta_c_top30.png",         "Stage 1 Δc regressor"),
        (fi_local,    "feature_importance_local_residual_top30.png",   "Stage 2 local residual"),
        (fi_gate,     "feature_importance_gate_top30.png",             "Stage 0 high-PGV gate"),
    ]:
        if fi_df is None or fi_df.empty: continue
        top = fi_df.head(30)
        fig, ax = plt.subplots(figsize=(8, 9))
        ax.barh(range(len(top)), top["importance"].values[::-1], alpha=0.8)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"].values[::-1], fontsize=7)
        ax.set_xlabel("Importance"); ax.set_title(f"{title} — Top 30")
        ax.grid(True, axis="x", alpha=0.4); fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=120, bbox_inches="tight"); plt.close(fig)

    # 9. High-PGV gate ROC
    if gate_metrics and "p_high_test" in gate_metrics and "y_test" in gate_metrics:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(gate_metrics["y_test"], gate_metrics["p_high_test"])
        auc = gate_metrics.get("test_auc", 0.0)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("Stage 0: High-PGV Gate ROC")
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        fig.savefig(output_dir / "high_pgv_gate_roc.png", dpi=120); plt.close(fig)

    print(f"  Plots saved to: {output_dir}")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="EAC-PXGB: Event-Amplitude Calibrated XGBoost")
    parser.add_argument("--p3_dir",    type=str, default=None)
    parser.add_argument("--output_tag", type=str, default="eac_pxgbr_linec_v001")
    args = parser.parse_args()

    print("=" * 80)
    print("EAC-PXGB — Event-Amplitude Calibrated Physics-Boosted XGBoost")
    print("=" * 80)

    # ── Discover inputs ──────────────────────────────────────────────────────
    p3_dir     = find_p3_dir(args.p3_dir)
    parquet_v2 = find_parquet_v2()
    pxgbr_dir  = find_pxgbr_dir()
    print(f"P3 savepred:   {p3_dir.name}")
    print(f"Parquet v2:    {parquet_v2}")
    print(f"PXGBR (ref):   {pxgbr_dir.name if pxgbr_dir else 'not found'}")

    # ── Load data ────────────────────────────────────────────────────────────
    p3_df  = load_p3(p3_dir)
    fo_df  = load_fo(parquet_v2)

    # ── Build event table ────────────────────────────────────────────────────
    ev_df  = build_event_table(p3_df, fo_df)

    ev_train = ev_df[ev_df["split"] == "train"].copy()
    ev_val   = ev_df[ev_df["split"] == "val"].copy()
    ev_test  = ev_df[ev_df["split"] == "test"].copy()

    ev_feat_cols = event_feature_cols(ev_df)
    print(f"\nEvent-level features: {len(ev_feat_cols)}")

    # Fill NaN
    for df_ in [ev_train, ev_val, ev_test]:
        for col in ev_feat_cols:
            if col in df_.columns:
                df_[col] = df_[col].fillna(0.0)

    Xe_tr = ev_train[ev_feat_cols].values.astype(np.float32)
    Xe_va = ev_val[ev_feat_cols].values.astype(np.float32)
    Xe_te = ev_test[ev_feat_cols].values.astype(np.float32)

    # ── Stage 0: High-PGV classifier ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 0: High-PGV Event Classifier (target > 8 mm/s)")
    print("=" * 60)

    gate_params = {**BASE_PARAMS, "max_depth": 3, "reg_lambda": 5,
                   "objective": "binary:logistic"}
    gate_params.pop("min_child_weight", None)
    gate_params["min_child_weight"] = 3

    gate_model = _xgb_classifier(gate_params)
    y_gate_tr  = ev_train["high_pgv_event"].values
    y_gate_va  = ev_val["high_pgv_event"].values
    y_gate_te  = ev_test["high_pgv_event"].values
    gate_model.fit(Xe_tr, y_gate_tr, eval_set=[(Xe_va, y_gate_va)], verbose=200)

    p_high_tr = gate_model.predict_proba(Xe_tr)[:, 1]
    p_high_va = gate_model.predict_proba(Xe_va)[:, 1]
    p_high_te = gate_model.predict_proba(Xe_te)[:, 1]

    gate_auc_val  = roc_auc_score(y_gate_va, p_high_va)
    gate_auc_test = roc_auc_score(y_gate_te, p_high_te)
    ap_test       = average_precision_score(y_gate_te, p_high_te)
    print(f"  Val  AUC: {gate_auc_val:.3f}")
    print(f"  Test AUC: {gate_auc_test:.3f}  |  AP: {ap_test:.3f}")

    # Attach p_high_event to event tables
    ev_train["p_high_event"] = p_high_tr
    ev_val["p_high_event"]   = p_high_va
    ev_test["p_high_event"]  = p_high_te

    gate_fi = pd.DataFrame({
        "feature":    ev_feat_cols,
        "importance": gate_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    gate_metrics_dict = {
        "val_auc": gate_auc_val, "test_auc": gate_auc_test, "test_ap": ap_test,
        "p_high_test": p_high_te.tolist(), "y_test": y_gate_te.tolist(),
    }

    # ── Stage 1: Event-level Δc correction ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1: Event-Level Δc Correction")
    print("=" * 60)

    # Feature columns for Stage 1 include p_high_event
    ev_feat_cols_s1 = event_feature_cols(ev_df) + ["p_high_event"]
    ev_feat_cols_s1 = [c for c in ev_feat_cols_s1 if c in ev_train.columns]

    for df_ in [ev_train, ev_val, ev_test]:
        for col in ev_feat_cols_s1:
            if col in df_.columns:
                df_[col] = df_[col].fillna(0.0)

    Xe1_tr = ev_train[ev_feat_cols_s1].values.astype(np.float32)
    Xe1_va = ev_val[ev_feat_cols_s1].values.astype(np.float32)
    Xe1_te = ev_test[ev_feat_cols_s1].values.astype(np.float32)

    y_dc_tr = ev_train["delta_c_target"].values.astype(np.float32)
    y_dc_va = ev_val["delta_c_target"].values.astype(np.float32)
    y_dc_te = ev_test["delta_c_target"].values.astype(np.float32)

    def _event_weights(ev: pd.DataFrame, variant: str) -> np.ndarray:
        w = np.ones(len(ev), dtype=np.float32)
        if variant == "C2":
            w += 2.0 * ev["high_pgv_event"].values.astype(np.float32)
            w += 2.0 * ev["high_mp4_event"].values.astype(np.float32)
        return w

    dc_params = {**BASE_PARAMS, "max_depth": 3, "reg_lambda": 5}

    dc_models  = {}
    dc_preds   = {}  # {split: {variant: array}}

    for variant, w_fn in [("C1", lambda e: np.ones(len(e))),
                          ("C2", lambda e: _event_weights(e, "C2"))]:
        print(f"\n  EAC-{variant}:")
        model = _xgb_regressor(dc_params)
        model.fit(Xe1_tr, y_dc_tr, sample_weight=w_fn(ev_train),
                  eval_set=[(Xe1_va, y_dc_va)], verbose=200)
        best_round = model.best_iteration
        rmse_dc_val = float(model.best_score)
        print(f"    best_round={best_round}  val_RMSE(Δc)={rmse_dc_val:.4f}")
        dc_models[variant]  = model
        dc_preds[variant]   = {
            "train": model.predict(Xe1_tr).astype(np.float32),
            "val":   model.predict(Xe1_va).astype(np.float32),
            "test":  model.predict(Xe1_te).astype(np.float32),
        }

    dc_fi = {
        v: pd.DataFrame({
            "feature":    ev_feat_cols_s1,
            "importance": dc_models[v].feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        for v in dc_models
    }

    # Use C2 as default for Stage 2
    best_dc_variant = "C2"
    for split, ev_part in [("train", ev_train), ("val", ev_val), ("test", ev_test)]:
        ev_part["delta_c_pred"] = dc_preds[best_dc_variant][split]
        ev_part["c_hat_calibrated"] = ev_part["c_hat_p3"] + ev_part["delta_c_pred"]

    # ── Apply Stage 1 correction to row-level data ───────────────────────────
    print("\nApplying Stage 1 correction to row-level predictions...")

    # Lookup delta_c_pred per event_id
    def _apply_dc(df_part: pd.DataFrame, ev_part: pd.DataFrame) -> pd.DataFrame:
        dc_map = dict(zip(ev_part["event_id"], ev_part["delta_c_pred"]))
        df_part = df_part.copy()
        df_part["delta_c_hat"]   = df_part["event_id"].map(dc_map).fillna(0.0)
        df_part["pred_log_eac_c"] = df_part["pred_log_p3"] + df_part["delta_c_hat"]
        df_part["pred_pgv_eac_c"] = np.exp(df_part["pred_log_eac_c"])
        return df_part

    p3_splits = {}
    for split_name, ev_part in [("train", ev_train), ("val", ev_val), ("test", ev_test)]:
        part = p3_df[p3_df["split"] == split_name].copy()
        p3_splits[split_name] = _apply_dc(part, ev_part)

    # ── Build row-level table for Stage 2 ────────────────────────────────────
    row_df = build_row_table(
        pd.concat(list(p3_splits.values()), ignore_index=True),
        fo_df, pd.concat([ev_train, ev_val, ev_test], ignore_index=True)
    )

    # Attach delta_c and eac_c to row table
    dc_row_map = {}
    for ev_part in [ev_train, ev_val, ev_test]:
        dc_row_map.update(dict(zip(ev_part["event_id"], ev_part["delta_c_pred"])))
    row_df["delta_c_hat"]    = row_df["event_id"].map(dc_row_map).fillna(0.0)
    row_df["pred_log_eac_c"] = row_df["pred_log_p3"] + row_df["delta_c_hat"]
    row_df["pred_pgv_eac_c"] = np.exp(row_df["pred_log_eac_c"])
    row_df["local_residual"] = row_df["target_log"] - row_df["pred_log_eac_c"]

    row_train = row_df[row_df["split"] == "train"].copy()
    row_val   = row_df[row_df["split"] == "val"].copy()
    row_test  = row_df[row_df["split"] == "test"].copy()

    # ── Stage 2: Local residual correction ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2: Row-Level Local Residual Correction")
    print("=" * 60)

    # Determine row feature columns (include eac_c features, exclude leakage)
    extra_for_stage2 = ["pred_log_eac_c", "pred_pgv_eac_c", "delta_c_hat",
                        "c_hat_p3", "epsilon_p3", "n_used",
                        "distance", "log_distance", "sensor_code", "track",
                        "p_high_event", "epsilon_p3_mean", "epsilon_p3_std",
                        "epsilon_p3_max", "epsilon_p3_min",
                        "train_speed_kmh", "train_speed_missing", "train_type_code"]
    row_feat_cols = row_feature_cols(row_train, extra_for_stage2)

    # Add pred_log_p3 explicitly if not already included
    if "pred_log_p3" not in row_feat_cols and "pred_log_p3" in row_train.columns:
        row_feat_cols = ["pred_log_p3"] + row_feat_cols

    print(f"  Row feature columns: {len(row_feat_cols)}")

    for df_ in [row_train, row_val, row_test]:
        for col in row_feat_cols:
            if col in df_.columns:
                df_[col] = df_[col].fillna(0.0)
            else:
                df_[col] = 0.0

    Xr_tr = row_train[row_feat_cols].values.astype(np.float32)
    Xr_va = row_val[row_feat_cols].values.astype(np.float32)
    Xr_te = row_test[row_feat_cols].values.astype(np.float32)

    y_lr_tr = row_train["local_residual"].values.astype(np.float32)
    y_lr_va = row_val["local_residual"].values.astype(np.float32)

    def _row_weights(df_: pd.DataFrame, variant: str) -> np.ndarray:
        w = np.ones(len(df_), dtype=np.float32)
        if variant == "R2":
            w += 1.5 * (df_["sensor"].values == "MP4").astype(np.float32)
            w += 1.0 * (df_["target_pgv"].values > 4.0).astype(np.float32)
            w += 1.0 * (df_["target_pgv"].values > 8.0).astype(np.float32)
        return w

    lr_params  = {**BASE_PARAMS, "max_depth": 4, "reg_lambda": 5}
    lr_models  = {}
    lr_preds   = {}

    for variant, w_fn in [("R1", lambda d: np.ones(len(d))),
                          ("R2", lambda d: _row_weights(d, "R2"))]:
        print(f"\n  EAC-C2+{variant}:")
        model = _xgb_regressor(lr_params)
        model.fit(Xr_tr, y_lr_tr, sample_weight=w_fn(row_train),
                  eval_set=[(Xr_va, y_lr_va)], verbose=200)
        print(f"    best_round={model.best_iteration}  val_RMSE={model.best_score:.4f}")
        lr_models[variant] = model
        lr_preds[variant]  = {
            "train": model.predict(Xr_tr),
            "val":   model.predict(Xr_va),
            "test":  model.predict(Xr_te),
        }

    local_fi = {
        v: pd.DataFrame({
            "feature":    row_feat_cols,
            "importance": lr_models[v].feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        for v in lr_models
    }

    # ── Final predictions ────────────────────────────────────────────────────
    test_df = row_test.copy()
    test_df["pred_log_eac_c"]   = test_df["pred_log_eac_c"].astype(np.float32)
    test_df["pred_pgv_eac_c"]   = np.exp(test_df["pred_log_eac_c"])

    for v in lr_preds:
        test_df[f"pred_log_eac_{v}"]        = test_df["pred_log_eac_c"] + lr_preds[v]["test"]
        test_df[f"pred_pgv_eac_{v}"]        = np.exp(test_df[f"pred_log_eac_{v}"])
        test_df[f"pred_log_eac_{v}_mono"]   = apply_monotonic(test_df, f"pred_log_eac_{v}")
        test_df[f"pred_pgv_eac_{v}_mono"]   = np.exp(test_df[f"pred_log_eac_{v}_mono"])

    # Rename for convenience
    test_df["pred_pgv_eac_r2"]      = test_df["pred_pgv_eac_R2"]
    test_df["pred_pgv_eac_r2_mono"] = test_df["pred_pgv_eac_R2_mono"]
    test_df["pred_log_eac_r2"]      = test_df["pred_log_eac_R2"]

    # Optionally load PXGBR predictions for comparison
    if pxgbr_dir:
        try:
            px_preds = pd.read_parquet(pxgbr_dir / "predictions_test.parquet")
            px_map_pgv = dict(zip(
                px_preds["event_id"] + "|" + px_preds["sensor"],
                px_preds["pred_pgv_R2"]
            ))
            px_map_log = dict(zip(
                px_preds["event_id"] + "|" + px_preds["sensor"],
                px_preds["pred_log_R2"]
            ))
            key = test_df["event_id"] + "|" + test_df["sensor"]
            test_df["pred_pgv_pxgbr"] = key.map(px_map_pgv)
            test_df["pred_log_pxgbr"] = key.map(px_map_log)
        except Exception as e:
            print(f"  [WARN] Could not load PXGBR predictions: {e}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TEST METRICS")
    print("=" * 80)

    tgt_log    = test_df["target_log"].values
    sensors    = test_df["sensor"].values
    event_ids  = test_df["event_id"].values

    all_metrics: Dict[str, Dict] = {}
    all_metrics["P3"] = compute_metrics(
        test_df["pred_log_p3"].values, tgt_log, sensors, event_ids, "P3"
    )
    if "pred_log_pxgbr" in test_df.columns:
        mask = test_df["pred_log_pxgbr"].notna()
        all_metrics["PXGBR-R2"] = compute_metrics(
            test_df.loc[mask, "pred_log_pxgbr"].values,
            tgt_log[mask.values], sensors[mask.values], event_ids[mask.values], "PXGBR-R2"
        )
    all_metrics["EAC-C2"]      = compute_metrics(
        test_df["pred_log_eac_c"].values, tgt_log, sensors, event_ids, "EAC-C2"
    )
    all_metrics["EAC-C2+R1"]   = compute_metrics(
        test_df["pred_log_eac_R1"].values, tgt_log, sensors, event_ids, "EAC-C2+R1"
    )
    all_metrics["EAC-C2+R2"]   = compute_metrics(
        test_df["pred_log_eac_R2"].values, tgt_log, sensors, event_ids, "EAC-C2+R2"
    )
    all_metrics["EAC-C2+R2_mono"] = compute_metrics(
        test_df["pred_log_eac_R2_mono"].values, tgt_log, sensors, event_ids, "EAC-C2+R2_mono"
    )

    print(f"\n{'Model':<22} {'RMSE(log)':>10} {'RMSE(PGV)':>10} {'R²(log)':>8} {'MP4 RMSE':>10}")
    print("-" * 65)
    for name, m in all_metrics.items():
        mp4_rmse = m["per_sensor"].get("MP4", {}).get("rmse_pgv", float("nan"))
        print(f"{name:<22} {m['rmse_log']:>10.4f} {m['rmse_pgv']:>10.4f} "
              f"{m['r2_log']:>8.4f} {mp4_rmse:>10.4f}")

    print("\nEAC-C2+R2 per-sensor:")
    for s, sm in all_metrics["EAC-C2+R2"]["per_sensor"].items():
        print(f"  {s:5s}: RMSE(PGV)={sm['rmse_pgv']:.3f}  bias(PGV)={sm['bias_pgv']:+.3f}")

    # ── Delta_c diagnostics ───────────────────────────────────────────────────
    ev_test["delta_c_pred"]  = dc_preds["C2"]["test"]
    ev_test["c_hat_calibrated"] = ev_test["c_hat_p3"] + ev_test["delta_c_pred"]
    dc_rmse = float(np.sqrt(np.mean(
        (ev_test["delta_c_pred"] - ev_test["delta_c_target"]) ** 2
    )))
    r2_before = float(np.corrcoef(ev_test["c_target_event"], ev_test["c_hat_p3"])[0, 1]) ** 2
    r2_after  = float(np.corrcoef(ev_test["c_target_event"], ev_test["c_hat_calibrated"])[0, 1]) ** 2
    print(f"\nStage 1 Δc RMSE (test): {dc_rmse:.4f}")
    print(f"c_hat R² before calibration: {r2_before:.4f}")
    print(f"c_hat R² after  calibration: {r2_after:.4f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = MODELS_ROOT / f"{args.output_tag}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    log_dir = Path("outputs/eac_pxgbr_linec_v1")
    log_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    metrics_out = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("per_sensor",)}
        for k, v in all_metrics.items()
    }
    metrics_out["per_sensor"] = {k: v.get("per_sensor", {}) for k, v in all_metrics.items()}
    metrics_out["gate"]       = {
        "val_auc": gate_metrics_dict["val_auc"],
        "test_auc": gate_metrics_dict["test_auc"],
        "test_ap":  gate_metrics_dict["test_ap"],
    }
    metrics_out["delta_c"]    = {"rmse": dc_rmse, "r2_before": r2_before, "r2_after": r2_after}
    metrics_out["p3_dir"]     = str(p3_dir)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=float)

    # metrics_table.csv
    rows = []
    for name, m in all_metrics.items():
        row = {"model": name}
        for k in ["rmse_log", "rmse_pgv", "mae_pgv", "r2_log", "bias_pgv", "mono_viol_rate"]:
            row[k] = m.get(k, float("nan"))
        for s in SENSOR_ORDER:
            sm = m["per_sensor"].get(s, {})
            row[f"{s}_rmse_pgv"] = sm.get("rmse_pgv", float("nan"))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "metrics_table.csv", index=False)

    # Predictions
    save_cols = (
        ["event_id", "sensor", "track", "distance", "split",
         "target_log", "target_pgv",
         "pred_log_p3", "pred_pgv_p3",
         "pred_log_eac_c", "pred_pgv_eac_c",
         "pred_log_eac_R1", "pred_pgv_eac_R1",
         "pred_log_eac_R2", "pred_pgv_eac_R2",
         "pred_log_eac_R2_mono", "pred_pgv_eac_R2_mono"]
        + (["pred_log_pxgbr", "pred_pgv_pxgbr"] if "pred_pgv_pxgbr" in test_df.columns else [])
    )
    test_df[[c for c in save_cols if c in test_df.columns]].to_parquet(
        out / "predictions_test.parquet", index=False
    )

    # Event-level predictions
    ev_test_save = ev_test[["event_id", "split", "c_hat_p3", "c_target_event",
                             "delta_c_target", "delta_c_pred", "c_hat_calibrated",
                             "high_pgv_event", "high_mp4_event",
                             "p_high_event"]].copy()
    ev_test_save.to_parquet(out / "event_predictions_test.parquet", index=False)

    # Feature importances
    for v in dc_models:
        dc_fi[v].to_csv(out / f"feature_importance_delta_c_{v}.csv", index=False)
    gate_fi.to_csv(out / "feature_importance_gate.csv", index=False)
    for v in lr_models:
        local_fi[v].to_csv(out / f"feature_importance_local_residual_{v}.csv", index=False)

    # Feature column lists
    (out / "event_feature_columns.txt").write_text(
        "\n".join(ev_feat_cols_s1), encoding="utf-8"
    )
    (out / "row_feature_columns.txt").write_text(
        "\n".join(row_feat_cols), encoding="utf-8"
    )

    # Save event tables
    ev_train[["event_id", "split"] + ev_feat_cols_s1 +
             ["c_target_event", "delta_c_target", "high_pgv_event",
              "high_mp4_event", "max_pgv"]
             ].to_parquet(out / "event_train_table.parquet", index=False)
    ev_val[["event_id", "split"] + ev_feat_cols_s1 +
           ["c_target_event", "delta_c_target", "high_pgv_event",
            "high_mp4_event", "max_pgv"]
           ].to_parquet(out / "event_val_table.parquet", index=False)
    ev_test_save.to_parquet(out / "event_test_table.parquet", index=False)

    # High-PGV gate predictions
    gate_out = pd.concat([
        ev_train[["event_id", "split"]].assign(
            p_high_event=p_high_tr, y_true=y_gate_tr),
        ev_val[["event_id", "split"]].assign(
            p_high_event=p_high_va, y_true=y_gate_va),
        ev_test[["event_id", "split"]].assign(
            p_high_event=p_high_te, y_true=y_gate_te),
    ], ignore_index=True)
    gate_out.to_parquet(out / "high_pgv_gate_predictions.parquet", index=False)
    with open(out / "high_pgv_gate_metrics.json", "w") as f:
        json.dump({k: v for k, v in gate_metrics_dict.items()
                   if k not in ("p_high_test", "y_test")}, f, indent=2, default=float)

    # Δc predictions
    dc_save = pd.concat([
        ev_train[["event_id", "split", "c_hat_p3", "c_target_event",
                  "delta_c_target"]].assign(delta_c_pred=dc_preds["C2"]["train"]),
        ev_val[["event_id", "split", "c_hat_p3", "c_target_event",
                "delta_c_target"]].assign(delta_c_pred=dc_preds["C2"]["val"]),
        ev_test[["event_id", "split", "c_hat_p3", "c_target_event",
                 "delta_c_target", "c_hat_calibrated"]].assign(
                    delta_c_pred=dc_preds["C2"]["test"]),
    ], ignore_index=True)
    dc_save.to_parquet(out / "delta_c_predictions.parquet", index=False)

    # Save XGB models
    gate_model.save_model(str(out / "model_gate.ubj"))
    for v, m in dc_models.items():
        m.save_model(str(out / f"model_delta_c_{v}.ubj"))
    for v, m in lr_models.items():
        m.save_model(str(out / f"model_local_residual_{v}.ubj"))

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        save_all_plots(
            test_df=test_df,
            metrics=all_metrics,
            gate_metrics=gate_metrics_dict,
            delta_c_df=ev_test,
            fi_gate=gate_fi,
            fi_delta_c=dc_fi.get("C2", pd.DataFrame()),
            fi_local=local_fi.get("R2", pd.DataFrame()),
            p_high_test=p_high_te,
            output_dir=out,
        )
    except Exception as e:
        print(f"[WARN] Some plots failed: {e}")

    print(f"\nOutput saved to: {out}")
    print(f"\nFinal comparison:")
    print(f"  P3 baseline:      RMSE(PGV)={all_metrics['P3']['rmse_pgv']:.4f}")
    if "PXGBR-R2" in all_metrics:
        print(f"  PXGBR-R2:         RMSE(PGV)={all_metrics['PXGBR-R2']['rmse_pgv']:.4f}")
    print(f"  EAC-C2:           RMSE(PGV)={all_metrics['EAC-C2']['rmse_pgv']:.4f}")
    print(f"  EAC-C2+R2:        RMSE(PGV)={all_metrics['EAC-C2+R2']['rmse_pgv']:.4f}")
    print(f"  EAC-C2+R2_mono:   RMSE(PGV)={all_metrics['EAC-C2+R2_mono']['rmse_pgv']:.4f}")
    delta = all_metrics["P3"]["rmse_pgv"] - all_metrics["EAC-C2+R2"]["rmse_pgv"]
    print(f"\n  EAC vs P3 improvement: {delta:+.4f} mm/s "
          f"({delta / all_metrics['P3']['rmse_pgv'] * 100:+.1f}%)")


if __name__ == "__main__":
    main()
