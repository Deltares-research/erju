"""
train_pxgbr_linec_v1.py — Physics-Boosted Residual XGBoost (PXGB-R)

Architecture:
    pred_log_final = pred_log_p3 + residual_xgb
    residual_xgb trained on: residual_log = target_log - pred_log_p3

Physics baseline retained, XGBoost corrects remaining errors.
Expected to recover the high-PGV underprediction of P3 (especially MP4).

Two variants:
    PXGBR-R1 — uniform sample weights
    PXGBR-R2 — MP4 + high-PGV up-weighted

Inputs:
    P3 all_predictions.parquet   (from train_cnn_curveprior_linec_v1_savepred.py)
    parquet_v002 dataset.parquet (FO engineered features)

Usage:
    python train_pxgbr_linec_v1.py
    python train_pxgbr_linec_v1.py --p3_dir /path/to/p3_savepred_dir
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
from sklearn.isotonic import IsotonicRegression

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))


def _get_data_root() -> Path:
    return Path(r"P:\11210978-erju-ai") if os.name == "nt" else Path("/p/11210978-erju-ai")


MODELS_ROOT  = _get_data_root() / "holten_models"
PARQUET_ROOT = _get_data_root() / "holten_parquet"

SENSOR_ORDER = ["MP4", "MP8", "MP10", "MP1", "MP2"]
LINE_C_SENSORS = SENSOR_ORDER

# Columns to explicitly exclude from feature matrix (leakage / targets)
EXCLUDE_COLS = {
    "event_id", "site_id", "sensor_id",
    "target_pgv_z_mms",        # the true target
    "train_type",              # string, use train_type_code instead
    "acc_side_of_track",       # constant for line-C
    "split",                   # partition label
    # Physics prediction columns (included explicitly as features)
    # "pred_log_p3", "pred_pgv_p3", "c_hat_p3", "epsilon_p3", "n_used" — these ARE features
    # P3-derived targets we don't want as features:
    "c_target",                # diagnostic only, derived from true label
}


# ===========================================================================
# DATA DISCOVERY
# ===========================================================================

def find_p3_dir(p3_dir_override: Optional[str] = None) -> Path:
    """Find the most recent P3 savepred output directory."""
    if p3_dir_override:
        return Path(p3_dir_override)
    hits = sorted(MODELS_ROOT.glob("cnn_curveprior_p3_savepred_linec_v001_vP3_*"),
                  key=lambda p: p.name)
    if not hits:
        raise FileNotFoundError(
            "No cnn_curveprior_p3_savepred_linec_v001_vP3_* directories found.\n"
            "Run: python train_cnn_curveprior_linec_v1_savepred.py --variant P3 "
            "--n_mode fit_corrected --lambda_residual 0.05 --epochs 100"
        )
    return hits[-1]


def find_parquet_v2() -> Path:
    hits = sorted(PARQUET_ROOT.glob("parquet_v002_*"), key=lambda p: p.name)
    if not hits:
        raise FileNotFoundError("No parquet_v002_* builds found")
    return hits[-1] / "dataset.parquet"


# ===========================================================================
# DATA LOADING AND FEATURE ENGINEERING
# ===========================================================================

def load_p3_predictions(p3_dir: Path) -> pd.DataFrame:
    """Load all-splits P3 predictions."""
    p = p3_dir / "all_predictions.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"all_predictions.parquet not found in {p3_dir}\n"
            "Re-run the savepred script to generate it."
        )
    df = pd.read_parquet(p)
    print(f"P3 predictions: {len(df):,} rows, {df['event_id'].nunique():,} events")
    print(f"Splits: {df['split'].value_counts().to_dict()}")
    return df


def load_fo_features(parquet_path: Path) -> pd.DataFrame:
    """Load FO engineered features from parquet v2, filtered to line-C sensors."""
    df = pd.read_parquet(parquet_path)
    df = df[df["sensor_id"].isin(LINE_C_SENSORS)].copy()
    df = df.dropna(subset=["target_pgv_z_mms", "track_number"])
    df = df[(df["target_pgv_z_mms"] > 0) & (df["track_number"].isin([1, 2]))]
    df["event_id"] = df["event_id"].astype(str)
    df["sensor"] = df["sensor_id"]  # alias for join
    print(f"FO features: {len(df):,} rows, {df.columns.tolist()[:5]}...")
    return df


def build_feature_matrix(p3_df: pd.DataFrame, fo_df: pd.DataFrame) -> pd.DataFrame:
    """Join P3 predictions with FO features and engineer the full feature matrix."""

    # Merge on event_id + sensor
    merged = p3_df.merge(fo_df, on=["event_id", "sensor"], how="inner",
                         suffixes=("_p3", "_fo"))
    print(f"After join: {len(merged):,} rows ({len(p3_df) - len(merged)} dropped)")

    # ── Target ──────────────────────────────────────────────────────────────
    merged["residual_log"] = merged["target_log"] - merged["pred_log_p3"]

    # ── Physics features from P3 ────────────────────────────────────────────
    merged["log_distance"] = np.log(merged["distance"] / 10.0)

    # ── Sensor encoding ─────────────────────────────────────────────────────
    sensor_map = {s: i for i, s in enumerate(SENSOR_ORDER)}
    merged["sensor_code"] = merged["sensor"].map(sensor_map).astype(float)

    # ── Metadata ─────────────────────────────────────────────────────────────
    # train_speed_kmh may be NaN; create missing indicator
    if "train_speed_kmh" in merged.columns:
        merged["train_speed_missing"] = merged["train_speed_kmh"].isna().astype(float)
        merged["train_speed_kmh"] = merged["train_speed_kmh"].fillna(0.0)
    else:
        merged["train_speed_missing"] = 0.0
        merged["train_speed_kmh"] = 0.0

    return merged


def identify_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of feature columns for XGBoost.

    Includes:
    - Physics features from P3
    - Sensor geometry features
    - Metadata features
    - All FO spectral / time-domain features from parquet v2

    Excludes target and identifier columns.
    """
    # Explicit physics + geometry + metadata features (always included)
    explicit = [
        "pred_log_p3", "pred_pgv_p3", "c_hat_p3", "epsilon_p3", "n_used",
        "distance", "log_distance", "track", "sensor_code",
        "train_speed_kmh", "train_speed_missing",
    ]
    # train_type_code if present
    if "train_type_code" in df.columns:
        explicit.append("train_type_code")
    if "track_number" in df.columns:
        explicit.append("track_number")

    # All FO feature columns: heuristic — numeric columns not in explicit or exclude lists
    non_fo = set(explicit) | EXCLUDE_COLS | {
        "event_id", "sensor", "split", "residual_log",
        "target_log", "target_pgv",
        # columns added during join
        "sensor_id", "sensor_y",  "sensor_x",
        "acc_distance_to_track_m", "acc_distance_to_track_2_m",
        "effective_distance_to_active_track_m",
        "site_id", "train_type",
    }

    fo_cols = [
        c for c in df.columns
        if c not in non_fo
        and pd.api.types.is_numeric_dtype(df[c])
        and not c.startswith("target_")
        and not c.startswith("pred_")     # only p3 predictions as explicit features
        and "pgv" not in c.lower()
    ]

    all_features = explicit + [c for c in fo_cols if c not in explicit]
    # Keep only columns that exist in df
    all_features = [c for c in all_features if c in df.columns]

    return all_features


# ===========================================================================
# TRAINING
# ===========================================================================

def compute_sample_weights(df: pd.DataFrame, variant: str) -> np.ndarray:
    """Compute per-row sample weights."""
    w = np.ones(len(df), dtype=np.float32)
    if variant == "R2":
        w += 1.5 * (df["sensor"] == "MP4").values.astype(np.float32)
        w += 1.0 * (df["target_pgv"] > 4.0).values.astype(np.float32)
        w += 1.0 * (df["target_pgv"] > 8.0).values.astype(np.float32)
    return w


def train_xgb_residual(
    X_train: pd.DataFrame, y_train: np.ndarray, w_train: np.ndarray,
    X_val:   pd.DataFrame, y_val:   np.ndarray,
    params: Dict, n_estimators: int = 2000, early_stopping_rounds: int = 50,
    verbose_eval: int = 100,
) -> Tuple[xgb.XGBRegressor, int]:
    """Train XGBoost regressor on log-space residuals with early stopping."""
    model = xgb.XGBRegressor(
        **params,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="rmse",
        device="cuda" if xgb.__version__ >= "2.0" else None,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        verbose=verbose_eval,
    )
    best_round = model.best_iteration
    print(f"  Best round: {best_round}  |  val RMSE(residual): {model.best_score:.4f}")
    return model, best_round


# ===========================================================================
# METRICS
# ===========================================================================

def compute_metrics(
    pred_log_final: np.ndarray,
    target_log:     np.ndarray,
    sensors:        np.ndarray,
    label:          str,
) -> Dict:
    """Comprehensive metric computation."""
    pred_pgv   = np.exp(pred_log_final)
    target_pgv = np.exp(target_log)

    def _rmse(a, b):  return float(np.sqrt(np.mean((a - b) ** 2)))
    def _mae(a, b):   return float(np.mean(np.abs(a - b)))
    def _r2(a, b):
        ss = np.sum((b - a) ** 2); st = np.sum((b - b.mean()) ** 2)
        return float(1.0 - ss / st) if st > 0 else 0.0
    def _bias(a, b):  return float(np.mean(a - b))

    m: Dict = {
        "model":    label,
        "rmse_log": _rmse(pred_log_final, target_log),
        "rmse_pgv": _rmse(pred_pgv,  target_pgv),
        "mae_pgv":  _mae(pred_pgv,   target_pgv),
        "r2_log":   _r2(pred_log_final, target_log),
        "bias_pgv": _bias(pred_pgv,  target_pgv),
    }

    # Per-sensor
    m["per_sensor"] = {}
    for s in SENSOR_ORDER:
        mask = sensors == s
        if mask.any():
            m["per_sensor"][s] = {
                "rmse_log": _rmse(pred_log_final[mask], target_log[mask]),
                "rmse_pgv": _rmse(pred_pgv[mask], target_pgv[mask]),
                "bias_pgv": _bias(pred_pgv[mask], target_pgv[mask]),
                "bias_log": _bias(pred_log_final[mask], target_log[mask]),
            }

    # High-PGV
    for thr in [4.0, 8.0]:
        mask = target_pgv > thr
        key  = f"pgv_gt_{int(thr)}"
        if mask.any():
            m[key] = {
                "n":        int(mask.sum()),
                "rmse_pgv": _rmse(pred_pgv[mask], target_pgv[mask]),
                "bias_pgv": _bias(pred_pgv[mask], target_pgv[mask]),
            }

    # MP4 high-PGV
    mp4_mask = sensors == "MP4"
    for thr in [4.0, 8.0]:
        hi_mask = (target_pgv > thr) & mp4_mask
        if hi_mask.any():
            m[f"mp4_pgv_gt_{int(thr)}"] = {
                "n":        int(hi_mask.sum()),
                "rmse_pgv": _rmse(pred_pgv[hi_mask], target_pgv[hi_mask]),
                "bias_pgv": _bias(pred_pgv[hi_mask], target_pgv[hi_mask]),
            }

    return m


def apply_monotonic(df: pd.DataFrame, pred_col: str) -> pd.Series:
    """Enforce decreasing log-PGV with distance per event (isotonic via cummin).

    For sensors sorted by distance (near → far), the prediction at each sensor
    is floored to the minimum of all predictions at closer sensors.
    """
    result = df[pred_col].copy()
    for event_id, grp in df.groupby("event_id"):
        grp_sorted = grp.sort_values("distance")
        mono = grp_sorted[pred_col].cummin().values  # decreasing log-PGV
        result.loc[grp_sorted.index] = mono
    return result


def monotonicity_violation_rate(df: pd.DataFrame, pred_col: str) -> float:
    """Fraction of event-adjacent sensor pairs where prediction is non-monotonic."""
    violations, total = 0, 0
    for _, grp in df.groupby("event_id"):
        grp_sorted = grp.sort_values("distance")[pred_col].values
        for i in range(len(grp_sorted) - 1):
            total += 1
            if grp_sorted[i] < grp_sorted[i + 1]:
                violations += 1
    return violations / total if total > 0 else 0.0


# ===========================================================================
# PLOTS
# ===========================================================================

def plot_loglog_comparison(dfs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Measured vs predicted scatter for P3 and PXGB-R2."""
    pairs = [(k, dfs[k]) for k in ["P3", "PXGBR-R2"] if k in dfs]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5), squeeze=False)
    for ax, (label, df) in zip(axes[0], pairs):
        col = "pred_pgv_p3" if label == "P3" else "pred_pgv_final"
        ax.scatter(df["target_pgv"], df[col], s=4, alpha=0.3, rasterized=True)
        lim = max(df["target_pgv"].max(), df[col].max()) * 1.05
        ax.plot([0.05, lim], [0.05, lim], "r--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        rmse = float(np.sqrt(np.mean((df[col] - df["target_pgv"]) ** 2)))
        ax.set_title(f"{label}\nRMSE(PGV)={rmse:.3f} mm/s")
        ax.set_xlabel("Measured PGV (mm/s)"); ax.set_ylabel("Predicted PGV (mm/s)")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Measured vs Predicted — P3 and PXGB-R2")
    fig.tight_layout()
    fig.savefig(output_dir / "measured_vs_predicted_p3_vs_pxgbr.png", dpi=120)
    plt.close(fig)


def plot_per_sensor_rmse(metrics_dict: Dict[str, Dict], output_dir: Path) -> None:
    """Side-by-side per-sensor RMSE(log) bar chart."""
    models = list(metrics_dict.keys())
    x = np.arange(len(SENSOR_ORDER))
    width = 0.8 / len(models)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax_i, metric_key in enumerate(["rmse_log", "rmse_pgv"]):
        for j, model in enumerate(models):
            vals = [metrics_dict[model]["per_sensor"].get(s, {}).get(metric_key, np.nan)
                    for s in SENSOR_ORDER]
            offset = (j - len(models) / 2 + 0.5) * width
            axes[ax_i].bar(x + offset, vals, width, label=model, alpha=0.8)
        axes[ax_i].set_xticks(x); axes[ax_i].set_xticklabels(SENSOR_ORDER)
        axes[ax_i].set_ylabel(f"RMSE ({metric_key.split('_')[1]})")
        axes[ax_i].set_title(f"Per-Sensor RMSE({metric_key.split('_')[1]})")
        axes[ax_i].legend(fontsize=7); axes[ax_i].grid(True, axis="y", alpha=0.4)
    fig.suptitle("P3 vs PXGB-R — Per-Sensor Error")
    fig.tight_layout()
    fig.savefig(output_dir / "per_sensor_rmse_p3_vs_pxgbr.png", dpi=120)
    plt.close(fig)


def plot_residual_vs_distance(dfs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Residual vs distance for P3 and PXGB-R2."""
    pairs = [(k, dfs[k]) for k in ["P3", "PXGBR-R2"] if k in dfs]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4), squeeze=False)
    for ax, (label, df) in zip(axes[0], pairs):
        col = "pred_pgv_p3" if label == "P3" else "pred_pgv_final"
        res = df[col] - df["target_pgv"]
        ax.scatter(df["distance"], res, s=4, alpha=0.3, rasterized=True)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Distance (m)"); ax.set_ylabel("Residual (mm/s)")
        ax.set_title(label); ax.grid(True, alpha=0.3)
    fig.suptitle("Residual vs Distance — P3 and PXGB-R2")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_vs_distance_p3_vs_pxgbr.png", dpi=120)
    plt.close(fig)


def plot_attenuation_profiles(test_df: pd.DataFrame, output_dir: Path,
                               tag: str = "representative") -> None:
    """Plot 2×3 attenuation profile grids comparing P3 vs PXGB-R2."""
    if "pred_pgv_final" not in test_df.columns:
        return

    # Select events
    if tag == "representative":
        ev_err = (
            test_df.assign(ae=lambda d: np.abs(d["pred_log_final"] - d["target_log"]))
                   .groupby("event_id")["ae"].median().sort_values()
        )
        ev_maxpgv = test_df.groupby("event_id")["target_pgv"].max()
        good = ev_maxpgv[ev_maxpgv >= 1.0].index
        sel_evs = list(ev_err[ev_err.index.isin(good)].head(6).index)
        title = "Representative Events — P3 vs PXGB-R2"
    elif tag == "high_pgv":
        sel_evs = list(test_df.groupby("event_id")["target_pgv"].max()
                               .sort_values(ascending=False).head(6).index)
        title = "High-PGV Events — P3 vs PXGB-R2"
    else:  # failures
        sel_evs = list(
            test_df.assign(sq=lambda d: (d["pred_pgv_final"] - d["target_pgv"]) ** 2)
                   .groupby("event_id")["sq"].mean()
                   .apply(np.sqrt).sort_values(ascending=False).head(6).index
        )
        title = "Failure Cases — P3 vs PXGB-R2"

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()
    for ax_i, ev in enumerate(sel_evs[:6]):
        ev_df = test_df[test_df["event_id"] == ev].sort_values("distance")
        if ev_df.empty:
            axes[ax_i].set_visible(False)
            continue
        d = ev_df["distance"].values
        t = ev_df["target_pgv"].values
        p3  = ev_df["pred_pgv_p3"].values
        pxgb = ev_df["pred_pgv_final"].values
        axes[ax_i].semilogy(d, t, "ko-", ms=5, lw=1.2, label="Measured")
        axes[ax_i].semilogy(d, p3, "b^--", ms=4, lw=1.0, alpha=0.7, label="P3")
        axes[ax_i].semilogy(d, pxgb, "g^--", ms=4, lw=1.2, label="PXGB-R2")
        axes[ax_i].set_title(f"{str(ev)[-12:-4]}\nTr{int(ev_df['track'].iloc[0])}", fontsize=8)
        axes[ax_i].set_xlabel("d (m)"); axes[ax_i].grid(True, which="both", alpha=0.3)
        if ax_i == 0:
            axes[ax_i].legend(fontsize=7, loc="upper right")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / f"attenuation_profiles_{tag}_p3_vs_pxgbr.png", dpi=120)
    plt.close(fig)


def plot_feature_importance(model: xgb.XGBRegressor, feature_names: List[str],
                             output_dir: Path, n_top: int = 30) -> None:
    imp = model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": imp})
    imp_df = imp_df.sort_values("importance", ascending=False)
    imp_df.to_csv(output_dir / "feature_importance.csv", index=False)

    top = imp_df.head(n_top)
    fig, ax = plt.subplots(figsize=(8, n_top * 0.3 + 1))
    ax.barh(range(len(top)), top["importance"].values[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values[::-1], fontsize=8)
    ax.set_xlabel("XGBoost feature importance (gain)")
    ax.set_title(f"Top {n_top} features — PXGB-R2")
    ax.grid(True, axis="x", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance_top30.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="PXGB-R: Physics-Boosted Residual XGBoost")
    parser.add_argument("--p3_dir", type=str, default=None,
                        help="Path to P3 savepred output directory (auto-discovers if not set)")
    parser.add_argument("--output_tag", type=str, default="pxgbr_linec_v001")
    args = parser.parse_args()

    print("=" * 80)
    print("PXGB-R — Physics-Boosted Residual XGBoost — Holten Line-C")
    print("=" * 80)

    # ── Discover inputs ──────────────────────────────────────────────────────
    p3_dir      = find_p3_dir(args.p3_dir)
    parquet_v2  = find_parquet_v2()
    print(f"P3 predictions: {p3_dir.name}")
    print(f"Parquet v2:     {parquet_v2}")

    # ── Load data ────────────────────────────────────────────────────────────
    p3_df = load_p3_predictions(p3_dir)
    fo_df = load_fo_features(parquet_v2)
    df    = build_feature_matrix(p3_df, fo_df)

    # ── Feature columns ───────────────────────────────────────────────────────
    feature_cols = identify_feature_columns(df)
    print(f"\nFeature matrix: {len(feature_cols)} features")
    print(f"  Physics:  {[c for c in feature_cols if c in ['pred_log_p3','c_hat_p3','epsilon_p3','distance','log_distance','n_used']]}")
    print(f"  Metadata: {[c for c in feature_cols if c in ['train_speed_kmh','train_speed_missing','train_type_code','track_number']]}")
    print(f"  FO:       {len([c for c in feature_cols if c not in ['pred_log_p3','pred_pgv_p3','c_hat_p3','epsilon_p3','n_used','distance','log_distance','track','sensor_code','train_speed_kmh','train_speed_missing','train_type_code','track_number','track']])} columns")

    # ── Splits ────────────────────────────────────────────────────────────────
    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()
    print(f"\nSplit sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Fill NaN in feature matrix (XGBoost handles NaN natively, but be explicit)
    for split in [train_df, val_df, test_df]:
        for col in feature_cols:
            if col in split.columns:
                split[col] = split[col].fillna(0.0)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["residual_log"].values.astype(np.float32)
    X_val   = val_df[feature_cols].values.astype(np.float32)
    y_val   = val_df["residual_log"].values.astype(np.float32)
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_test  = test_df["residual_log"].values.astype(np.float32)

    # ── Hyperparameters ───────────────────────────────────────────────────────
    # Single best config (from light grid knowledge); no expensive tuning here
    base_params = {
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_lambda":       5,
        "min_child_weight": 5,
        "objective":        "reg:squarederror",
        "random_state":     42,
    }

    # ── Train R1 (uniform weights) ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training PXGBR-R1 (uniform weights)")
    print("=" * 60)
    w_train_r1 = compute_sample_weights(train_df, "R1")
    model_r1, best_r1 = train_xgb_residual(
        X_train, y_train, w_train_r1, X_val, y_val, base_params,
    )

    # ── Train R2 (MP4 + high-PGV weighted) ───────────────────────────────────
    print("\n" + "=" * 60)
    print("Training PXGBR-R2 (MP4 + high-PGV weights)")
    print("=" * 60)
    w_train_r2 = compute_sample_weights(train_df, "R2")
    model_r2, best_r2 = train_xgb_residual(
        X_train, y_train, w_train_r2, X_val, y_val, base_params,
    )

    # ── Predict on test ───────────────────────────────────────────────────────
    test_df = test_df.copy()
    for model, tag in [(model_r1, "R1"), (model_r2, "R2")]:
        resid_pred = model.predict(X_test)
        test_df[f"residual_pred_{tag}"]   = resid_pred
        test_df[f"pred_log_{tag}"]        = test_df["pred_log_p3"] + resid_pred
        test_df[f"pred_pgv_{tag}"]        = np.exp(test_df[f"pred_log_{tag}"])

    # Monotonic post-processing (apply to R2)
    test_df["pred_log_R2_mono"] = apply_monotonic(test_df, "pred_log_R2")
    test_df["pred_pgv_R2_mono"] = np.exp(test_df["pred_log_R2_mono"])

    # Convenience columns for plotting
    test_df["pred_log_final"] = test_df["pred_log_R2"]
    test_df["pred_pgv_final"] = test_df["pred_pgv_R2"]

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TEST METRICS")
    print("=" * 80)

    all_metrics: Dict[str, Dict] = {}
    tgt_log = test_df["target_log"].values
    sensors = test_df["sensor"].values

    # P3 baseline
    all_metrics["P3"] = compute_metrics(
        test_df["pred_log_p3"].values, tgt_log, sensors, "P3"
    )

    for variant_tag in ["R1", "R2", "R2_mono"]:
        col = f"pred_log_{variant_tag}"
        all_metrics[f"PXGBR-{variant_tag}"] = compute_metrics(
            test_df[col].values, tgt_log, sensors, f"PXGBR-{variant_tag}"
        )

    # Print summary table
    print(f"\n{'Model':<20} {'RMSE(log)':>10} {'RMSE(PGV)':>10} {'R²(log)':>8} {'MP4 RMSE(PGV)':>14}")
    print("-" * 65)
    for name, m in all_metrics.items():
        mp4_rmse = m["per_sensor"].get("MP4", {}).get("rmse_pgv", float("nan"))
        print(f"{name:<20} {m['rmse_log']:>10.4f} {m['rmse_pgv']:>10.4f} "
              f"{m['r2_log']:>8.4f} {mp4_rmse:>14.4f}")

    # Per-sensor detail for R2
    print("\nPXGBR-R2 per-sensor RMSE(PGV) | bias(PGV):")
    for s in SENSOR_ORDER:
        sm = all_metrics["PXGBR-R2"]["per_sensor"].get(s, {})
        print(f"  {s:5s}: RMSE={sm.get('rmse_pgv', 0):.3f}  bias={sm.get('bias_pgv', 0):+.3f}")

    print("\nMonotonicity violation rate (PXGBR-R2 raw):",
          f"{monotonicity_violation_rate(test_df, 'pred_log_R2'):.3f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = MODELS_ROOT / f"{args.output_tag}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    log_dir = Path("outputs/pxgbr_linec_v1")
    log_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    metrics_out = {k: {kk: vv for kk, vv in v.items() if kk != "per_sensor"}
                   for k, v in all_metrics.items()}
    metrics_out["per_sensor"] = {k: v.get("per_sensor", {}) for k, v in all_metrics.items()}
    metrics_out["p3_dir"] = str(p3_dir)
    metrics_out["n_features"] = len(feature_cols)
    metrics_out["best_rounds"] = {"R1": best_r1, "R2": best_r2}
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=float)

    # metrics_table.csv
    rows = []
    for name, m in all_metrics.items():
        row = {"model": name}
        for k in ["rmse_log", "rmse_pgv", "mae_pgv", "r2_log", "bias_pgv"]:
            row[k] = m.get(k, float("nan"))
        for s in SENSOR_ORDER:
            sm = m["per_sensor"].get(s, {})
            row[f"{s}_rmse_pgv"] = sm.get("rmse_pgv", float("nan"))
            row[f"{s}_bias_pgv"] = sm.get("bias_pgv", float("nan"))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "metrics_table.csv", index=False)

    # predictions_test.parquet
    save_cols = ["event_id", "sensor", "track", "distance", "split",
                 "target_log", "target_pgv",
                 "pred_log_p3", "pred_pgv_p3",
                 "pred_log_R1", "pred_pgv_R1",
                 "pred_log_R2", "pred_pgv_R2",
                 "pred_log_R2_mono", "pred_pgv_R2_mono",
                 "residual_log"]
    test_df[[c for c in save_cols if c in test_df.columns]].to_parquet(
        out / "predictions_test.parquet", index=False
    )

    # Save models
    model_r1.save_model(str(out / "model_r1.ubj"))
    model_r2.save_model(str(out / "model_r2.ubj"))

    # Feature importance
    plot_feature_importance(model_r2, feature_cols, out)

    # ── Plots ─────────────────────────────────────────────────────────────────
    dfs_for_plot = {
        "P3":      test_df.rename(columns={}),
        "PXGBR-R2": test_df.rename(columns={}),
    }
    plot_loglog_comparison(
        {"P3": test_df, "PXGBR-R2": test_df}, out
    )
    plot_per_sensor_rmse(all_metrics, out)
    plot_residual_vs_distance(
        {"P3": test_df, "PXGBR-R2": test_df}, out
    )
    for tag in ["representative", "high_pgv", "failures"]:
        try:
            plot_attenuation_profiles(test_df, out, tag=tag)
        except Exception as e:
            print(f"[WARN] attenuation profile plot ({tag}) failed: {e}")

    print(f"\nOutput saved to: {out}")
    print(f"PXGBR-R2 test RMSE(PGV): {all_metrics['PXGBR-R2']['rmse_pgv']:.4f} mm/s")
    print(f"P3 baseline  RMSE(PGV):  {all_metrics['P3']['rmse_pgv']:.4f} mm/s")
    delta = all_metrics["P3"]["rmse_pgv"] - all_metrics["PXGBR-R2"]["rmse_pgv"]
    print(f"Improvement: {delta:+.4f} mm/s ({delta / all_metrics['P3']['rmse_pgv'] * 100:+.1f}%)")


if __name__ == "__main__":
    main()
