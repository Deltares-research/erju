"""Train XGBoost v8 — two-stage residual-on-physics model.

Architecture
------------
Stage 1 (event-level)
    FO features + train metadata  →  predict c_i
    c_i = log(PGV at reference distance r0)

Stage 2 (sensor-level)
    event features + distance features + z_phys + c_hat  →  log(PGV)

    where:
        z_phys = c_hat - n_global * log(r / r0)
        c_hat  = stage-1 prediction (OOF during training, model prediction at test)

Two variants
    Variant A  target = log(PGV)          [z_phys used as a feature]
    Variant B  target = log(PGV) - z_phys [explicit residual; final pred = z_phys + eps_hat]

Leakage-free CV structure (per outer fold)
    1. Re-fit n_global from training sensor rows only.
    2. Inner CV on training events to generate OOF c_hat.
       (stage-1 trained on 4/5 of training events, predicts on the 1/5 held out)
    3. Stage-1 trained on ALL training events to predict val c_hat.
    4. Training physics prior: z_phys_oof from OOF c_hat.
    5. Stage-2 trained on training sensor rows using z_phys_oof.
    6. Val evaluation: z_phys_val from stage-1 c_hat predictions.

Benchmark
    XGBoost v4 direct sensor-level model  →  RMSE = 1.79 mm/s
    Oracle S1 (true c_i + global n)        →  RMSE = 1.72 mm/s  (ceiling)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from src.ml.xgboost.config_xgb_v8 import CONFIG
from src.ml.xgboost.xgb_utils import create_build_folder, save_json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BANNER = "=== XGBoost v8 — Residual-on-Physics ==="

BENCHMARK_V4_RMSE = 1.79  # XGBoost v4 direct sensor-level model
ORACLE_S1_RMSE = 1.7218  # Oracle S1 (true c_i + global n) — representation ceiling

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _find_latest_v4_folder(parquet_root: Path) -> Path:
    builds = sorted(parquet_root.glob("parquet_v004_*"), key=lambda p: p.name)
    if not builds:
        raise FileNotFoundError(f"No parquet_v004_* folders in {parquet_root}")
    return builds[-1]


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------


def fit_n_global_from_rows(
    df: pd.DataFrame, distance_col: str, pgv_col: str, event_col: str, r0: float
) -> float:
    """Re-fit global attenuation exponent from a subset of sensor rows.

    Uses within-event (fixed-effects) OLS:
        log(PGV) - mean_event(log(PGV)) = -n * (log(r/r0) - mean_event(log(r/r0)))

    Returns fitted n_global (scalar).
    """
    df = df[[event_col, distance_col, pgv_col]].dropna()
    df = df[df[pgv_col] > 0][df[distance_col] > 0]

    log_pgv = np.log(df[pgv_col].values)
    log_r = np.log(df[distance_col].values / r0)

    # Demean within each event
    df2 = df.copy()
    df2["log_pgv"] = log_pgv
    df2["log_r"] = log_r
    means = df2.groupby(event_col)[["log_pgv", "log_r"]].transform("mean")
    dy = df2["log_pgv"].values - means["log_pgv"].values
    dx = df2["log_r"].values - means["log_r"].values

    denom = (dx * dx).sum()
    if denom < 1e-12:
        return 0.8162  # fallback
    n_global = -(dy * dx).sum() / denom
    return float(n_global)


def compute_z_phys(
    c_hat: np.ndarray, r: np.ndarray, n_global: float, r0: float
) -> np.ndarray:
    """Physics prior: z_phys = c_hat - n_global * log(r / r0)."""
    return c_hat - n_global * np.log(r / r0)


def add_distance_physics_features(
    df: pd.DataFrame, r: np.ndarray, n_global: float, r0: float
) -> pd.DataFrame:
    """Add safe (non-leaking) distance-derived physics features."""
    df = df.copy()
    r_clipped = np.maximum(r, 0.1)
    df["feat_log_r_ratio"] = np.log(r_clipped / r0)
    df["feat_inv_r"] = 1.0 / r_clipped
    df["feat_inv_sqrt_r"] = 1.0 / np.sqrt(r_clipped)
    df["feat_inv_r2"] = 1.0 / (r_clipped**2)
    df["feat_n_log_r_ratio"] = -n_global * np.log(r_clipped / r0)
    return df


# ---------------------------------------------------------------------------
# Feature matrix preparation
# ---------------------------------------------------------------------------


def _drop_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df.drop(columns=[c for c in cols if c in df.columns])


def build_stage1_features(
    df_event: pd.DataFrame, cfg
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y_ci, groups) for the event-level stage-1 model."""
    fc = cfg.features
    groups = df_event[fc.group_col].copy()
    y = df_event["c_i"].copy()

    drop = (
        [fc.group_col, "site_id"]
        + fc.attenuation_cols
        + fc.string_cols
        + [fc.sensor_id_col, fc.pgv_col, fc.distance_col, fc.side_col]
    )
    X = _drop_cols(df_event, drop)
    # Drop remaining non-numeric (safety)
    X = X.select_dtypes(include=[np.number])
    return X, y, groups


def build_stage2_features(
    df_sensor: pd.DataFrame,
    c_hat_map: pd.Series,  # index = event_id
    n_global: float,
    cfg,
    target_is_residual: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build sensor-level feature matrix for stage 2.

    Parameters
    ----------
    df_sensor
        Sensor-level rows (subset of v2).
    c_hat_map
        Series of predicted c_hat values, indexed by event_id.
    n_global
        Attenuation exponent for this fold.
    cfg
        XGBv8Config instance.
    target_is_residual
        If True, target = log(PGV) - z_phys  (Variant B).
        If False, target = log(PGV)            (Variant A).
    """
    fc = cfg.features
    r0 = cfg.physics.r0_m

    df = df_sensor.copy()

    # Map c_hat to each sensor row
    df["c_hat"] = df[fc.group_col].map(c_hat_map)

    # Distance
    r = df[fc.distance_col].values.astype(float)

    # Physics prior
    df["z_phys"] = compute_z_phys(df["c_hat"].values, r, n_global, r0)

    # Distance-physics features
    df = add_distance_physics_features(df, r, n_global, r0)

    # Encode side_of_track as integer if present
    if fc.side_col in df.columns:
        if df[fc.side_col].dtype == object:
            df[fc.side_col] = df[fc.side_col].astype("category").cat.codes

    # Target: log(PGV)
    log_pgv = np.log(df[fc.pgv_col].values.astype(float))
    z_phys = df["z_phys"].values

    if target_is_residual:
        y = pd.Series(log_pgv - z_phys, index=df.index)
    else:
        y = pd.Series(log_pgv, index=df.index)

    groups = df[fc.group_col].copy()

    # Drop columns that must not appear in the feature matrix
    drop = (
        [fc.group_col, fc.sensor_id_col, fc.pgv_col]
        + fc.attenuation_cols
        + fc.string_cols
    )
    # Keep distance_col and side_col as raw features
    X = _drop_cols(df, drop)
    X = X.select_dtypes(include=[np.number])

    return X, y, groups


# ---------------------------------------------------------------------------
# XGBoost helpers
# ---------------------------------------------------------------------------


def _make_xgb(
    model_cfg, n_estimators_override: Optional[int] = None
) -> xgb.XGBRegressor:
    n = (
        n_estimators_override
        if n_estimators_override is not None
        else model_cfg.n_estimators
    )
    kwargs = dict(
        objective=model_cfg.objective,
        tree_method=model_cfg.tree_method,
        max_depth=model_cfg.max_depth,
        learning_rate=model_cfg.learning_rate,
        n_estimators=n,
        subsample=model_cfg.subsample,
        colsample_bytree=model_cfg.colsample_bytree,
        min_child_weight=model_cfg.min_child_weight,
        reg_alpha=model_cfg.reg_alpha,
        reg_lambda=model_cfg.reg_lambda,
        random_state=model_cfg.random_state,
    )
    if hasattr(model_cfg, "early_stopping_rounds"):
        kwargs["early_stopping_rounds"] = model_cfg.early_stopping_rounds
        kwargs["eval_metric"] = model_cfg.eval_metric
    return xgb.XGBRegressor(**kwargs)


def _fit_with_early_stop(model, X_tr, y_tr, X_val, y_val, verbose=False):
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=verbose,
    )
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _metrics(
    log_pgv_true: np.ndarray, log_pgv_pred: np.ndarray, label: str = ""
) -> Dict[str, float]:
    pgv_true = np.exp(log_pgv_true)
    pgv_pred = np.exp(log_pgv_pred)

    rmse_log = float(np.sqrt(np.mean((log_pgv_true - log_pgv_pred) ** 2)))
    rmse = float(np.sqrt(np.mean((pgv_true - pgv_pred) ** 2)))
    mae = float(np.mean(np.abs(pgv_true - pgv_pred)))
    r2 = float(r2_score(pgv_true, pgv_pred))
    smape = float(
        100 * np.mean(2 * np.abs(pgv_true - pgv_pred) / (pgv_true + pgv_pred + 1e-9))
    )
    if label:
        print(
            f"    {label}: RMSE={rmse:.4f} mm/s  RMSE(log)={rmse_log:.4f}  R²={r2:.4f}"
        )
    return dict(rmse=rmse, rmse_log=rmse_log, mae=mae, r2=r2, smape=smape)


# ---------------------------------------------------------------------------
# Inner-CV OOF c_hat generation
# ---------------------------------------------------------------------------


def generate_oof_c_hat(
    X_event: pd.DataFrame,
    y_ci: pd.Series,
    event_ids: pd.Series,
    cfg,
) -> np.ndarray:
    """Generate OOF c_hat for all training events via inner GroupKFold CV.

    Uses a fixed number of estimators (no early stopping) for speed and to
    avoid needing a further validation split inside the inner fold.
    """
    n_inner = cfg.split.n_inner_folds
    n_est = cfg.stage1.inner_oof_n_estimators

    c_hat_oof = np.full(len(X_event), np.nan)
    inner_cv = GroupKFold(n_splits=n_inner)

    for _, (tr_idx, val_idx) in enumerate(
        inner_cv.split(X_event, y_ci, groups=event_ids)
    ):
        mdl = _make_xgb(cfg.stage1, n_estimators_override=n_est)
        # Remove early_stopping_rounds for fixed-round inner models
        mdl.set_params(early_stopping_rounds=None)
        mdl.fit(X_event.iloc[tr_idx], y_ci.iloc[tr_idx], verbose=False)
        c_hat_oof[val_idx] = mdl.predict(X_event.iloc[val_idx])

    return c_hat_oof


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_scatter(
    log_true: np.ndarray, log_pred: np.ndarray, title: str, out_path: Path
) -> None:
    pgv_true = np.exp(log_true)
    pgv_pred = np.exp(log_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pgv_true, pgv_pred, s=4, alpha=0.3, color="steelblue")
    lim = [min(pgv_true.min(), pgv_pred.min()), max(pgv_true.max(), pgv_pred.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Measured PGV (mm/s)")
    ax.set_ylabel("Predicted PGV (mm/s)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _plot_residuals_vs_distance(
    r: np.ndarray,
    log_true: np.ndarray,
    log_pred: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    resid = log_true - log_pred
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(r, resid, s=4, alpha=0.3, color="steelblue")
    ax.axhline(0, color="red", lw=1, ls="--")
    ax.set_xlabel("Sensor distance (m)")
    ax.set_ylabel("log(PGV) residual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = CONFIG
    fc = cfg.features

    print(_BANNER)
    print("=" * 70)
    print("XGBoost v8 — Residual-on-Physics  |  Two variants (A + B)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Discover paths
    # ------------------------------------------------------------------
    parquet_root = Path(r"P:\11210978-erju-ai\holten_parquet")

    if cfg.input_parquet_v4:
        v4_path = Path(cfg.input_parquet_v4)
        v4_dir = v4_path.parent
    else:
        v4_dir = _find_latest_v4_folder(parquet_root)
        v4_path = v4_dir / "dataset.parquet"

    global_fit_json = v4_dir / "global_fit_summary.json"
    per_event_parquet = v4_dir / "attenuation_per_event.parquet"

    v2_path = Path(cfg.input_parquet_v2)
    build_dir = create_build_folder(cfg.output_root_path(), cfg.output.version_name)
    plots_dir = build_dir / cfg.output.plots_subfolder
    plots_dir.mkdir(exist_ok=True)

    with open(global_fit_json) as f:
        fit_summary = json.load(f)
    n_global_global = fit_summary["n_global"]
    r0 = fit_summary["r0_m"]

    print(f"\nInput v4  : {v4_path}")
    print(f"Input v2  : {v2_path}")
    print(f"n_global  = {n_global_global:.4f}  (r0 = {r0} m)")
    print(f"Build dir : {build_dir}")

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    print("\n[1/7] Loading data ...")
    df_v4 = pd.read_parquet(v4_path)  # event-level: one row per event
    df_v2 = pd.read_parquet(v2_path)  # sensor-level: one row per event×sensor

    # Filter sensors
    if fc.exclude_sensor_ids and fc.sensor_id_col in df_v2.columns:
        df_v2 = df_v2[~df_v2[fc.sensor_id_col].isin(fc.exclude_sensor_ids)]

    # Keep only valid rows for sensor-level work
    df_v2 = df_v2.dropna(subset=[fc.distance_col, fc.pgv_col])
    df_v2 = df_v2[df_v2[fc.pgv_col] > 0][df_v2[fc.distance_col] > 0]

    # Per-event attenuation params (for c_i targets)
    df_params = pd.read_parquet(per_event_parquet)
    # Merge c_i into v4 event dataframe if not already there
    if "c_i" not in df_v4.columns:
        df_v4 = df_v4.merge(df_params[["event_id", "c_i"]], on="event_id", how="left")

    # Keep only events with valid c_i
    df_v4 = df_v4.dropna(subset=["c_i"])
    valid_events = set(df_v4["event_id"].unique())
    df_v2 = df_v2[df_v2["event_id"].isin(valid_events)]

    print(f"  Event-level rows : {len(df_v4):,}  events")
    print(
        f"  Sensor-level rows: {len(df_v2):,}  rows  |  "
        f"{df_v2['event_id'].nunique():,} events"
    )

    # ------------------------------------------------------------------
    # 3. Train / test split (event-level)
    # ------------------------------------------------------------------
    print("\n[2/7] Train/test split ...")
    rng = np.random.default_rng(cfg.split.random_seed)
    all_events = df_v4["event_id"].unique()
    n_test = max(1, int(len(all_events) * cfg.split.test_fraction))
    test_events = set(rng.choice(all_events, size=n_test, replace=False))
    train_val_events = set(all_events) - test_events

    df_v4_tv = df_v4[df_v4["event_id"].isin(train_val_events)].reset_index(drop=True)
    df_v4_test = df_v4[df_v4["event_id"].isin(test_events)].reset_index(drop=True)
    df_v2_tv = df_v2[df_v2["event_id"].isin(train_val_events)].reset_index(drop=True)
    df_v2_test = df_v2[df_v2["event_id"].isin(test_events)].reset_index(drop=True)

    print(
        f"  Train+Val: {len(train_val_events):,} events  |  "
        f"Test: {len(test_events):,} events"
    )

    # ------------------------------------------------------------------
    # 4. Build stage-1 feature matrix (event-level)
    # ------------------------------------------------------------------
    print("\n[3/7] Preparing stage-1 features ...")
    X1_tv, y_ci_tv, grp1_tv = build_stage1_features(df_v4_tv, cfg)
    X1_test, y_ci_test, _ = build_stage1_features(df_v4_test, cfg)
    event_ids_tv = df_v4_tv["event_id"]
    event_ids_test = df_v4_test["event_id"]

    print(f"  Stage-1 feature count: {X1_tv.shape[1]}")

    # ------------------------------------------------------------------
    # 5. Outer GroupKFold CV
    # ------------------------------------------------------------------
    print(f"\n[4/7] {cfg.split.n_cv_folds}-fold outer CV ...")
    outer_cv = GroupKFold(n_splits=cfg.split.n_cv_folds)

    oof_log_pgv_true_A = np.full(len(df_v2_tv), np.nan)
    oof_log_pgv_pred_A = np.full(len(df_v2_tv), np.nan)
    oof_log_pgv_pred_B = np.full(len(df_v2_tv), np.nan)
    oof_c_hat_events = np.full(len(df_v4_tv), np.nan)

    # We need to map sensor rows back to a flat index for OOF storage
    # Store sensor row indices for each fold
    best_rounds_s1: List[int] = []
    best_rounds_s2a: List[int] = []
    best_rounds_s2b: List[int] = []

    fold_metrics_A: List[Dict] = []
    fold_metrics_B: List[Dict] = []

    for fold_idx, (ev_tr_idx, ev_val_idx) in enumerate(
        outer_cv.split(X1_tv, y_ci_tv, groups=event_ids_tv)
    ):
        # Event-level subsets
        X1_tr = X1_tv.iloc[ev_tr_idx]
        y_ci_tr = y_ci_tv.iloc[ev_tr_idx]
        X1_val = X1_tv.iloc[ev_val_idx]
        y_ci_val = y_ci_tv.iloc[ev_val_idx]
        ev_ids_tr = event_ids_tv.iloc[ev_tr_idx]
        ev_ids_val = event_ids_tv.iloc[ev_val_idx]

        train_ev_set = set(ev_ids_tr.unique())
        val_ev_set = set(ev_ids_val.unique())

        # Sensor-level subsets
        sensor_tr_mask = df_v2_tv["event_id"].isin(train_ev_set)
        sensor_val_mask = df_v2_tv["event_id"].isin(val_ev_set)
        df_s_tr = df_v2_tv[sensor_tr_mask].reset_index(drop=True)
        df_s_val = df_v2_tv[sensor_val_mask].reset_index(drop=True)

        # ---- (a) Re-fit n_global from training sensor rows ----
        n_global_fold = fit_n_global_from_rows(
            df_s_tr, fc.distance_col, fc.pgv_col, "event_id", r0
        )

        # ---- (b) Inner CV: OOF c_hat for training events ----
        c_hat_oof_tr = generate_oof_c_hat(X1_tr, y_ci_tr, ev_ids_tr, cfg)
        c_hat_oof_tr_map = (
            pd.Series(c_hat_oof_tr, index=ev_ids_tr.values).groupby(level=0).first()
        )  # one c_hat per event_id

        # ---- (c) Train stage-1 on all training events ----
        s1_model = _make_xgb(cfg.stage1)
        _fit_with_early_stop(s1_model, X1_tr, y_ci_tr, X1_val, y_ci_val, verbose=False)
        br_s1 = s1_model.best_iteration + 1
        best_rounds_s1.append(br_s1)

        # Predict c_hat for validation events
        c_hat_val = s1_model.predict(X1_val)
        c_hat_val_map = pd.Series(c_hat_val, index=ev_ids_val.values)

        # Store OOF c_hat for event-level diagnostics
        oof_c_hat_events[ev_val_idx] = c_hat_val

        # ---- (d) Build sensor-level feature matrices ----
        # Training: use OOF c_hat (avoids leakage)
        X2_tr_A, y2_tr_A, _ = build_stage2_features(
            df_s_tr, c_hat_oof_tr_map, n_global_fold, cfg, target_is_residual=False
        )
        X2_tr_B, y2_tr_B, _ = build_stage2_features(
            df_s_tr, c_hat_oof_tr_map, n_global_fold, cfg, target_is_residual=True
        )
        # Validation
        X2_val_A, y2_val_A, _ = build_stage2_features(
            df_s_val, c_hat_val_map, n_global_fold, cfg, target_is_residual=False
        )
        X2_val_B, y2_val_B, _ = build_stage2_features(
            df_s_val, c_hat_val_map, n_global_fold, cfg, target_is_residual=True
        )

        # Align columns (training set is the reference)
        common_cols_A = [c for c in X2_tr_A.columns if c in X2_val_A.columns]
        common_cols_B = [c for c in X2_tr_B.columns if c in X2_val_B.columns]

        # ---- (e) Train stage-2 Variant A ----
        s2a_model = _make_xgb(cfg.stage2)
        _fit_with_early_stop(
            s2a_model,
            X2_tr_A[common_cols_A],
            y2_tr_A,
            X2_val_A[common_cols_A],
            y2_val_A,
            verbose=False,
        )
        best_rounds_s2a.append(s2a_model.best_iteration + 1)
        log_pred_A = s2a_model.predict(X2_val_A[common_cols_A])

        # ---- (f) Train stage-2 Variant B ----
        s2b_model = _make_xgb(cfg.stage2)
        z_phys_val = X2_val_B["z_phys"].values
        _fit_with_early_stop(
            s2b_model,
            X2_tr_B[common_cols_B],
            y2_tr_B,
            X2_val_B[common_cols_B],
            y2_val_B,
            verbose=False,
        )
        best_rounds_s2b.append(s2b_model.best_iteration + 1)
        eps_pred_B = s2b_model.predict(X2_val_B[common_cols_B])
        log_pred_B = z_phys_val + eps_pred_B

        # ---- (g) Store OOF predictions ----
        val_sensor_flat_idx = np.where(sensor_val_mask)[0]
        log_pgv_true_val = np.log(df_s_val[fc.pgv_col].values.astype(float))

        oof_log_pgv_true_A[val_sensor_flat_idx] = log_pgv_true_val
        oof_log_pgv_pred_A[val_sensor_flat_idx] = log_pred_A
        oof_log_pgv_pred_B[val_sensor_flat_idx] = log_pred_B

        m_A = _metrics(log_pgv_true_val, log_pred_A)
        m_B = _metrics(log_pgv_true_val, log_pred_B)
        fold_metrics_A.append(m_A)
        fold_metrics_B.append(m_B)

        print(
            f"  Fold {fold_idx + 1}/{cfg.split.n_cv_folds} | "
            f"n_global={n_global_fold:.4f} | "
            f"br_s1={br_s1:3d}  br_s2a={best_rounds_s2a[-1]:3d}  "
            f"br_s2b={best_rounds_s2b[-1]:3d}"
        )
        print(
            f"    Var-A: RMSE={m_A['rmse']:.4f} mm/s  "
            f"RMSE(log)={m_A['rmse_log']:.4f}  R²={m_A['r2']:.4f}"
        )
        print(
            f"    Var-B: RMSE={m_B['rmse']:.4f} mm/s  "
            f"RMSE(log)={m_B['rmse_log']:.4f}  R²={m_B['r2']:.4f}"
        )

    # ------------------------------------------------------------------
    # 6. OOF summary
    # ------------------------------------------------------------------
    valid_mask = ~np.isnan(oof_log_pgv_true_A)
    oof_m_A = _metrics(oof_log_pgv_true_A[valid_mask], oof_log_pgv_pred_A[valid_mask])
    oof_m_B = _metrics(oof_log_pgv_true_A[valid_mask], oof_log_pgv_pred_B[valid_mask])

    mean_br_s1 = int(round(np.mean(best_rounds_s1)))
    mean_br_s2a = int(round(np.mean(best_rounds_s2a)))
    mean_br_s2b = int(round(np.mean(best_rounds_s2b)))

    print(f"\n{'=' * 70}")
    print(f"  OOF summary")
    print(
        f"  Variant A (direct log-PGV):  "
        f"RMSE={oof_m_A['rmse']:.4f} mm/s  RMSE(log)={oof_m_A['rmse_log']:.4f}  "
        f"R²={oof_m_A['r2']:.4f}"
    )
    print(
        f"  Variant B (residual):        "
        f"RMSE={oof_m_B['rmse']:.4f} mm/s  RMSE(log)={oof_m_B['rmse_log']:.4f}  "
        f"R²={oof_m_B['r2']:.4f}"
    )
    print(f"  Mean best rounds: s1={mean_br_s1}  s2a={mean_br_s2a}  s2b={mean_br_s2b}")

    # ------------------------------------------------------------------
    # 7. Final models on all train+val data
    # ------------------------------------------------------------------
    print(
        f"\n[5/7] Training final models (rounds: s1={mean_br_s1}, "
        f"s2a={mean_br_s2a}, s2b={mean_br_s2b}) ..."
    )

    # Stage 1 final
    s1_final = _make_xgb(cfg.stage1, n_estimators_override=mean_br_s1)
    s1_final.set_params(early_stopping_rounds=None)
    s1_final.fit(X1_tv, y_ci_tv, verbose=False)

    # Re-fit n_global on all train+val sensor rows
    n_global_final = fit_n_global_from_rows(
        df_v2_tv, fc.distance_col, fc.pgv_col, "event_id", r0
    )
    print(f"  n_global (train+val) = {n_global_final:.4f}")

    # OOF c_hat for all train+val events (for stage-2 training, avoid leakage)
    c_hat_oof_all = generate_oof_c_hat(X1_tv, y_ci_tv, event_ids_tv, cfg)
    c_hat_oof_all_map = (
        pd.Series(c_hat_oof_all, index=event_ids_tv.values).groupby(level=0).first()
    )

    # Stage-2 features for train+val
    X2_tv_A, y2_tv_A, _ = build_stage2_features(
        df_v2_tv, c_hat_oof_all_map, n_global_final, cfg, target_is_residual=False
    )
    X2_tv_B, y2_tv_B, _ = build_stage2_features(
        df_v2_tv, c_hat_oof_all_map, n_global_final, cfg, target_is_residual=True
    )

    s2a_final = _make_xgb(cfg.stage2, n_estimators_override=mean_br_s2a)
    s2a_final.set_params(early_stopping_rounds=None)
    s2a_final.fit(X2_tv_A, y2_tv_A, verbose=False)

    s2b_final = _make_xgb(cfg.stage2, n_estimators_override=mean_br_s2b)
    s2b_final.set_params(early_stopping_rounds=None)
    s2b_final.fit(X2_tv_B, y2_tv_B, verbose=False)

    # ------------------------------------------------------------------
    # 8. Test evaluation
    # ------------------------------------------------------------------
    print(f"\n[6/7] Evaluating on held-out test set ...")

    # Stage-1 predictions for test events
    c_hat_test = s1_final.predict(X1_test)
    c_hat_test_map = pd.Series(c_hat_test, index=event_ids_test.values)

    # Stage-2 features for test
    X2_test_A, y2_test_A, _ = build_stage2_features(
        df_v2_test, c_hat_test_map, n_global_final, cfg, target_is_residual=False
    )
    X2_test_B, y2_test_B, _ = build_stage2_features(
        df_v2_test, c_hat_test_map, n_global_final, cfg, target_is_residual=True
    )

    # Align columns to training set
    cols_A = [c for c in X2_tv_A.columns if c in X2_test_A.columns]
    cols_B = [c for c in X2_tv_B.columns if c in X2_test_B.columns]

    log_pgv_test_true = np.log(df_v2_test[fc.pgv_col].values.astype(float))
    r_test = df_v2_test[fc.distance_col].values.astype(float)

    # Variant A
    log_pred_test_A = s2a_final.predict(X2_test_A[cols_A])
    test_m_A = _metrics(log_pgv_test_true, log_pred_test_A)

    # Variant B
    z_phys_test = X2_test_B["z_phys"].values
    eps_pred_test_B = s2b_final.predict(X2_test_B[cols_B])
    log_pred_test_B = z_phys_test + eps_pred_test_B
    test_m_B = _metrics(log_pgv_test_true, log_pred_test_B)

    print(f"\n  Variant A (direct log-PGV):")
    print(
        f"    Test RMSE = {test_m_A['rmse']:.4f} mm/s  "
        f"RMSE(log)={test_m_A['rmse_log']:.4f}  R²={test_m_A['r2']:.4f}"
    )
    print(
        f"    vs XGBoost v4 benchmark: {test_m_A['rmse'] - BENCHMARK_V4_RMSE:+.4f} mm/s"
    )

    print(f"\n  Variant B (explicit residual):")
    print(
        f"    Test RMSE = {test_m_B['rmse']:.4f} mm/s  "
        f"RMSE(log)={test_m_B['rmse_log']:.4f}  R²={test_m_B['r2']:.4f}"
    )
    print(
        f"    vs XGBoost v4 benchmark: {test_m_B['rmse'] - BENCHMARK_V4_RMSE:+.4f} mm/s"
    )

    print(f"\n  Oracle S1 ceiling (reference): {ORACLE_S1_RMSE:.4f} mm/s")

    # ------------------------------------------------------------------
    # 9. Plots
    # ------------------------------------------------------------------
    print(f"\n[7/7] Saving artefacts ...")

    # OOF scatter plots
    _plot_scatter(
        oof_log_pgv_true_A[valid_mask],
        oof_log_pgv_pred_A[valid_mask],
        f"Variant A OOF  RMSE={oof_m_A['rmse']:.3f} mm/s  R²={oof_m_A['r2']:.3f}",
        plots_dir / "oof_varA_scatter.png",
    )
    _plot_scatter(
        oof_log_pgv_true_A[valid_mask],
        oof_log_pgv_pred_B[valid_mask],
        f"Variant B OOF  RMSE={oof_m_B['rmse']:.3f} mm/s  R²={oof_m_B['r2']:.3f}",
        plots_dir / "oof_varB_scatter.png",
    )

    # Test scatter plots
    _plot_scatter(
        log_pgv_test_true,
        log_pred_test_A,
        f"Variant A Test  RMSE={test_m_A['rmse']:.3f} mm/s  R²={test_m_A['r2']:.3f}",
        plots_dir / "test_varA_scatter.png",
    )
    _plot_scatter(
        log_pgv_test_true,
        log_pred_test_B,
        f"Variant B Test  RMSE={test_m_B['rmse']:.3f} mm/s  R²={test_m_B['r2']:.3f}",
        plots_dir / "test_varB_scatter.png",
    )

    # Residuals vs distance
    _plot_residuals_vs_distance(
        r_test,
        log_pgv_test_true,
        log_pred_test_A,
        "Variant A: residuals vs distance (test)",
        plots_dir / "test_varA_residuals_vs_dist.png",
    )
    _plot_residuals_vs_distance(
        r_test,
        log_pgv_test_true,
        log_pred_test_B,
        "Variant B: residuals vs distance (test)",
        plots_dir / "test_varB_residuals_vs_dist.png",
    )

    # Feature importance plots
    for s2_model, name, cols in [
        (s2a_final, "varA", cols_A),
        (s2b_final, "varB", cols_B),
    ]:
        imp = pd.Series(s2_model.feature_importances_, index=cols)
        top = imp.nlargest(25)
        fig, ax = plt.subplots(figsize=(8, 6))
        top.sort_values().plot.barh(ax=ax, color="steelblue")
        ax.set_title(f"Stage-2 {name} top-25 feature importances")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(plots_dir / f"s2_{name}_importance.png", dpi=120)
        plt.close()

    # Save models
    s1_final.save_model(str(build_dir / cfg.output.model_s1_filename))
    s2a_final.save_model(str(build_dir / cfg.output.model_s2a_filename))
    s2b_final.save_model(str(build_dir / cfg.output.model_s2b_filename))

    # Save n_global used in final model
    final_physics = {"n_global": n_global_final, "r0_m": r0}
    save_json(build_dir / "final_physics.json", final_physics)

    # Save OOF predictions
    oof_df = df_v2_tv[[fc.group_col, fc.distance_col, fc.pgv_col]].copy()
    oof_df["log_pgv_true"] = np.log(oof_df[fc.pgv_col].clip(lower=1e-9))
    oof_df["log_pgv_pred_A"] = oof_log_pgv_pred_A
    oof_df["log_pgv_pred_B"] = oof_log_pgv_pred_B
    oof_df.to_parquet(build_dir / cfg.output.oof_predictions_filename, index=False)

    # Summary JSON
    summary = {
        "version": cfg.output.version_name,
        "build_folder": str(build_dir),
        "n_train_val_events": len(train_val_events),
        "n_test_events": len(test_events),
        "n_global_final": n_global_final,
        "r0_m": r0,
        "mean_best_rounds": {"s1": mean_br_s1, "s2a": mean_br_s2a, "s2b": mean_br_s2b},
        "oof": {
            "varA": oof_m_A,
            "varB": oof_m_B,
        },
        "test": {
            "varA": test_m_A,
            "varB": test_m_B,
        },
        "benchmarks": {
            "xgb_v4_direct_rmse": BENCHMARK_V4_RMSE,
            "oracle_s1_rmse": ORACLE_S1_RMSE,
        },
    }
    save_json(build_dir / cfg.output.summary_filename, summary)
    save_json(build_dir / cfg.output.config_snapshot_filename, cfg.as_dict())

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Build folder : {build_dir}")
    print(f"\nOOF results:")
    print(
        f"  Variant A : RMSE={oof_m_A['rmse']:.4f} mm/s  "
        f"RMSE(log)={oof_m_A['rmse_log']:.4f}  R²={oof_m_A['r2']:.4f}"
    )
    print(
        f"  Variant B : RMSE={oof_m_B['rmse']:.4f} mm/s  "
        f"RMSE(log)={oof_m_B['rmse_log']:.4f}  R²={oof_m_B['r2']:.4f}"
    )
    print(f"\nTest results:")
    print(
        f"  Variant A : RMSE={test_m_A['rmse']:.4f} mm/s  "
        f"(vs v4: {test_m_A['rmse'] - BENCHMARK_V4_RMSE:+.4f})"
    )
    print(
        f"  Variant B : RMSE={test_m_B['rmse']:.4f} mm/s  "
        f"(vs v4: {test_m_B['rmse'] - BENCHMARK_V4_RMSE:+.4f})"
    )
    print(f"\nOracle S1 ceiling : {ORACLE_S1_RMSE:.4f} mm/s")
    print(f"XGBoost v4 direct : {BENCHMARK_V4_RMSE:.4f} mm/s")


if __name__ == "__main__":
    main()
