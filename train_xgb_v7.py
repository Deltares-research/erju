"""Train XGBoost v7 — Scenario 2: per-event attenuation exponent.

Two XGBoost models are trained independently:
    Model A → predicts c_i (log-intensity at reference distance r0)
    Model B → predicts n_i (event-specific attenuation exponent)

After training, PGV is reconstructed at the original sensor distances using:
    PGV_pred = exp(c_pred - n_pred * log(r / r0))

This is compared against:
    XGBoost v4 (direct model)  : RMSE = 1.79 mm/s
    XGBoost v6 (Scenario 1)    : RMSE = <see xgb_v006 summary>

Important caveat: n_i was estimated from 3-9 sensors per event.
If n_i is too noisy for the model to learn, sensor-level reconstruction
may not improve over Scenario 1.  The OOF / test comparison gives an
honest answer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from src.ml.xgboost.attenuation_utils import (
    compute_all_metrics,
    plot_ci_distribution,
    plot_mean_attenuation_curve,
    plot_measured_vs_predicted_log_log,
    plot_ni_distribution,
    plot_residuals_vs_distance,
    plot_residuals_vs_predicted,
    reconstruct_and_evaluate_event_specific,
    reconstruct_and_evaluate_global,
)
from src.ml.xgboost.config_xgb_v7 import CONFIG
from src.ml.xgboost.xgb_utils import (
    create_build_folder,
    make_event_level_test_split,
    save_json,
)
from src.utils.geometry_utils import apply_corrected_distances

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BANNER = "=== XGBoost Physics-Attenuation Model ==="

BENCHMARK_V4_RMSE = 1.79  # XGBoost v4 direct model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_latest_v4_folder(parquet_root: Path) -> Path:
    builds = sorted(parquet_root.glob("parquet_v004_*"), key=lambda p: p.name)
    if not builds:
        raise FileNotFoundError(f"No parquet_v004_* folders in {parquet_root}")
    return builds[-1]


def _prepare_features(
    df: pd.DataFrame,
    cfg_features: Any,
    targets: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (X, Y_targets, groups)."""
    group_col = cfg_features.group_col
    groups = df[group_col].copy()

    drop_cols = set(
        cfg_features.identifier_cols
        + cfg_features.curve_param_cols
        + cfg_features.string_cols
        + cfg_features.sensor_cols
    )
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    Y = df[targets].copy()
    return X, Y, groups


def _train_single_target(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_cfg: Any,
    verbose: bool,
) -> Tuple[xgb.XGBRegressor, np.ndarray, int]:
    """Train one XGBoost model for one target, return (model, val_preds, best_round)."""
    model = xgb.XGBRegressor(
        objective=model_cfg.objective,
        tree_method=model_cfg.tree_method,
        max_depth=model_cfg.max_depth,
        learning_rate=model_cfg.learning_rate,
        n_estimators=model_cfg.n_estimators,
        subsample=model_cfg.subsample,
        colsample_bytree=model_cfg.colsample_bytree,
        min_child_weight=model_cfg.min_child_weight,
        reg_alpha=model_cfg.reg_alpha,
        reg_lambda=model_cfg.reg_lambda,
        random_state=model_cfg.random_state,
        early_stopping_rounds=model_cfg.early_stopping_rounds,
        eval_metric=model_cfg.eval_metric,
        verbosity=0,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=100 if verbose else False,
    )
    best_round = int(model.best_iteration) + 1
    return model, model.predict(X_val), best_round


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = CONFIG

    print(_BANNER)
    print("=" * 70)
    print("XGBoost v7 — Scenario 2 (per-event n)  |  Predict [c_i, n_i]")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Discover Parquet v4
    # ------------------------------------------------------------------
    parquet_root = Path(r"P:\11210978-erju-ai\holten_parquet")
    if cfg.input_parquet:
        dataset_path = Path(cfg.input_parquet)
        v4_folder = dataset_path.parent
    else:
        v4_folder = _find_latest_v4_folder(parquet_root)
        dataset_path = v4_folder / "dataset_s2.parquet"

    summary_path = v4_folder / "global_fit_summary.json"
    print(f"Input  : {dataset_path}")

    # Load n_global for comparison plots
    n_global: Optional[float] = None
    r0 = cfg.attenuation.r0_m
    if summary_path.exists():
        with open(summary_path) as fh:
            gs = json.load(fh)
        n_global = float(gs["n_global"])
        r0 = float(gs.get("r0_m", r0))
        print(f"n_global (reference) = {n_global:.4f}  (r0 = {r0:.1f} m)")

    # ------------------------------------------------------------------
    # 2. Load sensor-level v2 for evaluation
    # ------------------------------------------------------------------
    sensor_parquet = Path(cfg.eval.sensor_parquet)
    # Auto-discover latest v2 build if configured path is missing
    if not sensor_parquet.exists():
        parquet_root = Path(r"P:\11210978-erju-ai\holten_parquet")
        v2_builds = sorted(parquet_root.glob("parquet_v002_*"), key=lambda p: p.name)
        if not v2_builds:
            raise FileNotFoundError("No parquet_v002_* builds found.")
        sensor_parquet = v2_builds[-1] / "dataset.parquet"
        print(f"  (auto-discovered latest v2: {sensor_parquet})")
    df_sensor = pd.read_parquet(sensor_parquet)
    if cfg.eval.exclude_sensor_ids:
        df_sensor = df_sensor[~df_sensor["sensor_id"].isin(cfg.eval.exclude_sensor_ids)]
    # Ensure corrected distance columns are present
    if "effective_distance_to_active_track_m" not in df_sensor.columns:
        df_sensor = apply_corrected_distances(df_sensor)

    # ------------------------------------------------------------------
    # 3. Load event-level Scenario 2 dataset
    # ------------------------------------------------------------------
    df = pd.read_parquet(dataset_path)
    build_dir = create_build_folder(
        output_root=cfg.output_root_path(),
        version_name=cfg.output.version_name,
    )
    plots_dir = build_dir / cfg.output.plots_subfolder
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Build  : {build_dir}")

    targets = [cfg.features.target_ci_col, cfg.features.target_ni_col]
    df = df[df[targets].notna().all(axis=1)].reset_index(drop=True)
    print(
        f"\n[1/6] Event-level dataset (quality_flag=1): {len(df):,} events, "
        f"{len(df.columns)} columns"
    )

    # Describe fitted targets
    print(f"       c_i: mean={df['c_i'].mean():.3f}  std={df['c_i'].std():.3f}")
    print(
        f"       n_i: mean={df['n_i'].mean():.3f}  std={df['n_i'].std():.3f}  "
        f"p5={df['n_i'].quantile(0.05):.3f}  p95={df['n_i'].quantile(0.95):.3f}"
    )

    # ------------------------------------------------------------------
    # 4. Train / test split
    # ------------------------------------------------------------------
    df_tv, df_test = make_event_level_test_split(
        df=df,
        group_col=cfg.features.group_col,
        test_fraction=cfg.split.test_fraction,
        random_seed=cfg.split.random_seed,
    )
    print(f"\n[2/6] Train+Val: {len(df_tv):,} events  |  Test: {len(df_test):,} events")

    # ------------------------------------------------------------------
    # 5. Prepare features
    # ------------------------------------------------------------------
    X_tv, Y_tv, groups_tv = _prepare_features(df_tv, cfg.features, targets)
    X_test, Y_test, _ = _prepare_features(df_test, cfg.features, targets)
    print(f"\n[3/6] Feature count: {X_tv.shape[1]}")

    # ------------------------------------------------------------------
    # 6. GroupKFold CV — train c_i model and n_i model separately
    # ------------------------------------------------------------------
    print(f"\n[4/6] {cfg.split.n_cv_folds}-fold CV (c_i model + n_i model) ...")
    gkf = GroupKFold(n_splits=cfg.split.n_cv_folds)

    oof_ci = np.full(len(X_tv), np.nan)
    oof_ni = np.full(len(X_tv), np.nan)
    best_rounds_ci: List[int] = []
    best_rounds_ni: List[int] = []

    tv_event_ids = set(df_tv[cfg.features.group_col].unique())
    df_sensor_tv = df_sensor[df_sensor["event_id"].isin(tv_event_ids)]

    for fold_idx, (tr_idx, val_idx) in enumerate(
        gkf.split(X_tv, Y_tv["c_i"], groups=groups_tv), start=1
    ):
        X_tr = X_tv.iloc[tr_idx]
        X_val = X_tv.iloc[val_idx]
        y_ci_tr = Y_tv["c_i"].iloc[tr_idx]
        y_ci_val = Y_tv["c_i"].iloc[val_idx]
        y_ni_tr = Y_tv["n_i"].iloc[tr_idx]
        y_ni_val = Y_tv["n_i"].iloc[val_idx]

        _, ci_val_preds, br_ci = _train_single_target(
            X_tr, y_ci_tr, X_val, y_ci_val, cfg.model, verbose=False
        )
        _, ni_val_preds, br_ni = _train_single_target(
            X_tr, y_ni_tr, X_val, y_ni_val, cfg.model, verbose=False
        )

        oof_ci[val_idx] = ci_val_preds
        oof_ni[val_idx] = ni_val_preds
        best_rounds_ci.append(br_ci)
        best_rounds_ni.append(br_ni)

        # Sensor-level reconstruction
        val_eids = df_tv[cfg.features.group_col].iloc[val_idx].values
        c_s = pd.Series(ci_val_preds, index=val_eids)
        n_s = pd.Series(ni_val_preds, index=val_eids)
        df_sv = df_sensor_tv[df_sensor_tv["event_id"].isin(val_eids)]
        _, _, fold_metrics = reconstruct_and_evaluate_event_specific(
            df_sensor=df_sv,
            c_pred_series=c_s,
            n_pred_series=n_s,
            r0=r0,
            event_col="event_id",
            distance_col=cfg.eval.distance_col,
            pgv_col=cfg.eval.pgv_col,
        )
        if cfg.verbose_folds:
            print(
                f"  Fold {fold_idx}/{cfg.split.n_cv_folds} | "
                f"br_ci={br_ci:4d}  br_ni={br_ni:4d} | "
                f"sensor RMSE={fold_metrics['rmse_mms']:.4f} mm/s  "
                f"RMSE(log)={fold_metrics['rmse_log']:.4f}"
            )

    mean_br_ci = int(round(np.mean(best_rounds_ci)))
    mean_br_ni = int(round(np.mean(best_rounds_ni)))

    # OOF sensor-level
    oof_c_s = pd.Series(oof_ci, index=df_tv[cfg.features.group_col].values)
    oof_n_s = pd.Series(oof_ni, index=df_tv[cfg.features.group_col].values)
    oof_true, oof_pred_mms, oof_sensor_metrics = (
        reconstruct_and_evaluate_event_specific(
            df_sensor=df_sensor_tv,
            c_pred_series=oof_c_s,
            n_pred_series=oof_n_s,
            r0=r0,
            event_col="event_id",
            distance_col=cfg.eval.distance_col,
            pgv_col=cfg.eval.pgv_col,
        )
    )
    ci_rmse = float(np.sqrt(mean_squared_error(Y_tv["c_i"].to_numpy(), oof_ci)))
    ni_rmse = float(np.sqrt(mean_squared_error(Y_tv["n_i"].to_numpy(), oof_ni)))
    print(f"\n  OOF c_i RMSE  : {ci_rmse:.4f}")
    print(f"  OOF n_i RMSE  : {ni_rmse:.4f}")
    print(
        f"  OOF sensor RMSE(PGV)  : {oof_sensor_metrics['rmse_mms']:.4f} mm/s "
        f"(v4 benchmark: {BENCHMARK_V4_RMSE:.2f})"
    )
    print(f"  OOF sensor RMSE(log)  : {oof_sensor_metrics['rmse_log']:.4f}")
    print(f"  Mean best round ci/ni : {mean_br_ci} / {mean_br_ni}")

    # ------------------------------------------------------------------
    # 7. Final models
    # ------------------------------------------------------------------
    print(f"\n[5/6] Training final models ({mean_br_ci}/{mean_br_ni} rounds) ...")

    def _final(n_rounds: int) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            objective=cfg.model.objective,
            tree_method=cfg.model.tree_method,
            max_depth=cfg.model.max_depth,
            learning_rate=cfg.model.learning_rate,
            n_estimators=n_rounds,
            subsample=cfg.model.subsample,
            colsample_bytree=cfg.model.colsample_bytree,
            min_child_weight=cfg.model.min_child_weight,
            reg_alpha=cfg.model.reg_alpha,
            reg_lambda=cfg.model.reg_lambda,
            random_state=cfg.model.random_state,
            verbosity=0,
        )

    model_ci = _final(mean_br_ci)
    model_ci.fit(X_tv, Y_tv["c_i"], verbose=False)

    model_ni = _final(mean_br_ni)
    model_ni.fit(X_tv, Y_tv["n_i"], verbose=False)

    # ------------------------------------------------------------------
    # 8. Test evaluation
    # ------------------------------------------------------------------
    print(f"\n[6/6] Evaluating on held-out test set ...")
    test_eids = df_test[cfg.features.group_col].values
    test_ci_pred = model_ci.predict(X_test)
    test_ni_pred = model_ni.predict(X_test)

    df_sensor_test = df_sensor[df_sensor["event_id"].isin(set(test_eids))]
    test_c_s = pd.Series(test_ci_pred, index=test_eids)
    test_n_s = pd.Series(test_ni_pred, index=test_eids)

    y_test_true, y_test_pred, test_sensor_metrics = (
        reconstruct_and_evaluate_event_specific(
            df_sensor=df_sensor_test,
            c_pred_series=test_c_s,
            n_pred_series=test_n_s,
            r0=r0,
            event_col="event_id",
            distance_col=cfg.eval.distance_col,
            pgv_col=cfg.eval.pgv_col,
        )
    )
    delta = test_sensor_metrics["rmse_mms"] - BENCHMARK_V4_RMSE
    sign = "+" if delta >= 0 else ""
    print(
        f"  Test sensor RMSE(PGV) : {test_sensor_metrics['rmse_mms']:.4f} mm/s "
        f"(v4: {BENCHMARK_V4_RMSE:.2f})"
    )
    print(f"  Test sensor RMSE(log) : {test_sensor_metrics['rmse_log']:.4f}")
    print(f"  Test sensor R²        : {test_sensor_metrics['r2_mms']:.4f}")
    print(
        f"  vs. XGBoost v4        : {sign}{delta:.4f} mm/s "
        f"({'WORSE' if delta > 0 else 'BETTER ✓'})"
    )

    # ------------------------------------------------------------------
    # 9. Diagnostic plots
    # ------------------------------------------------------------------
    print("\n  Generating plots ...")

    plot_measured_vs_predicted_log_log(
        y_true=y_test_true,
        y_pred=y_test_pred,
        metrics=test_sensor_metrics,
        title="XGBoost v7 — Test set (Scenario 2, per-event n)",
        out_path=plots_dir / "test_measured_vs_predicted.png",
    )
    test_dist = df_sensor_test.loc[
        (df_sensor_test[cfg.eval.pgv_col] > 0)
        & (df_sensor_test[cfg.eval.distance_col] > 0)
        & df_sensor_test["event_id"].isin(set(test_eids)),
        cfg.eval.distance_col,
    ].to_numpy()
    plot_residuals_vs_distance(
        y_true=y_test_true,
        y_pred=y_test_pred,
        distances=test_dist,
        title="XGBoost v7 — Residuals vs distance (test)",
        out_path=plots_dir / "test_residuals_vs_distance.png",
    )
    plot_residuals_vs_predicted(
        y_true=y_test_true,
        y_pred=y_test_pred,
        title="XGBoost v7 — Residuals vs predicted (test)",
        out_path=plots_dir / "test_residuals_vs_predicted.png",
    )
    plot_ci_distribution(
        ci_values=test_ci_pred,
        title="XGBoost v7 — Predicted c_i (test)",
        out_path=plots_dir / "test_ci_distribution.png",
    )
    plot_ni_distribution(
        ni_values=test_ni_pred,
        n_global=n_global,
        title="XGBoost v7 — Predicted n_i (test) vs global n",
        out_path=plots_dir / "test_ni_distribution.png",
    )
    # Also n_i true distribution from fitted curves
    plot_ni_distribution(
        ni_values=df_test["n_i"].to_numpy(),
        n_global=n_global,
        title="XGBoost v7 — True fitted n_i (test events)",
        out_path=plots_dir / "test_ni_true_distribution.png",
    )

    # ------------------------------------------------------------------
    # 10. Save artefacts
    # ------------------------------------------------------------------
    model_ci.save_model(str(build_dir / cfg.output.model_ci_filename))
    model_ni.save_model(str(build_dir / cfg.output.model_ni_filename))

    oof_df = pd.DataFrame(
        {
            "event_id": df_tv[cfg.features.group_col].values,
            "c_i_true": Y_tv["c_i"].to_numpy(),
            "c_i_pred": oof_ci,
            "n_i_true": Y_tv["n_i"].to_numpy(),
            "n_i_pred": oof_ni,
        }
    )
    oof_df.to_parquet(build_dir / cfg.output.oof_predictions_filename, index=False)

    summary = {
        "build_folder": str(build_dir),
        "input_parquet_v4_s2": str(dataset_path),
        "sensor_parquet_v2": str(sensor_parquet),
        "n_events_train_val": int(len(df_tv)),
        "n_events_test": int(len(df_test)),
        "n_features": int(X_tv.shape[1]),
        "r0_m": float(r0),
        "n_global_reference": n_global,
        "oof_ci_rmse": float(ci_rmse),
        "oof_ni_rmse": float(ni_rmse),
        "oof_sensor_metrics": oof_sensor_metrics,
        "test_sensor_metrics": test_sensor_metrics,
        "mean_best_round_ci": mean_br_ci,
        "mean_best_round_ni": mean_br_ni,
        "benchmark_xgb_v4_rmse_mms": BENCHMARK_V4_RMSE,
        "delta_vs_benchmark_mms": float(
            test_sensor_metrics["rmse_mms"] - BENCHMARK_V4_RMSE
        ),
        "experiment_notes": cfg.experiment_notes,
    }
    save_json(build_dir / cfg.output.summary_filename, summary)
    save_json(build_dir / cfg.output.config_snapshot_filename, cfg.as_dict())

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Build folder   : {build_dir}")
    print(
        f"OOF  RMSE(log)={oof_sensor_metrics['rmse_log']:.4f}  "
        f"RMSE={oof_sensor_metrics['rmse_mms']:.4f} mm/s  "
        f"R²={oof_sensor_metrics['r2_mms']:.4f}"
    )
    print(
        f"Test RMSE(log)={test_sensor_metrics['rmse_log']:.4f}  "
        f"RMSE={test_sensor_metrics['rmse_mms']:.4f} mm/s  "
        f"R²={test_sensor_metrics['r2_mms']:.4f}"
    )
    print(f"XGBoost v4 benchmark : {BENCHMARK_V4_RMSE:.4f} mm/s")
    print(
        f"Delta vs benchmark   : {sign}{delta:.4f} mm/s "
        f"({'WORSE' if delta > 0 else 'BETTER ✓'})"
    )


if __name__ == "__main__":
    main()
