"""Train XGBoost v6 — Scenario 1: global attenuation exponent.

The model predicts c_i = log(PGV at reference distance r0) for each event.
After training, PGV is reconstructed at the original sensor distances and
evaluated against measured PGV.

Workflow
--------
1.  Discover the latest Parquet v4 build folder (event-level dataset).
2.  Load n_global from the build's global_fit_summary.json.
3.  Load sensor-level Parquet v2 for sensor-level reconstruction evaluation.
4.  Hold out a fixed test set (by event_id — same fraction/seed as XGBoost v4).
5.  GroupKFold CV on train+val events:
      - Features = FO octave-band stats + texture + train metadata
      - Target   = c_i (log-intensity at r0)
      - After each fold: reconstruct PGV at sensor level and compute metrics
6.  Train final model on all train+val data (mean OOF best_round).
7.  Evaluate on held-out test events (sensor-level reconstruction).
8.  Save all artefacts to a versioned build folder.

Benchmark comparison (printed in summary)
    XGBoost v4 (direct, sensor-level): RMSE = 1.79 mm/s
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from src.ml.xgboost.attenuation_utils import (
    compute_all_metrics,
    plot_ci_distribution,
    plot_mean_attenuation_curve,
    plot_measured_vs_predicted_log_log,
    plot_residuals_vs_distance,
    plot_residuals_vs_predicted,
    reconstruct_and_evaluate_global,
)
from src.ml.xgboost.config_xgb_v6 import CONFIG
from src.ml.xgboost.xgb_utils import (
    create_build_folder,
    make_event_level_test_split,
    save_json,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BANNER = "=== XGBoost Physics-Attenuation Model ==="

BENCHMARK_RMSE = 1.79  # XGBoost v4 sensor-level direct model


def _find_latest_parquet_v4(parquet_root: Path) -> Tuple[Path, Path]:
    """Return (dataset.parquet, global_fit_summary.json) for latest v4 build."""
    builds = sorted(parquet_root.glob("parquet_v004_*"), key=lambda p: p.name)
    if not builds:
        raise FileNotFoundError(f"No parquet_v004_* folders found in {parquet_root}")
    latest = builds[-1]
    dataset = latest / "dataset.parquet"
    summary = latest / "global_fit_summary.json"
    if not dataset.exists():
        raise FileNotFoundError(f"dataset.parquet not found in {latest}")
    if not summary.exists():
        raise FileNotFoundError(f"global_fit_summary.json not found in {latest}")
    return dataset, summary


def _prepare_features(
    df: pd.DataFrame,
    cfg_features: Any,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, groups) from event-level DataFrame."""
    target = cfg_features.target_col
    group_col = cfg_features.group_col

    groups = df[group_col].copy()
    y = df[target].copy()

    drop_cols = set(
        cfg_features.identifier_cols
        + cfg_features.curve_param_cols
        + cfg_features.string_cols
        + cfg_features.sensor_cols
    )
    drop_cols.add(target)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y, groups


def _train_fold(
    fold_idx: int,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_cfg: Any,
    n_folds: int,
    verbose: bool,
) -> Tuple[xgb.XGBRegressor, np.ndarray, int]:
    """Train one XGBoost fold and return (model, val_preds, best_round)."""
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
    val_preds = model.predict(X_val)
    if verbose:
        val_rmse_ci = float(np.sqrt(mean_squared_error(y_val, val_preds)))
        print(
            f"  Fold {fold_idx}/{n_folds} | best_round={best_round:4d} "
            f"| val RMSE(c_i)={val_rmse_ci:.4f}"
        )
    return model, val_preds, best_round


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = CONFIG

    print(_BANNER)
    print("=" * 70)
    print("XGBoost v6 — Scenario 1 (global n)  |  Predict c_i per event")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Discover Parquet v4
    # ------------------------------------------------------------------
    if cfg.input_parquet:
        dataset_path = Path(cfg.input_parquet)
        v4_folder = dataset_path.parent
        summary_path = v4_folder / "global_fit_summary.json"
    else:
        parquet_root = Path(r"P:\11210978-erju-ai\holten_parquet")
        dataset_path, summary_path = _find_latest_parquet_v4(parquet_root)
        v4_folder = dataset_path.parent

    print(f"Input  : {dataset_path}")

    # ------------------------------------------------------------------
    # 2. Load global n
    # ------------------------------------------------------------------
    with open(summary_path) as fh:
        global_summary = json.load(fh)
    n_global = float(global_summary["n_global"])
    r0 = float(global_summary.get("r0_m", cfg.attenuation.r0_m))
    print(f"n_global = {n_global:.4f}  (r0 = {r0:.1f} m)")

    # ------------------------------------------------------------------
    # 3. Load sensor-level v2 for sensor-level evaluation
    # ------------------------------------------------------------------
    sensor_parquet = Path(cfg.eval.sensor_parquet)
    print(f"\nSensor-level v2: {sensor_parquet}")
    df_sensor = pd.read_parquet(sensor_parquet)
    if cfg.eval.exclude_sensor_ids:
        df_sensor = df_sensor[~df_sensor["sensor_id"].isin(cfg.eval.exclude_sensor_ids)]

    # ------------------------------------------------------------------
    # 4. Load event-level v4 + setup build folder
    # ------------------------------------------------------------------
    df = pd.read_parquet(dataset_path)
    build_dir = create_build_folder(
        output_root=cfg.output_root_path(),
        version_name=cfg.output.version_name,
    )
    plots_dir = build_dir / cfg.output.plots_subfolder
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Build  : {build_dir}")
    print(f"\n[1/6] Event-level dataset: {len(df):,} events, {len(df.columns)} columns")

    # Filter rows missing the target
    target_col = cfg.features.target_col
    df = df[df[target_col].notna()].reset_index(drop=True)
    print(f"       After dropping NaN c_i: {len(df):,} events")

    # ------------------------------------------------------------------
    # 5. Event-level train / test split
    # ------------------------------------------------------------------
    df_tv, df_test = make_event_level_test_split(
        df=df,
        group_col=cfg.features.group_col,
        test_fraction=cfg.split.test_fraction,
        random_seed=cfg.split.random_seed,
    )
    print(f"\n[2/6] Train+Val: {len(df_tv):,} events  |  Test: {len(df_test):,} events")

    # ------------------------------------------------------------------
    # 6. Prepare features
    # ------------------------------------------------------------------
    X_tv, y_tv, groups_tv = _prepare_features(df_tv, cfg.features)
    X_test, y_test, _ = _prepare_features(df_test, cfg.features)
    print(f"\n[3/6] Feature count: {X_tv.shape[1]}")

    # ------------------------------------------------------------------
    # 7. GroupKFold CV (but since each row IS one event, groups = event_id,
    #    so GroupKFold behaves as KFold here)
    # ------------------------------------------------------------------
    print(f"\n[4/6] {cfg.split.n_cv_folds}-fold CV on c_i prediction ...")
    gkf = GroupKFold(n_splits=cfg.split.n_cv_folds)
    oof_preds_ci = np.full(len(X_tv), np.nan)
    best_rounds: List[int] = []
    fold_sensor_metrics: List[Dict] = []

    # Subset sensor DataFrame for train+val events
    tv_event_ids = set(df_tv[cfg.features.group_col].unique())
    df_sensor_tv = df_sensor[df_sensor["event_id"].isin(tv_event_ids)]

    for fold_idx, (tr_idx, val_idx) in enumerate(
        gkf.split(X_tv, y_tv, groups=groups_tv), start=1
    ):
        X_tr, y_tr = X_tv.iloc[tr_idx], y_tv.iloc[tr_idx]
        X_val, y_val = X_tv.iloc[val_idx], y_tv.iloc[val_idx]

        _model, val_preds, best_round = _train_fold(
            fold_idx=fold_idx,
            X_tr=X_tr,
            y_tr=y_tr,
            X_val=X_val,
            y_val=y_val,
            model_cfg=cfg.model,
            n_folds=cfg.split.n_cv_folds,
            verbose=cfg.verbose_folds,
        )
        oof_preds_ci[val_idx] = val_preds
        best_rounds.append(best_round)

        # Sensor-level reconstruction for this fold
        val_event_ids = df_tv[cfg.features.group_col].iloc[val_idx].values
        c_pred_s = pd.Series(val_preds, index=val_event_ids)
        df_sensor_val = df_sensor_tv[df_sensor_tv["event_id"].isin(val_event_ids)]
        _, _, sensor_metrics = reconstruct_and_evaluate_global(
            df_sensor=df_sensor_val,
            c_pred_series=c_pred_s,
            n_global=n_global,
            r0=r0,
            event_col="event_id",
            distance_col=cfg.eval.distance_col,
            pgv_col=cfg.eval.pgv_col,
        )
        fold_sensor_metrics.append(sensor_metrics)
        if cfg.verbose_folds:
            print(
                f"         → sensor RMSE={sensor_metrics['rmse_mms']:.4f} mm/s  "
                f"RMSE(log)={sensor_metrics['rmse_log']:.4f}  "
                f"R²={sensor_metrics['r2_mms']:.4f}"
            )

    # OOF summary on c_i
    oof_rmse_ci = float(np.sqrt(mean_squared_error(y_tv.to_numpy(), oof_preds_ci)))
    mean_best_round = int(round(np.mean(best_rounds)))

    # OOF sensor-level reconstruction
    oof_c_pred = pd.Series(oof_preds_ci, index=df_tv[cfg.features.group_col].values)
    _, oof_y_pred_mms, oof_sensor_metrics = reconstruct_and_evaluate_global(
        df_sensor=df_sensor_tv,
        c_pred_series=oof_c_pred,
        n_global=n_global,
        r0=r0,
        event_col="event_id",
        distance_col=cfg.eval.distance_col,
        pgv_col=cfg.eval.pgv_col,
    )
    df_sensor_tv_eval = df_sensor_tv[
        (df_sensor_tv[cfg.eval.pgv_col] > 0)
        & (df_sensor_tv[cfg.eval.distance_col] > 0)
        & df_sensor_tv["event_id"].isin(df_tv[cfg.features.group_col].values)
    ]
    oof_y_true_mms = df_sensor_tv_eval[cfg.eval.pgv_col].to_numpy()

    print(f"\n  OOF c_i  RMSE : {oof_rmse_ci:.4f}")
    print(
        f"  OOF sensor RMSE(PGV) : {oof_sensor_metrics['rmse_mms']:.4f} mm/s "
        f"(benchmark v4: {BENCHMARK_RMSE:.2f} mm/s)"
    )
    print(f"  OOF sensor RMSE(log) : {oof_sensor_metrics['rmse_log']:.4f}")
    print(f"  OOF sensor R²        : {oof_sensor_metrics['r2_mms']:.4f}")
    print(f"  Mean best round      : {mean_best_round}")

    # ------------------------------------------------------------------
    # 8. Final model
    # ------------------------------------------------------------------
    print(f"\n[5/6] Training final model ({mean_best_round} rounds on train+val) ...")
    final_model = xgb.XGBRegressor(
        objective=cfg.model.objective,
        tree_method=cfg.model.tree_method,
        max_depth=cfg.model.max_depth,
        learning_rate=cfg.model.learning_rate,
        n_estimators=mean_best_round,
        subsample=cfg.model.subsample,
        colsample_bytree=cfg.model.colsample_bytree,
        min_child_weight=cfg.model.min_child_weight,
        reg_alpha=cfg.model.reg_alpha,
        reg_lambda=cfg.model.reg_lambda,
        random_state=cfg.model.random_state,
        verbosity=0,
    )
    final_model.fit(X_tv, y_tv, verbose=False)

    # ------------------------------------------------------------------
    # 9. Test set evaluation
    # ------------------------------------------------------------------
    print(f"\n[6/6] Evaluating on held-out test set ...")
    test_event_ids = df_test[cfg.features.group_col].values
    test_c_pred = final_model.predict(X_test)
    c_pred_test_s = pd.Series(test_c_pred, index=test_event_ids)

    df_sensor_test = df_sensor[df_sensor["event_id"].isin(set(test_event_ids))]
    y_test_true, y_test_pred_mms, test_sensor_metrics = reconstruct_and_evaluate_global(
        df_sensor=df_sensor_test,
        c_pred_series=c_pred_test_s,
        n_global=n_global,
        r0=r0,
        event_col="event_id",
        distance_col=cfg.eval.distance_col,
        pgv_col=cfg.eval.pgv_col,
    )
    print(
        f"  Test sensor RMSE(PGV) : {test_sensor_metrics['rmse_mms']:.4f} mm/s "
        f"(benchmark v4: {BENCHMARK_RMSE:.2f} mm/s)"
    )
    print(f"  Test sensor RMSE(log) : {test_sensor_metrics['rmse_log']:.4f}")
    print(f"  Test sensor R²        : {test_sensor_metrics['r2_mms']:.4f}")

    delta = test_sensor_metrics["rmse_mms"] - BENCHMARK_RMSE
    sign = "+" if delta >= 0 else ""
    print(
        f"  vs. XGBoost v4        : {sign}{delta:.4f} mm/s "
        f"({'WORSE' if delta > 0 else 'BETTER'})"
    )

    # ------------------------------------------------------------------
    # 10. Diagnostic plots
    # ------------------------------------------------------------------
    print("\n  Generating plots ...")

    # Sensor-level test: measured vs predicted
    plot_measured_vs_predicted_log_log(
        y_true=y_test_true,
        y_pred=y_test_pred_mms,
        metrics=test_sensor_metrics,
        title="XGBoost v6 — Test set (Scenario 1, global n)",
        out_path=plots_dir / "test_measured_vs_predicted.png",
    )
    # Residuals vs distance on test set
    test_dist = df_sensor_test.loc[
        (df_sensor_test[cfg.eval.pgv_col] > 0)
        & (df_sensor_test[cfg.eval.distance_col] > 0)
        & df_sensor_test["event_id"].isin(set(test_event_ids)),
        cfg.eval.distance_col,
    ].to_numpy()
    plot_residuals_vs_distance(
        y_true=y_test_true,
        y_pred=y_test_pred_mms,
        distances=test_dist,
        title="XGBoost v6 — Residuals vs distance (test)",
        out_path=plots_dir / "test_residuals_vs_distance.png",
    )
    plot_residuals_vs_predicted(
        y_true=y_test_true,
        y_pred=y_test_pred_mms,
        title="XGBoost v6 — Residuals vs predicted (test)",
        out_path=plots_dir / "test_residuals_vs_predicted.png",
    )

    # c_i distribution (predicted vs fitted)
    plot_ci_distribution(
        ci_values=test_c_pred,
        title="XGBoost v6 — Predicted c_i distribution (test)",
        out_path=plots_dir / "test_ci_distribution.png",
    )

    # Mean attenuation curve (using all sensor data)
    plot_mean_attenuation_curve(
        df_sensor=df_sensor,
        n_global=n_global,
        r0=r0,
        out_path=plots_dir / "mean_attenuation_curve.png",
        pgv_col=cfg.eval.pgv_col,
        distance_col=cfg.eval.distance_col,
    )

    # ------------------------------------------------------------------
    # 11. Save artefacts
    # ------------------------------------------------------------------
    final_model.save_model(str(build_dir / cfg.output.model_filename))

    # OOF predictions (event-level c_i)
    oof_df = pd.DataFrame(
        {
            "event_id": df_tv[cfg.features.group_col].values,
            "c_i_true": y_tv.to_numpy(),
            "c_i_pred": oof_preds_ci,
        }
    )
    oof_df.to_parquet(build_dir / cfg.output.oof_predictions_filename, index=False)

    summary = {
        "build_folder": str(build_dir),
        "input_parquet_v4": str(dataset_path),
        "sensor_parquet_v2": str(sensor_parquet),
        "n_events_train_val": int(len(df_tv)),
        "n_events_test": int(len(df_test)),
        "n_features": int(X_tv.shape[1]),
        "n_global": float(n_global),
        "r0_m": float(r0),
        "oof_rmse_ci": float(oof_rmse_ci),
        "oof_sensor_metrics": oof_sensor_metrics,
        "test_sensor_metrics": test_sensor_metrics,
        "mean_best_round": mean_best_round,
        "best_rounds_per_fold": best_rounds,
        "benchmark_xgb_v4_rmse_mms": BENCHMARK_RMSE,
        "delta_vs_benchmark_mms": float(
            test_sensor_metrics["rmse_mms"] - BENCHMARK_RMSE
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
    print(f"XGBoost v4 benchmark : {BENCHMARK_RMSE:.4f} mm/s")
    print(
        f"Delta vs benchmark   : {sign}{delta:.4f} mm/s "
        f"({'WORSE' if delta > 0 else 'BETTER ✓'})"
    )


if __name__ == "__main__":
    main()
