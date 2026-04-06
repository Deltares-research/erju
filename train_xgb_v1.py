"""Train XGBoost v2 regression model for PGV_z prediction.

Workflow:
  1. Load Parquet dataset
  2. Hold out a fixed test set (by event_id)
  3. Prepare features + feature engineering (geometry features)
  4. GroupKFold cross-validation on train+val events → OOF predictions
  5. Train final model on all train+val data using mean OOF best_round
  6. Evaluate final model on held-out test set
  7. Save all artifacts to a versioned build folder

All metrics and saved artifacts are in original mm/s units (target is
back-transformed from log-space before reporting).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from config_xgb import CONFIG
from src.ml.xgb_utils import (
    build_split_manifest,
    create_build_folder,
    engineer_features,
    extract_feature_importance,
    load_dataset,
    make_event_level_test_split,
    plot_feature_importance,
    plot_predicted_vs_actual,
    plot_residuals,
    plot_training_curves,
    prepare_features,
    save_json,
    save_model,
    save_oof_predictions,
    train_final_model,
    train_groupkfold,
    write_build_log,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    cfg = CONFIG
    plots_dir: Path

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    parquet_path = cfg.input_parquet_path()
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet dataset not found: {parquet_path}")

    build_dir = create_build_folder(
        output_root=cfg.output_root_path(),
        version_name=cfg.output.version_name,
    )
    plots_dir = build_dir / cfg.output.plots_subfolder
    plots_dir.mkdir(parents=True, exist_ok=True)

    log_lines: List[str] = [
        f"Build folder : {build_dir}",
        f"Input Parquet: {parquet_path}",
    ]

    print("=" * 70)
    print("XGBoost v2 — PGV_z Regression")
    print("=" * 70)
    print(f"Input : {parquet_path}")
    print(f"Build : {build_dir}")

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("\n[1/6] Loading dataset ...")
    df = load_dataset(parquet_path)
    n_events = df["event_id"].nunique()
    print(
        f"      Rows: {len(df):,}  |  Unique events: {n_events:,}  |  Columns: {len(df.columns)}"
    )
    log_lines.append(
        f"Rows (raw): {len(df)}  |  Events (raw): {n_events}  |  Columns: {len(df.columns)}"
    )

    # Exclude sensors whose stored targets are not in mm/s (e.g. in-track
    # sensors that record raw acceleration in g).
    if cfg.exclude_sensor_ids:
        before = len(df)
        df = df[~df["sensor_id"].isin(cfg.exclude_sensor_ids)].copy()
        dropped = before - len(df)
        print(
            f"      Excluded sensor_ids {cfg.exclude_sensor_ids}: "
            f"-{dropped:,} rows → {len(df):,} rows remaining"
        )
        log_lines.append(
            f"Excluded sensor_ids: {cfg.exclude_sensor_ids} (-{dropped} rows)"
        )
        n_events = df["event_id"].nunique()

    print(f"      Final: {len(df):,} rows  |  {n_events:,} unique events")

    # ------------------------------------------------------------------
    # 2. Train/test split by event_id
    # ------------------------------------------------------------------
    print("\n[2/6] Creating event-level train/test split ...")
    train_val_df, test_df = make_event_level_test_split(
        df=df,
        group_col=cfg.features.group_col,
        test_fraction=cfg.split.test_fraction,
        random_seed=cfg.split.random_seed,
    )
    manifest = build_split_manifest(train_val_df, test_df, cfg.features.group_col)
    print(
        f"      Train+Val: {manifest['n_train_val_events']} events / {manifest['n_train_val_rows']} rows"
    )
    print(
        f"      Test     : {manifest['n_test_events']} events / {manifest['n_test_rows']} rows"
    )

    save_json(build_dir / cfg.output.split_manifest_filename, manifest)
    log_lines.append(
        f"Train+Val events: {manifest['n_train_val_events']} | Test events: {manifest['n_test_events']}"
    )

    # ------------------------------------------------------------------
    # 3. Prepare features + feature engineering
    # ------------------------------------------------------------------
    print("\n[3/6] Preparing features ...")
    X_tv, y_tv, groups_tv = prepare_features(
        df=train_val_df,
        target_col=cfg.features.target_col,
        identifier_cols=cfg.features.identifier_cols,
        string_cols=cfg.features.string_cols,
    )
    X_test, y_test, _ = prepare_features(
        df=test_df,
        target_col=cfg.features.target_col,
        identifier_cols=cfg.features.identifier_cols,
        string_cols=cfg.features.string_cols,
    )
    X_tv = engineer_features(X_tv, cfg.fe)
    X_test = engineer_features(X_test, cfg.fe)
    feature_names: List[str] = list(X_tv.columns)
    print(f"      Feature count: {len(feature_names)}")
    if cfg.fe.add_geometry_features:
        print("      + feat_log1p_distance, feat_inv_distance_sq added")
    if cfg.fe.log_transform_target:
        print("      + log1p target transform enabled (metrics reported in mm/s)")
    log_lines.append(f"Feature count: {len(feature_names)}")

    # Apply log1p to the target for training.  y_tv / y_test are kept in
    # original mm/s for computing reported metrics and saving artifacts.
    if cfg.fe.log_transform_target:
        y_tv_model = pd.Series(np.log1p(y_tv.values), index=y_tv.index)
        y_test_model = pd.Series(np.log1p(y_test.values), index=y_test.index)
    else:
        y_tv_model = y_tv
        y_test_model = y_test

    # ------------------------------------------------------------------
    # 4. GroupKFold cross-validation
    # ------------------------------------------------------------------
    cv_note = " (fold metrics in log-space)" if cfg.fe.log_transform_target else ""
    print(
        f"\n[4/6] GroupKFold CV ({cfg.split.n_cv_folds} folds by event_id){cv_note} ..."
    )
    oof_preds_model, fold_history, best_rounds = train_groupkfold(
        X=X_tv,
        y=y_tv_model,
        groups=groups_tv,
        model_cfg=cfg.model,
        n_folds=cfg.split.n_cv_folds,
        verbose=cfg.verbose_folds,
    )

    # Back-transform OOF predictions to mm/s before computing metrics.
    oof_preds_mms = (
        np.expm1(oof_preds_model) if cfg.fe.log_transform_target else oof_preds_model
    )
    oof_metrics = _metrics(y_tv.values, oof_preds_mms)
    mean_best_round = int(round(np.mean(best_rounds)))
    print(f"\n  OOF RMSE : {oof_metrics['rmse']:.4f} mm/s")
    print(f"  OOF MAE  : {oof_metrics['mae']:.4f} mm/s")
    print(f"  OOF R²   : {oof_metrics['r2']:.4f}")
    print(f"  Mean best round: {mean_best_round} (used for final model)")

    training_history = {
        "folds": fold_history,
        "oof_metrics": oof_metrics,
        "best_rounds": best_rounds,
        "mean_best_round": mean_best_round,
    }
    save_json(build_dir / cfg.output.training_history_filename, training_history)

    save_oof_predictions(
        train_val_df=train_val_df,
        oof_preds=oof_preds_mms,
        target_col=cfg.features.target_col,
        output_path=build_dir / cfg.output.oof_predictions_filename,
    )

    log_lines.append(
        f"OOF RMSE={oof_metrics['rmse']:.4f}  MAE={oof_metrics['mae']:.4f}  R2={oof_metrics['r2']:.4f}"
    )
    log_lines.append(f"Mean best round: {mean_best_round}")

    # ------------------------------------------------------------------
    # 5. Final model on all train+val data
    # ------------------------------------------------------------------
    print(f"\n[5/6] Training final model ({mean_best_round} rounds on train+val) ...")
    final_model = train_final_model(
        X=X_tv,
        y=y_tv_model,
        model_cfg=cfg.model,
        n_rounds=mean_best_round,
    )
    save_model(final_model, build_dir / cfg.output.final_model_filename)

    # ------------------------------------------------------------------
    # 6. Evaluate on held-out test set
    # ------------------------------------------------------------------
    print("\n[6/6] Evaluating on held-out test set ...")
    test_preds_model = final_model.predict(X_test)
    # Back-transform to mm/s for interpretable metrics.
    test_preds_mms = (
        np.expm1(test_preds_model) if cfg.fe.log_transform_target else test_preds_model
    )
    test_metrics = _metrics(y_test.values, test_preds_mms)

    print(f"  Test RMSE : {test_metrics['rmse']:.4f} mm/s")
    print(f"  Test MAE  : {test_metrics['mae']:.4f} mm/s")
    print(f"  Test R²   : {test_metrics['r2']:.4f}")

    log_lines.append(
        f"Test RMSE={test_metrics['rmse']:.4f}  MAE={test_metrics['mae']:.4f}  R2={test_metrics['r2']:.4f}"
    )

    # ------------------------------------------------------------------
    # Save config snapshot and summary
    # ------------------------------------------------------------------
    save_json(build_dir / cfg.output.config_snapshot_filename, cfg.as_dict())

    summary = {
        "experiment_notes": cfg.experiment_notes,
        "build_folder": str(build_dir),
        "input_parquet": str(parquet_path),
        "n_rows": int(len(df)),
        "n_events": int(n_events),
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "feature_engineering": {
            "log_transform_target": cfg.fe.log_transform_target,
            "add_geometry_features": cfg.fe.add_geometry_features,
        },
        "split": manifest,
        "oof_metrics": oof_metrics,
        "test_metrics": test_metrics,
        "mean_best_round": mean_best_round,
        "best_rounds_per_fold": best_rounds,
    }
    save_json(build_dir / "summary.json", summary)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    importance_df = extract_feature_importance(final_model, feature_names)
    importance_df.to_csv(
        build_dir / cfg.output.feature_importance_filename, index=False
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\n  Generating plots ...")

    plot_predicted_vs_actual(
        y_true=y_tv.values,
        y_pred=oof_preds_mms,
        title="OOF: Predicted vs Actual PGV_z",
        output_path=plots_dir / "oof_predicted_vs_actual.png",
    )
    plot_residuals(
        y_true=y_tv.values,
        y_pred=oof_preds_mms,
        title="OOF",
        output_path=plots_dir / "oof_residuals.png",
    )
    plot_predicted_vs_actual(
        y_true=y_test.values,
        y_pred=test_preds_mms,
        title="Test: Predicted vs Actual PGV_z",
        output_path=plots_dir / "test_predicted_vs_actual.png",
    )
    plot_residuals(
        y_true=y_test.values,
        y_pred=test_preds_mms,
        title="Test",
        output_path=plots_dir / "test_residuals.png",
    )
    plot_training_curves(
        fold_history=fold_history,
        output_path=plots_dir / "training_curves.png",
    )
    plot_feature_importance(
        importance_df=importance_df,
        top_n=30,
        output_path=plots_dir / "feature_importance.png",
    )

    write_build_log(build_dir / cfg.output.build_log_filename, log_lines)

    # ------------------------------------------------------------------
    # Final summary print
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Build folder : {build_dir}")
    print(
        f"OOF  RMSE={oof_metrics['rmse']:.4f}  MAE={oof_metrics['mae']:.4f}  R²={oof_metrics['r2']:.4f}"
    )
    print(
        f"Test RMSE={test_metrics['rmse']:.4f}  MAE={test_metrics['mae']:.4f}  R²={test_metrics['r2']:.4f}"
    )
    print(f"Final model  : {build_dir / cfg.output.final_model_filename}")
    print(f"Plots        : {plots_dir}")
    print(f"Summary      : {build_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
