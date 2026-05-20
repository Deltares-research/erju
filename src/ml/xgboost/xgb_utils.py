"""Utilities for XGBoost v1 training — split, train, evaluate, save artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
import xgboost as xgb

# ---------------------------------------------------------------------------
# Build folder
# ---------------------------------------------------------------------------


def create_build_folder(output_root: Path, version_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_root / f"{version_name}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------


def load_dataset(parquet_path: Path) -> pd.DataFrame:
    """Load the Parquet dataset and report basic shape."""
    df = pd.read_parquet(parquet_path)
    return df


def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    identifier_cols: List[str],
    string_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split DataFrame into X, y, and groups (event_id).

    Drops identifier columns and raw string columns.
    Returns:
        X      — feature DataFrame (numeric + int-encoded categoricals)
        y      — target Series
        groups — event_id Series aligned with X/y index
    """
    groups = df["event_id"].copy()
    y = df[target_col].copy()

    drop_cols = set(identifier_cols) | set(string_cols) | {target_col}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return X, y, groups


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(X: pd.DataFrame, fe_cfg: Any) -> pd.DataFrame:
    """Apply training-time feature engineering to the feature matrix.

    All derived features are computed from columns already in the Parquet — no
    raw NetCDF data is needed.  Every transformation is controlled by flags in
    FeatureEngineeringConfig, which is saved to config_snapshot.json so the
    exact transformations applied in each experiment are fully traceable.

    Currently adds (when add_geometry_features=True):
      feat_log1p_distance  = log(1 + effective_distance_to_active_track_m)
      feat_inv_distance_sq = 1 / (effective_distance_to_active_track_m^2 + 1)
    Falls back to acc_distance_to_track_m if the effective column is absent
    (e.g. when loading an old parquet v2 build).
    """
    X = X.copy()
    if fe_cfg.add_geometry_features:
        dist_col = (
            "effective_distance_to_active_track_m"
            if "effective_distance_to_active_track_m" in X.columns
            else "acc_distance_to_track_m"
        )
        if dist_col in X.columns:
            d = X[dist_col].clip(lower=0.0)
            X["feat_log1p_distance"] = np.log1p(d)
            X["feat_inv_distance_sq"] = 1.0 / (d**2 + 1.0)
    return X


# ---------------------------------------------------------------------------
# Event-level train / test split
# ---------------------------------------------------------------------------


def make_event_level_test_split(
    df: pd.DataFrame,
    group_col: str,
    test_fraction: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a fixed fraction of events as the final test set.

    Sampling is done at the event level so no event appears in both splits.
    Returns (train_val_df, test_df).
    """
    rng = np.random.default_rng(random_seed)
    all_events = df[group_col].unique()
    n_test = max(1, int(len(all_events) * test_fraction))
    test_events = set(rng.choice(all_events, size=n_test, replace=False))

    mask_test = df[group_col].isin(test_events)
    return df[~mask_test].reset_index(drop=True), df[mask_test].reset_index(drop=True)


def build_split_manifest(
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str,
) -> Dict[str, Any]:
    return {
        "train_val_event_ids": sorted(train_val_df[group_col].unique().tolist()),
        "test_event_ids": sorted(test_df[group_col].unique().tolist()),
        "n_train_val_events": int(train_val_df[group_col].nunique()),
        "n_test_events": int(test_df[group_col].nunique()),
        "n_train_val_rows": int(len(train_val_df)),
        "n_test_rows": int(len(test_df)),
    }


# ---------------------------------------------------------------------------
# GroupKFold cross-validation training
# ---------------------------------------------------------------------------


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _train_one_fold(
    fold_idx: int,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    groups_tr: pd.Series,
    groups_val: pd.Series,
    model_cfg: Any,
    n_folds: int,
    log_transform: bool,
    verbose: bool,
) -> Tuple[xgb.XGBRegressor, np.ndarray, Dict]:
    """Train XGBoost on one fold and return (model, val_preds, fold_record).

    eval_set passes both the train subset and the val subset so that
    model.evals_result() gives per-round curves for both.
    Metrics inside XGBoost are computed in whatever space y_tr/y_val are in
    (log-space when log_transform=True).  Original-space metrics are computed
    separately after inverse-transforming predictions.
    """
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
    val_preds_model = model.predict(X_val)
    train_preds_model = model.predict(X_tr)

    # Per-round curves from XGBoost internal eval (in model-space, i.e. log-space
    # when log_transform=True).  Keys depend on eval_metric e.g. "rmse" or "mae".
    evals = model.evals_result()
    # evals_result returns {"validation_0": {...}, "validation_1": {...}}
    train_curves_raw: Dict[str, List[float]] = evals.get("validation_0", {})
    val_curves_raw: Dict[str, List[float]] = evals.get("validation_1", {})

    # Back-transform predictions to original mm/s space for reported metrics
    if log_transform:
        val_preds_mms = np.expm1(val_preds_model)
        train_preds_mms = np.expm1(train_preds_model)
        y_val_mms = np.expm1(y_val.values)
        y_tr_mms = np.expm1(y_tr.values)
    else:
        val_preds_mms = val_preds_model
        train_preds_mms = train_preds_model
        y_val_mms = y_val.values
        y_tr_mms = y_tr.values

    val_metrics_mms = _metrics(y_val_mms, val_preds_mms)
    train_metrics_mms = _metrics(y_tr_mms, train_preds_mms)

    # Overfit detection: val RMSE at best round vs. train RMSE at best round
    # (both in model-space so comparison is fair across log/linear configs)
    _metric_key = model_cfg.eval_metric  # e.g. "rmse"
    _train_at_best = train_curves_raw.get(_metric_key, [np.nan])[model.best_iteration]
    _val_at_best = val_curves_raw.get(_metric_key, [np.nan])[model.best_iteration]
    # Overfit flag: train metric is substantially better than val metric
    overfit_ratio = _val_at_best / _train_at_best if _train_at_best > 0 else np.nan
    overfit_flag = bool(overfit_ratio > 1.5) if np.isfinite(overfit_ratio) else False

    fold_record: Dict = {
        "fold": fold_idx,
        "best_round": best_round,
        "best_val_score_model_space": float(_val_at_best),
        "final_train_score_model_space": float(_train_at_best),
        "overfit_flag": overfit_flag,
        "overfit_ratio_val_over_train": (
            float(overfit_ratio) if np.isfinite(overfit_ratio) else None
        ),
        # Final-round metrics in original mm/s space
        "train_mms": train_metrics_mms,
        "val_mms": val_metrics_mms,
        # Legacy keys kept for backward compat with existing callers
        "train": train_metrics_mms,
        "val": val_metrics_mms,
        "n_train_rows": int(len(y_tr)),
        "n_val_rows": int(len(y_val)),
        "n_train_events": int(groups_tr.nunique()),
        "n_val_events": int(groups_val.nunique()),
        # Per-round curves (model-space).  Label clearly to avoid confusion.
        "curves_model_space": {
            "label": "log-space (log1p)" if log_transform else "original mm/s",
            "train": {
                k: [float(v) for v in vals] for k, vals in train_curves_raw.items()
            },
            "val": {k: [float(v) for v in vals] for k, vals in val_curves_raw.items()},
        },
    }

    if verbose:
        overfit_str = " *** OVERFIT ***" if overfit_flag else ""
        print(
            f"  Fold {fold_idx}/{n_folds} | best_round={best_round:>4} | "
            f"val RMSE={val_metrics_mms['rmse']:.4f} mm/s | "
            f"val MAE={val_metrics_mms['mae']:.4f} mm/s | "
            f"val R²={val_metrics_mms['r2']:.4f}{overfit_str}"
        )

    return model, val_preds_model, fold_record


def train_groupkfold(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_cfg: Any,
    n_folds: int,
    verbose: bool,
    log_transform: bool = False,
) -> Tuple[np.ndarray, List[Dict], List[int]]:
    """Run GroupKFold cross-validation.

    Returns:
        oof_preds     — out-of-fold predictions in model-space (log-space when
                        log_transform=True); back-transform with expm1 for mm/s
        fold_history  — list of fold record dicts (see _train_one_fold)
        best_rounds   — best n_estimators per fold
    """
    oof_preds = np.full(len(y), np.nan, dtype=np.float64)
    fold_history: List[Dict] = []
    best_rounds: List[int] = []

    kf = GroupKFold(n_splits=n_folds)

    for fold_idx, (train_idx, val_idx) in enumerate(
        kf.split(X, y, groups=groups), start=1
    ):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        _, val_preds_model, fold_record = _train_one_fold(
            fold_idx=fold_idx,
            X_tr=X_tr,
            y_tr=y_tr,
            X_val=X_val,
            y_val=y_val,
            groups_tr=groups.iloc[train_idx],
            groups_val=groups.iloc[val_idx],
            model_cfg=model_cfg,
            n_folds=n_folds,
            log_transform=log_transform,
            verbose=verbose,
        )

        best_rounds.append(fold_record["best_round"])
        oof_preds[val_idx] = val_preds_model
        fold_history.append(fold_record)

    return oof_preds, fold_history, best_rounds


def summarize_fold_cv(fold_history: List[Dict]) -> Dict:
    """Return a compact summary dict of CV results across all folds."""
    best_rounds = [f["best_round"] for f in fold_history]
    val_rmse = [f["val_mms"]["rmse"] for f in fold_history]
    val_mae = [f["val_mms"]["mae"] for f in fold_history]
    val_r2 = [f["val_mms"]["r2"] for f in fold_history]
    overfit_folds = [f["fold"] for f in fold_history if f.get("overfit_flag")]

    return {
        "n_folds": len(fold_history),
        "per_fold": [
            {
                "fold": f["fold"],
                "best_round": f["best_round"],
                "val_rmse_mms": f["val_mms"]["rmse"],
                "val_mae_mms": f["val_mms"]["mae"],
                "val_r2": f["val_mms"]["r2"],
                "overfit_flag": f.get("overfit_flag", False),
                "overfit_ratio": f.get("overfit_ratio_val_over_train"),
            }
            for f in fold_history
        ],
        "best_round_mean": float(np.mean(best_rounds)),
        "best_round_std": float(np.std(best_rounds)),
        "val_rmse_mean": float(np.mean(val_rmse)),
        "val_rmse_std": float(np.std(val_rmse)),
        "val_mae_mean": float(np.mean(val_mae)),
        "val_mae_std": float(np.std(val_mae)),
        "val_r2_mean": float(np.mean(val_r2)),
        "val_r2_std": float(np.std(val_r2)),
        "overfit_folds": overfit_folds,
    }


# ---------------------------------------------------------------------------
# Final model training
# ---------------------------------------------------------------------------


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_cfg: Any,
    n_rounds: int,
) -> xgb.XGBRegressor:
    """Train a final model on all train+val data using a fixed round count.

    n_rounds should be the mean best_round from OOF cross-validation.
    No early stopping — the round count is fixed from OOF.
    """
    model = xgb.XGBRegressor(
        objective=model_cfg.objective,
        tree_method=model_cfg.tree_method,
        max_depth=model_cfg.max_depth,
        learning_rate=model_cfg.learning_rate,
        n_estimators=n_rounds,
        subsample=model_cfg.subsample,
        colsample_bytree=model_cfg.colsample_bytree,
        min_child_weight=model_cfg.min_child_weight,
        reg_alpha=model_cfg.reg_alpha,
        reg_lambda=model_cfg.reg_lambda,
        random_state=model_cfg.random_state,
        verbosity=1,
    )
    model.fit(X, y, verbose=100)
    return model


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def extract_feature_importance(
    model: xgb.XGBRegressor,
    feature_names: List[str],
) -> pd.DataFrame:
    scores_gain = model.get_booster().get_score(importance_type="gain")
    scores_weight = model.get_booster().get_score(importance_type="weight")
    scores_cover = model.get_booster().get_score(importance_type="cover")

    df = pd.DataFrame({"feature": feature_names})
    df["gain"] = df["feature"].map(scores_gain).fillna(0.0)
    df["weight"] = df["feature"].map(scores_weight).fillna(0.0)
    df["cover"] = df["feature"].map(scores_cover).fillna(0.0)
    df = df.sort_values("gain", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_model(model: xgb.XGBRegressor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def save_oof_predictions(
    train_val_df: pd.DataFrame,
    oof_preds: np.ndarray,
    target_col: str,
    output_path: Path,
) -> None:
    out = pd.DataFrame(
        {
            "event_id": train_val_df["event_id"].values,
            "sensor_id": train_val_df["sensor_id"].values,
            "y_true": train_val_df[target_col].values,
            "y_pred_oof": oof_preds,
            "residual": train_val_df[target_col].values - oof_preds,
        }
    )
    out.to_parquet(output_path, index=False)


def write_build_log(log_path: Path, lines: List[str]) -> None:
    with open(log_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


# ---------------------------------------------------------------------------
# Diagnostics plots
# ---------------------------------------------------------------------------


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    lim = (0, max(y_true.max(), y_pred.max()) * 1.05)
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
    ax.plot(lim, lim, "r--", lw=1, label="perfect")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual PGV_z (mm/s)")
    ax.set_ylabel("Predicted PGV_z (mm/s)")
    ax.set_title(title)
    ax.legend()
    _savefig(fig, output_path)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color="steelblue")
    axes[0].axhline(0, color="red", lw=1, linestyle="--")
    axes[0].set_xlabel("Predicted PGV_z (mm/s)")
    axes[0].set_ylabel("Residual (mm/s)")
    axes[0].set_title(f"{title} — Residual vs Predicted")

    axes[1].hist(residuals, bins=50, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", lw=1, linestyle="--")
    axes[1].set_xlabel("Residual (mm/s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{title} — Residual Distribution")

    fig.tight_layout()
    _savefig(fig, output_path)


def plot_training_curves(
    fold_history: List[Dict],
    output_path: Path,
) -> None:
    """Plot val RMSE per fold as a bar/line summary."""
    folds = [f["fold"] for f in fold_history]
    val_rmse = [f["val"]["rmse"] for f in fold_history]
    train_rmse = [f["train"]["rmse"] for f in fold_history]
    best_rounds = [f["best_round"] for f in fold_history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(folds, val_rmse, color="steelblue", label="Val RMSE", alpha=0.7)
    axes[0].bar(folds, train_rmse, color="orange", label="Train RMSE", alpha=0.5)
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("RMSE (mm/s)")
    axes[0].set_title("Per-fold RMSE (train vs val)")
    axes[0].legend()
    axes[0].set_xticks(folds)

    axes[1].bar(folds, best_rounds, color="steelblue", alpha=0.7)
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Best round (early stopping)")
    axes[1].set_title("Early stopping round per fold")
    axes[1].set_xticks(folds)

    fig.tight_layout()
    _savefig(fig, output_path)


def plot_fold_learning_curves(
    fold_history: List[Dict],
    output_dir: Path,
    metric: str = "rmse",
) -> None:
    """Save one learning-curve PNG per fold to output_dir/fold_curves/.

    Curves are in model-space (log-space when log_transform_target=True).
    The label stored in fold_record["curves_model_space"]["label"] is shown
    in the y-axis so it is always clear which space the plot is in.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold_record in fold_history:
        fold_idx = fold_record["fold"]
        curves = fold_record.get("curves_model_space", {})
        space_label = curves.get("label", "model-space")
        train_vals = curves.get("train", {}).get(metric, [])
        val_vals = curves.get("val", {}).get(metric, [])

        if not train_vals and not val_vals:
            continue

        rounds = list(range(1, len(val_vals) + 1))
        best_round = fold_record["best_round"]
        overfit = fold_record.get("overfit_flag", False)
        val_rmse_mms = fold_record["val_mms"]["rmse"]

        fig, ax = plt.subplots(figsize=(9, 4))
        if train_vals:
            ax.plot(
                rounds[: len(train_vals)],
                train_vals,
                lw=1.2,
                color="#E08040",
                alpha=0.8,
                label=f"Train {metric}",
            )
        if val_vals:
            ax.plot(rounds, val_vals, lw=1.5, color="#4878CF", label=f"Val {metric}")
        ax.axvline(
            best_round,
            color="green",
            lw=1.0,
            ls="--",
            label=f"Best round = {best_round}",
        )

        overfit_str = "  *** OVERFIT DETECTED ***" if overfit else ""
        ax.set_xlabel("Boosting round")
        ax.set_ylabel(f"{metric.upper()} ({space_label})")
        ax.set_title(
            f"Fold {fold_idx} learning curve  |  "
            f"best_round={best_round}  |  val RMSE={val_rmse_mms:.4f} mm/s{overfit_str}"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _savefig(fig, output_dir / f"fold_{fold_idx:02d}.png")


def plot_aggregated_cv_curves(
    fold_history: List[Dict],
    output_path: Path,
    metric: str = "rmse",
) -> None:
    """Plot mean ± std learning curves aggregated across all folds.

    Folds are aligned by boosting round; shorter folds are ignored beyond
    their last round (no padding/extrapolation).  The uncertainty band
    therefore narrows at high round numbers where fewer folds contribute.
    Curves are in model-space; labelled accordingly.
    """
    train_arrays: List[np.ndarray] = []
    val_arrays: List[np.ndarray] = []
    space_label = "model-space"

    for fold_record in fold_history:
        curves = fold_record.get("curves_model_space", {})
        space_label = curves.get("label", "model-space")
        t = curves.get("train", {}).get(metric, [])
        v = curves.get("val", {}).get(metric, [])
        if t:
            train_arrays.append(np.array(t))
        if v:
            val_arrays.append(np.array(v))

    def _band(arrays: List[np.ndarray]):
        max_len = max(len(a) for a in arrays)
        mat = np.full((len(arrays), max_len), np.nan)
        for i, a in enumerate(arrays):
            mat[i, : len(a)] = a
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        count = np.sum(~np.isnan(mat), axis=0)
        return mean, std, count

    fig, ax = plt.subplots(figsize=(10, 5))

    if train_arrays:
        t_mean, t_std, _ = _band(train_arrays)
        rounds = np.arange(1, len(t_mean) + 1)
        ax.plot(rounds, t_mean, color="#E08040", lw=1.5, label=f"Train {metric} mean")
        ax.fill_between(
            rounds, t_mean - t_std, t_mean + t_std, color="#E08040", alpha=0.20
        )

    if val_arrays:
        v_mean, v_std, v_count = _band(val_arrays)
        rounds = np.arange(1, len(v_mean) + 1)
        ax.plot(rounds, v_mean, color="#4878CF", lw=1.8, label=f"Val {metric} mean")
        ax.fill_between(
            rounds,
            v_mean - v_std,
            v_mean + v_std,
            color="#4878CF",
            alpha=0.20,
            label=f"Val {metric} ±1 std",
        )

    best_rounds = [f["best_round"] for f in fold_history]
    mean_best = float(np.mean(best_rounds))
    ax.axvline(
        mean_best,
        color="green",
        lw=1.2,
        ls="--",
        label=f"Mean best round = {mean_best:.0f}",
    )

    ax.set_xlabel("Boosting round")
    ax.set_ylabel(f"{metric.upper()} ({space_label})")
    ax.set_title(
        f"Aggregated CV learning curves  ({len(fold_history)} folds)  |  "
        f"mean best round = {mean_best:.0f} ± {float(np.std(best_rounds)):.1f}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_path)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int,
    output_path: Path,
) -> None:
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(top["feature"][::-1], top["gain"][::-1], color="steelblue")
    ax.set_xlabel("Gain")
    ax.set_title(f"Top {top_n} features by gain")
    fig.tight_layout()
    _savefig(fig, output_path)
