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
      feat_log1p_distance  = log(1 + acc_distance_to_track_m)
      feat_inv_distance_sq = 1 / (acc_distance_to_track_m^2 + 1)
    """
    X = X.copy()
    if fe_cfg.add_geometry_features and "acc_distance_to_track_m" in X.columns:
        d = X["acc_distance_to_track_m"].clip(lower=0.0)
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


def train_groupkfold(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_cfg: Any,
    n_folds: int,
    verbose: bool,
) -> Tuple[np.ndarray, List[Dict], List[int]]:
    """Run GroupKFold cross-validation.

    Returns:
        oof_preds     — out-of-fold predictions aligned with X/y index
        fold_history  — list of dicts with per-fold metrics and training history
        best_rounds   — best n_estimators per fold (from early stopping)
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
            verbosity=0,  # suppress per-round XGBoost output; we print our own
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            verbose=100 if verbose else False,
        )

        best_round = int(model.best_iteration) + 1
        best_rounds.append(best_round)

        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds

        train_preds = model.predict(X_tr)
        fold_metrics = {
            "fold": fold_idx,
            "best_round": best_round,
            "train": _metrics(y_tr.values, train_preds),
            "val": _metrics(y_val.values, val_preds),
            "n_train_rows": int(len(y_tr)),
            "n_val_rows": int(len(y_val)),
            "n_train_events": int(groups.iloc[train_idx].nunique()),
            "n_val_events": int(groups.iloc[val_idx].nunique()),
        }
        fold_history.append(fold_metrics)

        if verbose:
            print(
                f"  Fold {fold_idx}/{n_folds} | "
                f"best_round={best_round:>4} | "
                f"val RMSE={fold_metrics['val']['rmse']:.4f} | "
                f"val MAE={fold_metrics['val']['mae']:.4f} | "
                f"val R²={fold_metrics['val']['r2']:.4f}"
            )

    return oof_preds, fold_history, best_rounds


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
