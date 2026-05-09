"""Shared evaluation, metrics, and diagnostic plots for attenuation-curve models.

Used by train_xgb_v6 (Scenario 1 — global n) and train_xgb_v7 (Scenario 2 —
per-event n).

All models are evaluated at the **sensor level**: predicted event parameters
(c_i and optionally n_i) are used to reconstruct PGV at the original
accelerometer distances, then compared to measured PGV.

Primary metrics are in log-space (to match the loss function used during
attenuation-curve fitting).  Normal-space (mm/s) metrics are also reported
for interpretability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def compute_metrics_log(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute RMSE, MAE, R² in log-space (inputs already log-transformed)."""
    return {
        "rmse_log": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_log": float(mean_absolute_error(y_true, y_pred)),
        "r2_log": float(r2_score(y_true, y_pred)),
    }


def compute_metrics_linear(
    y_true_mms: np.ndarray,
    y_pred_mms: np.ndarray,
) -> Dict[str, float]:
    """Compute RMSE, MAE, R², MAPE in normal (mm/s) space."""
    rmse = float(np.sqrt(mean_squared_error(y_true_mms, y_pred_mms)))
    mae = float(mean_absolute_error(y_true_mms, y_pred_mms))
    r2 = float(r2_score(y_true_mms, y_pred_mms))
    # Symmetric MAPE (avoids division by zero when y_true ≈ 0)
    denom = (np.abs(y_true_mms) + np.abs(y_pred_mms)) / 2.0
    smape = float(np.mean(np.abs(y_true_mms - y_pred_mms) / np.clip(denom, 1e-9, None)))
    return {"rmse_mms": rmse, "mae_mms": mae, "r2_mms": r2, "smape": smape}


def compute_all_metrics(
    y_true_mms: np.ndarray,
    y_pred_mms: np.ndarray,
) -> Dict[str, float]:
    """Compute both log-space and linear-space metrics from mm/s arrays."""
    eps = 1e-6
    log_true = np.log(np.clip(y_true_mms, eps, None))
    log_pred = np.log(np.clip(y_pred_mms, eps, None))
    metrics = compute_metrics_log(log_true, log_pred)
    metrics.update(compute_metrics_linear(y_true_mms, y_pred_mms))
    return metrics


# ---------------------------------------------------------------------------
# Sensor-level reconstruction and evaluation
# ---------------------------------------------------------------------------


def reconstruct_and_evaluate_global(
    df_sensor: pd.DataFrame,
    c_pred_series: pd.Series,
    n_global: float,
    r0: float = 10.0,
    event_col: str = "event_id",
    distance_col: str = "acc_distance_to_track_m",
    pgv_col: str = "target_pgv_z_mms",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Reconstruct PGV (Scenario 1) at each sensor and compute metrics.

    Parameters
    ----------
    df_sensor
        Sensor-level DataFrame filtered to the evaluation set.
    c_pred_series
        Predicted c_i values indexed by *event_col* values.
    n_global
        Global attenuation exponent.
    r0, event_col, distance_col, pgv_col
        Column names and reference distance.

    Returns
    -------
    (y_true, y_pred, metrics_dict)  — arrays in mm/s, dict of all metrics.
    """
    mask = (df_sensor[pgv_col] > 0) & (df_sensor[distance_col] > 0)
    df_eval = df_sensor[mask].copy()
    df_eval = df_eval[df_eval[event_col].isin(c_pred_series.index)]

    c_vals = c_pred_series.loc[df_eval[event_col]].to_numpy(dtype=np.float64)
    r_vals = np.clip(df_eval[distance_col].to_numpy(dtype=np.float64), 1e-3, None)
    y_pred = np.exp(c_vals - n_global * np.log(r_vals / r0))
    y_true = df_eval[pgv_col].to_numpy(dtype=np.float64)

    return y_true, y_pred, compute_all_metrics(y_true, y_pred)


def reconstruct_and_evaluate_event_specific(
    df_sensor: pd.DataFrame,
    c_pred_series: pd.Series,
    n_pred_series: pd.Series,
    r0: float = 10.0,
    event_col: str = "event_id",
    distance_col: str = "acc_distance_to_track_m",
    pgv_col: str = "target_pgv_z_mms",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Reconstruct PGV (Scenario 2) at each sensor and compute metrics.

    Parameters
    ----------
    df_sensor
        Sensor-level DataFrame filtered to the evaluation set.
    c_pred_series, n_pred_series
        Predicted c_i / n_i values indexed by *event_col* values.
    """
    shared_events = c_pred_series.index.intersection(n_pred_series.index)
    mask = (
        (df_sensor[pgv_col] > 0)
        & (df_sensor[distance_col] > 0)
        & df_sensor[event_col].isin(shared_events)
    )
    df_eval = df_sensor[mask].copy()

    c_vals = c_pred_series.loc[df_eval[event_col]].to_numpy(dtype=np.float64)
    n_vals = n_pred_series.loc[df_eval[event_col]].to_numpy(dtype=np.float64)
    r_vals = np.clip(df_eval[distance_col].to_numpy(dtype=np.float64), 1e-3, None)
    y_pred = np.exp(c_vals - n_vals * np.log(r_vals / r0))
    y_true = df_eval[pgv_col].to_numpy(dtype=np.float64)

    return y_true, y_pred, compute_all_metrics(y_true, y_pred)


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------


def plot_measured_vs_predicted_log_log(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    title: str,
    out_path: Path,
) -> None:
    """Log-log scatter of measured vs predicted PGV with 1:1 and ±factor-2 lines."""
    fig, ax = plt.subplots(figsize=(6, 6))

    lim_lo = max(min(y_true.min(), y_pred.min()) * 0.5, 1e-3)
    lim_hi = max(y_true.max(), y_pred.max()) * 2.0

    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color="steelblue", label="samples")
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1.5, label="1:1")
    ax.plot([lim_lo, lim_hi], [lim_lo * 2, lim_hi * 2], "r:", lw=1, label="×2 / ÷2")
    ax.plot([lim_lo, lim_hi], [lim_lo / 2, lim_hi / 2], "r:", lw=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Measured PGV [mm/s]")
    ax.set_ylabel("Predicted PGV [mm/s]")
    ax.set_title(
        f"{title}\n"
        f"RMSE={metrics['rmse_mms']:.3f} mm/s  "
        f"RMSE(log)={metrics['rmse_log']:.3f}  "
        f"R²={metrics['r2_mms']:.3f}"
    )
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals_vs_distance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    distances: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Residuals (log-space) vs sensor distance to track."""
    eps = 1e-6
    res = np.log(np.clip(y_pred, eps, None)) - np.log(np.clip(y_true, eps, None))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(distances, res, alpha=0.3, s=8, color="steelblue")
    ax.axhline(0, color="k", lw=1.2, ls="--")
    ax.axhline(np.log(2), color="r", lw=0.8, ls=":", label="±log(2) = ±factor 2")
    ax.axhline(-np.log(2), color="r", lw=0.8, ls=":")
    ax.set_xlabel("Distance to track [m]")
    ax.set_ylabel("Residual: log(pred/true)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Residuals (log-space) vs predicted PGV — checks for heteroscedasticity."""
    eps = 1e-6
    res = np.log(np.clip(y_pred, eps, None)) - np.log(np.clip(y_true, eps, None))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred, res, alpha=0.3, s=8, color="darkorange")
    ax.axhline(0, color="k", lw=1.2, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Predicted PGV [mm/s]")
    ax.set_ylabel("Residual: log(pred/true)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ci_distribution(
    ci_values: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Histogram of fitted c_i (log-intensity at reference distance)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        ci_values[np.isfinite(ci_values)],
        bins=40,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("c_i  =  log(PGV at r0)")
    ax.set_ylabel("Number of events")
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ni_distribution(
    ni_values: np.ndarray,
    n_global: Optional[float] = None,
    title: str = "",
    out_path: Optional[Path] = None,
) -> None:
    """Histogram of per-event n_i with optional global-n reference line."""
    fig, ax = plt.subplots(figsize=(6, 4))
    finite = ni_values[np.isfinite(ni_values)]
    ax.hist(finite, bins=40, color="darkorange", edgecolor="white", linewidth=0.5)
    if n_global is not None:
        ax.axvline(
            n_global, color="k", lw=1.5, ls="--", label=f"n_global = {n_global:.3f}"
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("n_i  (attenuation exponent)")
    ax.set_ylabel("Number of events")
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_attenuation_curves_sample(
    df_sensor: pd.DataFrame,
    event_params: pd.DataFrame,
    n_model: float,
    r0: float,
    n_samples: int = 9,
    scenario_label: str = "global n",
    out_path: Optional[Path] = None,
    event_col: str = "event_id",
    distance_col: str = "acc_distance_to_track_m",
    pgv_col: str = "target_pgv_z_mms",
) -> None:
    """Plot measured data + fitted attenuation curve for a random sample of events.

    Parameters
    ----------
    event_params
        DataFrame with at least columns [event_id, c_i].
        For scenario 2 it should also have [n_i].
    n_model
        Scalar exponent to use when ``n_i`` is not in *event_params*
        (i.e. Scenario 1).
    """
    rng = np.random.default_rng(42)
    available = event_params[event_col].unique()
    sample_ids = rng.choice(
        available, size=min(n_samples, len(available)), replace=False
    )

    nrows = int(np.ceil(n_samples / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(12, nrows * 3.5))
    axes = axes.flatten()

    r_curve = np.logspace(np.log10(1.0), np.log10(30.0), 80)

    for ax, eid in zip(axes, sample_ids):
        row = event_params[event_params[event_col] == eid].iloc[0]
        c_i = float(row["c_i"])
        n_i = (
            float(row["n_i"])
            if "n_i" in row.index and np.isfinite(row["n_i"])
            else n_model
        )

        # Measured sensor points
        ev_data = df_sensor[df_sensor[event_col] == eid]
        ev_data = ev_data[(ev_data[pgv_col] > 0) & (ev_data[distance_col] > 0)]
        ax.scatter(
            ev_data[distance_col],
            ev_data[pgv_col],
            color="steelblue",
            s=40,
            zorder=5,
            label="measured",
        )

        # Fitted curve
        pgv_curve = np.exp(c_i - n_i * np.log(r_curve / r0))
        ax.plot(r_curve, pgv_curve, "r-", lw=1.5, label=f"n={n_i:.2f}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("r [m]", fontsize=8)
        ax.set_ylabel("PGV [mm/s]", fontsize=8)
        ax.set_title(str(eid)[:20], fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", ls=":", alpha=0.4)

    # Hide unused subplots
    for ax in axes[len(sample_ids) :]:
        ax.set_visible(False)

    fig.suptitle(f"Sample attenuation curves — {scenario_label}", fontsize=11)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mean_attenuation_curve(
    df_sensor: pd.DataFrame,
    n_global: float,
    r0: float,
    out_path: Path,
    event_col: str = "event_id",
    distance_col: str = "acc_distance_to_track_m",
    pgv_col: str = "target_pgv_z_mms",
) -> None:
    """Plot the mean attenuation curve with all measured data points."""
    mask = (df_sensor[pgv_col] > 0) & (df_sensor[distance_col] > 0)
    dv = df_sensor[mask].copy()

    # Normalise each event to reference distance
    event_means = (
        dv.groupby(event_col)
        .apply(
            lambda g: np.exp(
                np.log(g[pgv_col]).mean()
                + n_global * np.log((g[distance_col] / r0)).mean()
            )
        )
        .rename("exp_ci")
    )
    dv["_norm_pgv"] = dv[pgv_col] / event_means.loc[dv[event_col]].values

    r_curve = np.logspace(np.log10(1.0), np.log10(30.0), 100)
    pgv_mean_curve = (r_curve / r0) ** (-n_global)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        dv[distance_col],
        dv["_norm_pgv"],
        alpha=0.2,
        s=6,
        color="steelblue",
        label="normalised data",
    )
    ax.plot(r_curve, pgv_mean_curve, "r-", lw=2, label=f"mean curve n={n_global:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distance to track [m]")
    ax.set_ylabel("Normalised PGV (–)")
    ax.set_title(f"Mean attenuation curve  (r0 = {r0:.0f} m)")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
