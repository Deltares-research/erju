"""metrics.py
=============
Evaluation, metrics, bootstrap CIs and plots for M0/M1/M2, reusing
src.ml.spectral.eval_spectral's metric primitives (read-only import).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.ml.spectral.eval_spectral import _mae, _pearson, _r2, _rmse, _spearman, total_rms_from_db

from src.ml.linec_multisensor.data import SENSORS


@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, trues, rs, tracks, ev_idx = [], [], [], [], []
    for wf, meta, nv, tgt, r, track, evi in loader:
        wf_d, meta_d, nv_d, r_d, track_d = (t.to(device) for t in (wf, meta, nv, r, track))
        pred = model(wf_d, meta_d, nv_d, r_d, track_d).cpu().numpy()
        preds.append(pred)
        trues.append(tgt.numpy())
        rs.append(r.numpy())
        tracks.append(track.numpy())
        ev_idx.append(evi.numpy())
    return (np.concatenate(preds), np.concatenate(trues), np.concatenate(rs),
            np.concatenate(tracks), np.concatenate(ev_idx))


def _per_sensor_total_rms(x_db: np.ndarray) -> np.ndarray:
    """(N,5,19) dB -> (N,5) total RMS mm/s per sensor."""
    return np.stack([total_rms_from_db(x_db[:, j, :]) for j in range(x_db.shape[1])], axis=1)


def compute_full_metrics(pred_db: np.ndarray, true_db: np.ndarray, r_m: np.ndarray,
                          band_nominal: np.ndarray, strong_threshold_mms: Optional[float] = None) -> dict:
    """pred_db/true_db: (N,5,19). r_m: (N,5)."""
    n_events, n_sensors, n_bands = true_db.shape
    resid = true_db - pred_db

    macro_rmse = float(_rmse(true_db.reshape(-1), pred_db.reshape(-1)))
    macro_mae = float(_mae(true_db.reshape(-1), pred_db.reshape(-1)))
    macro_r2 = float(_r2(true_db.reshape(-1), pred_db.reshape(-1)))

    per_band = [float(_rmse(true_db[:, :, f].reshape(-1), pred_db[:, :, f].reshape(-1))) for f in range(n_bands)]
    per_sensor = {
        s: {
            "rmse_db": float(_rmse(true_db[:, j, :].reshape(-1), pred_db[:, j, :].reshape(-1))),
            "mae_db": float(_mae(true_db[:, j, :].reshape(-1), pred_db[:, j, :].reshape(-1))),
            "r2": float(_r2(true_db[:, j, :].reshape(-1), pred_db[:, j, :].reshape(-1))),
        }
        for j, s in enumerate(SENSORS)
    }

    true_tot = _per_sensor_total_rms(true_db)  # (N,5)
    pred_tot = _per_sensor_total_rms(pred_db)
    total_rms = {
        "rmse_mms": float(_rmse(true_tot.reshape(-1), pred_tot.reshape(-1))),
        "pearson": float(_pearson(true_tot.reshape(-1), pred_tot.reshape(-1))),
        "spearman": float(_spearman(true_tot.reshape(-1), pred_tot.reshape(-1))),
        "r2": float(_r2(true_tot.reshape(-1), pred_tot.reshape(-1))),
    }

    true_level_db = 10.0 * np.log10(np.clip(np.sum(10.0 ** (true_db / 10.0), axis=-1), 1e-300, None))  # (N,5)
    pred_level_db = 10.0 * np.log10(np.clip(np.sum(10.0 ** (pred_db / 10.0), axis=-1), 1e-300, None))
    true_shape = true_db - true_level_db[..., None]
    pred_shape = pred_db - pred_level_db[..., None]
    shape_only_rmse_db = float(_rmse(true_shape.reshape(-1), pred_shape.reshape(-1)))

    resid_flat = resid.reshape(-1)
    r_rep = np.repeat(r_m.reshape(-1), n_bands)
    amp_rep = np.repeat(np.repeat(true_tot, n_bands, axis=1).reshape(-1), 1)
    residual_vs_distance_pearson = float(_pearson(resid_flat, r_rep))
    residual_vs_amplitude_pearson = float(_pearson(resid_flat, amp_rep))

    # Amplitude-compression diagnostic: slope/intercept of predicted vs measured total RMS
    slope, intercept = np.polyfit(true_tot.reshape(-1), pred_tot.reshape(-1), 1)
    amplitude_compression = {
        "slope": float(slope), "intercept": float(intercept),
        "compression_pct": float(100.0 * (1.0 - slope)),
    }

    # Strong-event metrics: top-quartile events by measured mean total RMS across sensors
    ev_mean_tot = true_tot.mean(axis=1)
    thresh = strong_threshold_mms if strong_threshold_mms is not None else float(np.percentile(ev_mean_tot, 75))
    strong_mask = ev_mean_tot >= thresh
    n_strong = int(strong_mask.sum())
    strong_event_metrics = {
        "threshold_mms": float(thresh), "n_events": n_strong,
        "macro_rmse_db": float(_rmse(true_db[strong_mask].reshape(-1), pred_db[strong_mask].reshape(-1))) if n_strong else float("nan"),
        "total_rms_rmse_mms": float(_rmse(true_tot[strong_mask].reshape(-1), pred_tot[strong_mask].reshape(-1))) if n_strong else float("nan"),
    }

    return {
        "n_events": int(n_events), "macro_rmse_db": macro_rmse, "macro_mae_db": macro_mae, "macro_r2": macro_r2,
        "per_band_rmse_db": per_band, "per_sensor": per_sensor, "total_rms": total_rms,
        "shape_only_rmse_db": shape_only_rmse_db,
        "residual_vs_distance_pearson": residual_vs_distance_pearson,
        "residual_vs_amplitude_pearson": residual_vs_amplitude_pearson,
        "amplitude_compression": amplitude_compression,
        "strong_event_metrics": strong_event_metrics,
    }


# ── Bootstrap CIs (event-level resampling) ───────────────────────────────────

def bootstrap_macro_rmse_ci(pred_db: np.ndarray, true_db: np.ndarray, n_boot: int = 1000,
                             seed: int = 0, ci: float = 0.95) -> dict:
    rng = np.random.default_rng(seed)
    n = true_db.shape[0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = _rmse(true_db[idx].reshape(-1), pred_db[idx].reshape(-1))
    lo, hi = np.percentile(vals, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"mean": float(vals.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n_boot": n_boot}


def paired_bootstrap_diff_ci(pred_a: np.ndarray, pred_b: np.ndarray, true_db: np.ndarray,
                              n_boot: int = 1000, seed: int = 0, ci: float = 0.95) -> dict:
    """Paired event-level bootstrap CI for (rmse_a - rmse_b), same resample for both models."""
    rng = np.random.default_rng(seed)
    n = true_db.shape[0]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rmse_a = _rmse(true_db[idx].reshape(-1), pred_a[idx].reshape(-1))
        rmse_b = _rmse(true_db[idx].reshape(-1), pred_b[idx].reshape(-1))
        diffs[b] = rmse_a - rmse_b
    lo, hi = np.percentile(diffs, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"mean_diff_db": float(diffs.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_boot": n_boot, "significant": bool(lo > 0 or hi < 0)}


# ── Plots ─────────────────────────────────────────────────────────────────────

def _save_fig(fig, path: Path) -> None:
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_measured_vs_predicted(true_db: np.ndarray, pred_db: np.ndarray, out_dir: Path, split_name: str) -> None:
    true_tot = _per_sensor_total_rms(true_db)
    pred_tot = _per_sensor_total_rms(pred_db)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(SENSORS)))
    for j, s in enumerate(SENSORS):
        ax.scatter(true_tot[:, j], pred_tot[:, j], s=12, alpha=0.6, color=colors[j], label=s, edgecolors="none")
    lo = float(min(true_tot.min(), pred_tot.min())); hi = float(max(true_tot.max(), pred_tot.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, "k--", lw=1, label="1:1")
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal", adjustable="box")
    r2 = _r2(true_tot.reshape(-1), pred_tot.reshape(-1))
    rmse = _rmse(true_tot.reshape(-1), pred_tot.reshape(-1))
    ax.set_title(f"{split_name}  R2={r2:.3f}  RMSE={rmse:.4f} mm/s")
    ax.set_xlabel("measured total RMS (mm/s)"); ax.set_ylabel("predicted total RMS (mm/s)")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
    fig.tight_layout()
    _save_fig(fig, out_dir / f"measured_vs_predicted_{split_name}.png")


def plot_per_band_rmse(band_nominal: np.ndarray, per_band_rmse: List[float], out_dir: Path, split_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"{hz:g}" for hz in band_nominal], per_band_rmse, color="steelblue")
    ax.set_xlabel("nominal band (Hz)"); ax.set_ylabel("RMSE (dB)")
    ax.set_title(f"Per-band macro RMSE ({split_name})")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=7)
    fig.tight_layout()
    _save_fig(fig, out_dir / f"per_band_rmse_{split_name}.png")


def plot_training_history(history: List[Dict], out_dir: Path) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train_loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("composite loss"); axes[0].legend()
    axes[1].plot(epochs, [h["val_macro_rmse_db"] for h in history], color="darkorange")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val macro RMSE (dB)")
    fig.tight_layout()
    _save_fig(fig, out_dir / "training_history.png")


def plot_amplitude_compression(true_db: np.ndarray, pred_db: np.ndarray, out_dir: Path, split_name: str) -> None:
    true_tot = _per_sensor_total_rms(true_db).reshape(-1)
    pred_tot = _per_sensor_total_rms(pred_db).reshape(-1)
    slope, intercept = np.polyfit(true_tot, pred_tot, 1)
    fig, ax = plt.subplots(figsize=(5.5, 5)); ax.scatter(true_tot, pred_tot, s=10, alpha=0.4)
    xs = np.linspace(true_tot.min(), true_tot.max(), 100)
    ax.plot(xs, slope * xs + intercept, "r-", label=f"fit slope={slope:.3f}")
    ax.plot(xs, xs, "k--", lw=1, label="1:1")
    ax.set_xlabel("measured total RMS (mm/s)"); ax.set_ylabel("predicted total RMS (mm/s)")
    ax.set_title(f"Amplitude compression ({split_name})"); ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out_dir / f"amplitude_compression_{split_name}.png")


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj
