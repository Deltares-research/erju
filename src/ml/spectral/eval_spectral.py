"""eval_spectral.py
===================
Evaluation, metrics, long-format prediction output, and all 14 required plots
for the MP8 spectral and PGV experiments.

Usage
-----
    results = evaluate_split(model, loader, stats, events_df, cfg, device,
                             band_cols, split_name="val")
    save_eval_outputs(results, out_dir, cfg.name, split_name, history,
                      metadata_baseline_rmse_per_band)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from src.ml.spectral.config_spectral_v001 import (
    META_BASELINE_RMSE_PER_BAND, PRIMARY_NOMINALS, V_REF,
)
from src.ml.spectral.dataset_spectral import DataStats


def amp_ctx_for_eval(device: torch.device):
    """Return a no-op or bfloat16 autocast context for evaluation."""
    use_amp = (device.type == "cuda" and torch.cuda.is_bf16_supported())
    return torch.amp.autocast(device_type=device.type,
                               dtype=torch.bfloat16, enabled=use_amp)


# ── Inverse transforms ────────────────────────────────────────────────────────

def invert_spectral(pred_std: np.ndarray, stats: DataStats) -> np.ndarray:
    """Undo standardization → dB values."""
    return pred_std * stats.target_std + stats.target_mean


def invert_pgv(pred_std: np.ndarray, stats: DataStats) -> np.ndarray:
    """Undo standardization + log → mm/s."""
    log_pgv = pred_std * stats.target_std + stats.target_mean
    return np.exp(log_pgv)


def db_to_rms_mms(db: np.ndarray) -> np.ndarray:
    return V_REF * np.power(10.0, np.asarray(db) / 20.0)


def total_rms_from_db(band_db: np.ndarray) -> np.ndarray:
    """(n, 19) dB array → (n,) total RMS mm/s."""
    return np.sqrt(np.sum(db_to_rms_mms(band_db) ** 2, axis=1))


# ── Metric helpers ────────────────────────────────────────────────────────────

def _rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))
def _mae(y, p):  return float(mean_absolute_error(y, p))
def _r2(y, p):   return float(r2_score(y, p))
def _bias(y, p): return float(np.mean(p - y))
def _res_std(y, p): return float(np.std(p - y))
def _pearson(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(pearsonr(a, b)[0])
def _spearman(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(spearmanr(a, b)[0])


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
    is_multitask: bool,
    is_pgv:  bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect standardized predictions and targets.

    Returns
    -------
    preds_std    : (N, 19) or (N, 1) standardized predictions
    targets_std  : (N, 19) or (N, 1) standardized targets
    pgv_aux_std  : (N,) standardized log-PGV (always available)
    pgv_pred_std : (N,) pgv head predictions (for S5) or zeros
    ev_indices   : (N,) int
    """
    model.eval()
    ctx = amp_ctx_for_eval(device)
    all_preds, all_tgts, all_pgv_aux, all_pgv_pred, all_ev_idx = [], [], [], [], []

    for wf, meta, n_valid, tgt, pgv_aux, ev_idx in loader:
        wf, meta, n_valid = wf.to(device), meta.to(device), n_valid.to(device)
        with ctx:
            out = model(wf, meta, n_valid)

        if is_multitask:
            spec_p, pgv_p = out
            all_preds.append(spec_p.float().cpu().numpy())
            all_pgv_pred.append(pgv_p.float().cpu().numpy())
        elif is_pgv:
            all_preds.append(out.float().cpu().unsqueeze(-1).numpy())
            all_pgv_pred.append(out.float().cpu().numpy())
        else:
            all_preds.append(out.float().cpu().numpy())
            all_pgv_pred.append(np.zeros(wf.size(0), dtype=np.float32))

        all_tgts.append(tgt.float().cpu().numpy())
        all_pgv_aux.append(pgv_aux.float().cpu().numpy())
        all_ev_idx.append(ev_idx.cpu().numpy())

    preds_std   = np.concatenate(all_preds,    axis=0)
    targets_std = np.concatenate(all_tgts,     axis=0)
    pgv_aux_std = np.concatenate(all_pgv_aux,  axis=0)
    pgv_pred_std= np.concatenate(all_pgv_pred, axis=0)
    ev_indices  = np.concatenate(all_ev_idx,   axis=0)

    return preds_std, targets_std, pgv_aux_std, pgv_pred_std, ev_indices


# ── Full evaluation ───────────────────────────────────────────────────────────

def evaluate_split(
    model:      nn.Module,
    loader:     DataLoader,
    stats:      DataStats,
    events_df:  pd.DataFrame,
    cfg,
    device:     torch.device,
    band_cols:  List[str],
    split_name: str = "val",
) -> dict:
    """Run full evaluation on one split. Returns a results dict."""
    is_mt = cfg.use_pgv_aux
    is_pgv = (cfg.target_type == "pgv") and not is_mt

    preds_std, tgts_std, pgv_aux_std, pgv_pred_std, ev_idx = run_inference(
        model, loader, device, is_mt, is_pgv)

    # ── Invert standardization ────────────────────────────────────────────────
    if is_pgv:
        pred_pgv_mms  = invert_pgv(preds_std[:, 0], stats)
        true_pgv_mms  = invert_pgv(tgts_std[:, 0], stats)
        pred_spec_db  = np.full((len(preds_std), len(PRIMARY_NOMINALS)), np.nan)
        true_spec_db  = np.full((len(preds_std), len(PRIMARY_NOMINALS)), np.nan)
    else:
        pred_spec_db  = invert_spectral(preds_std, stats)  # (N, 19) dB
        # For S3 residual: add back metadata baseline prediction
        if cfg.use_meta_residual:
            ev_ids = events_df.iloc[ev_idx]["event_id"].values
            # Retrieve per-event baseline predictions from events_df columns
            resid_cols = [f"resid_{c}" for c in band_cols]
            base_cols  = band_cols  # original dB cols
            actual_db  = events_df.iloc[ev_idx][band_cols].values.astype(np.float32)
            meta_pred_db = actual_db - events_df.iloc[ev_idx][resid_cols].values.astype(np.float32)
            pred_spec_db = pred_spec_db + meta_pred_db

        true_spec_db  = events_df.iloc[ev_idx][band_cols].values.astype(np.float32)
        # PGV from auxiliary
        pgv_mean = stats.pgv_mean or 0.0
        pgv_std_ = stats.pgv_std  or 1.0
        true_pgv_mms = np.exp(pgv_aux_std * pgv_std_ + pgv_mean)
        if is_mt:
            pred_pgv_mms = np.exp(pgv_pred_std * pgv_std_ + pgv_mean)
        else:
            pred_pgv_mms = np.full(len(pred_spec_db), np.nan)

    # ── Event metadata ────────────────────────────────────────────────────────
    sub_df = events_df.iloc[ev_idx].reset_index(drop=True)

    # ── Per-band spectral metrics ─────────────────────────────────────────────
    per_band = []
    if not is_pgv:
        for i, hz in enumerate(PRIMARY_NOMINALS):
            yt = true_spec_db[:, i]
            yp = pred_spec_db[:, i]
            m = np.isfinite(yt) & np.isfinite(yp)
            if m.sum() < 5:
                continue
            meta_rmse_i = META_BASELINE_RMSE_PER_BAND[i]
            rmse_i = _rmse(yt[m], yp[m])
            per_band.append({
                "band_hz":       hz,
                "rmse_db":       rmse_i,
                "mae_db":        _mae(yt[m], yp[m]),
                "r2":            _r2(yt[m], yp[m]),
                "bias_db":       _bias(yt[m], yp[m]),
                "resid_std_db":  _res_std(yt[m], yp[m]),
                "meta_rmse_db":  meta_rmse_i,
                "delta_rmse_db": meta_rmse_i - rmse_i,  # positive = improvement
            })

    macro_rmse = float(np.mean([b["rmse_db"] for b in per_band])) if per_band else float("nan")
    macro_mae  = float(np.mean([b["mae_db"]  for b in per_band])) if per_band else float("nan")
    macro_r2   = float(np.mean([b["r2"]      for b in per_band])) if per_band else float("nan")

    # ── Total RMS metrics ─────────────────────────────────────────────────────
    if not is_pgv:
        pred_total = total_rms_from_db(pred_spec_db)
        true_total = total_rms_from_db(true_spec_db)
    else:
        pred_total = pred_pgv_mms
        true_total = true_pgv_mms

    tot_metrics = {
        "rmse_mms":      _rmse(true_total, pred_total),
        "mae_mms":       _mae(true_total, pred_total),
        "r2":            _r2(true_total, pred_total),
        "pearson_r":     _pearson(true_total, pred_total),
        "spearman_r":    _spearman(true_total, pred_total),
        "calib_slope":   float(np.polyfit(true_total, pred_total, 1)[0]),
        "true_std_mms":  float(np.std(true_total)),
        "pred_std_mms":  float(np.std(pred_total)),
        "pred_compressed": bool(np.std(pred_total) < 0.7 * np.std(true_total)),
    }

    # ── PGV direct metrics ───────────────────────────────────────────────────
    pgv_metrics: dict = {}
    if is_pgv or is_mt:
        valid = np.isfinite(pred_pgv_mms) & np.isfinite(true_pgv_mms) & (true_pgv_mms > 0)
        tp = true_pgv_mms[valid]
        pp = pred_pgv_mms[valid]
        log_rmse = _rmse(np.log(tp), np.log(pp)) if valid.sum() > 5 else float("nan")
        pgv_metrics = {
            "rmse_mms":    _rmse(tp, pp),
            "mae_mms":     _mae(tp, pp),
            "r2":          _r2(tp, pp),
            "pearson_r":   _pearson(tp, pp),
            "spearman_r":  _spearman(tp, pp),
            "log_rmse":    log_rmse,
            "calib_slope": float(np.polyfit(tp, pp, 1)[0]),
            "true_std":    float(np.std(tp)),
            "pred_std":    float(np.std(pp)),
        }

    # ── Per-event spectral RMSE ───────────────────────────────────────────────
    if not is_pgv:
        per_ev_rmse = np.sqrt(np.mean((pred_spec_db - true_spec_db) ** 2, axis=1))
    else:
        per_ev_rmse = np.abs(pred_pgv_mms - true_pgv_mms)

    # ── Long-format predictions DataFrame ────────────────────────────────────
    pred_rows = []
    for i in range(len(sub_df)):
        ev = sub_df.iloc[i]
        for j, hz in enumerate(PRIMARY_NOMINALS):
            if is_pgv:
                break
            pred_rows.append({
                "event_id":            ev["event_id"],
                "split":               split_name,
                "model_name":          cfg.name,
                "seed":                cfg.seed,
                "track_number":        int(ev["track_number"]),
                "train_type":          ev.get("train_type", ""),
                "train_family":        ev.get("train_family", ""),
                "band_nominal_hz":     hz,
                "measured_level_db":   float(true_spec_db[i, j]),
                "predicted_level_db":  float(pred_spec_db[i, j]),
                "residual_db":         float(true_spec_db[i, j] - pred_spec_db[i, j]),
                "absolute_error_db":   float(abs(true_spec_db[i, j] - pred_spec_db[i, j])),
                "measured_velocity_mms":   float(db_to_rms_mms(true_spec_db[i:i+1, j:j+1]).item()),
                "predicted_velocity_mms":  float(db_to_rms_mms(pred_spec_db[i:i+1, j:j+1]).item()),
            })

    preds_df = pd.DataFrame(pred_rows)

    # ── Event-level metrics DataFrame ─────────────────────────────────────────
    ev_metric_rows = []
    for i in range(len(sub_df)):
        ev = sub_df.iloc[i]
        ev_metric_rows.append({
            "event_id":                ev["event_id"],
            "model_name":              cfg.name,
            "seed":                    cfg.seed,
            "split":                   split_name,
            "measured_total_rms_mms":  float(true_total[i]),
            "predicted_total_rms_mms": float(pred_total[i]),
            "event_spectral_rmse_db":  float(per_ev_rmse[i]),
            "event_spectral_mae_db":   float(np.mean(np.abs(
                (pred_spec_db[i] - true_spec_db[i]) if not is_pgv else 0.0))),
            "measured_raw_pgv_mms":    float(true_pgv_mms[i]),
            "predicted_raw_pgv_mms":   float(pred_pgv_mms[i]) if (is_pgv or is_mt) else float("nan"),
            "track_number":            int(ev["track_number"]),
            "train_family":            ev.get("train_family", ""),
        })
    ev_metrics_df = pd.DataFrame(ev_metric_rows)

    return {
        "per_band":       per_band,
        "macro":          {"rmse_db": macro_rmse, "mae_db": macro_mae, "r2": macro_r2},
        "total_rms":      tot_metrics,
        "pgv_metrics":    pgv_metrics,
        "predictions_df": preds_df,
        "ev_metrics_df":  ev_metrics_df,
        "pred_spec_db":   pred_spec_db,
        "true_spec_db":   true_spec_db,
        "pred_total":     pred_total,
        "true_total":     true_total,
        "pred_pgv_mms":   pred_pgv_mms,
        "true_pgv_mms":   true_pgv_mms,
        "sub_df":         sub_df,
        "per_ev_rmse":    per_ev_rmse,
        "split":          split_name,
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────

_HZ_LABELS = [f"{hz:.4g}" for hz in PRIMARY_NOMINALS]
_XPOS = np.arange(len(PRIMARY_NOMINALS))


def _fig_save(fig, path: Path, dpi: int = 120) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── 14 Required plots ─────────────────────────────────────────────────────────

def plot_per_band_metrics(res: dict, out: Path) -> None:
    """Plot 1: Per-band RMSE, MAE, R² vs frequency."""
    per_band = res["per_band"]
    if not per_band:
        return
    rmse_v = [b["rmse_db"]    for b in per_band]
    mae_v  = [b["mae_db"]     for b in per_band]
    r2_v   = [b["r2"]         for b in per_band]
    meta_v = [b["meta_rmse_db"] for b in per_band]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    x = _XPOS

    axes[0].bar(x, meta_v, color="lightgray", alpha=0.7, label="Metadata RMSE (dB)", zorder=1)
    axes[0].bar(x, rmse_v, color="steelblue", alpha=0.9, label="Model RMSE (dB)", zorder=2)
    axes[0].set_ylabel("dB")
    axes[0].set_title("Per-band RMSE vs frequency")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(x, mae_v, color="steelblue", alpha=0.9)
    axes[1].set_ylabel("dB")
    axes[1].set_title("Per-band MAE")
    axes[1].grid(True, alpha=0.3)

    colors = ["steelblue" if v >= 0 else "tomato" for v in r2_v]
    axes[2].bar(x, r2_v, color=colors, alpha=0.9)
    axes[2].axhline(0, color="black", linewidth=0.7)
    axes[2].set_ylabel("R²")
    axes[2].set_title("Per-band R²")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(_HZ_LABELS, rotation=45, ha="right")
    axes[2].set_xlabel("Band nominal frequency (Hz)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    _fig_save(fig, out / "01_per_band_rmse_mae_r2.png")


def plot_improvement_over_baseline(res: dict, out: Path) -> None:
    """Plot 2: Per-band delta-RMSE vs metadata baseline."""
    per_band = res["per_band"]
    if not per_band:
        return
    delta = [b["delta_rmse_db"] for b in per_band]

    fig, ax = plt.subplots(figsize=(11, 4))
    colors = ["steelblue" if d >= 0 else "tomato" for d in delta]
    ax.bar(_XPOS, delta, color=colors, alpha=0.9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(_XPOS)
    ax.set_xticklabels(_HZ_LABELS, rotation=45, ha="right")
    ax.set_xlabel("Band nominal frequency (Hz)")
    ax.set_ylabel("ΔRMSE (dB)  [positive = improvement]")
    ax.set_title("Per-band improvement over metadata baseline\n"
                 "(blue = model better, red = model worse)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _fig_save(fig, out / "02_improvement_over_baseline.png")


def plot_residual_violins(res: dict, out: Path) -> None:
    """Plot 3: Violin plots of signed residuals per band."""
    true_db = res["true_spec_db"]
    pred_db = res["pred_spec_db"]
    if true_db is None or np.all(np.isnan(true_db)):
        return
    residuals = true_db - pred_db  # (N, 19) signed: positive = underpredicted

    fig, ax = plt.subplots(figsize=(14, 5))
    positions = _XPOS
    parts = ax.violinplot([residuals[:, i] for i in range(len(PRIMARY_NOMINALS))],
                          positions=positions, showmedians=True, showextrema=True,
                          widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue"); pc.set_alpha(0.6)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(_HZ_LABELS, rotation=45, ha="right")
    ax.set_xlabel("Band nominal frequency (Hz)")
    ax.set_ylabel("Residual (dB)  [true − predicted]")
    ax.set_title("Signed residuals per band")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _fig_save(fig, out / "03_residual_violins.png")


def plot_absolute_error_violins(res: dict, out: Path) -> None:
    """Plot 4: Violin plots of absolute errors per band."""
    true_db = res["true_spec_db"]
    pred_db = res["pred_spec_db"]
    if true_db is None or np.all(np.isnan(true_db)):
        return
    abs_err = np.abs(true_db - pred_db)

    fig, ax = plt.subplots(figsize=(14, 5))
    parts = ax.violinplot([abs_err[:, i] for i in range(len(PRIMARY_NOMINALS))],
                          positions=_XPOS, showmedians=True, showextrema=True,
                          widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor("darkorange"); pc.set_alpha(0.6)
    ax.set_xticks(_XPOS)
    ax.set_xticklabels(_HZ_LABELS, rotation=45, ha="right")
    ax.set_xlabel("Band nominal frequency (Hz)")
    ax.set_ylabel("Absolute error (dB)")
    ax.set_title("Absolute errors per band")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _fig_save(fig, out / "04_absolute_error_violins.png")


def plot_event_rmse_distribution(res: dict, out: Path) -> None:
    """Plot 5: Distribution of per-event spectral RMSE."""
    per_ev_rmse = res["per_ev_rmse"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per_ev_rmse, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(np.median(per_ev_rmse), color="red", linestyle="--",
               label=f"Median={np.median(per_ev_rmse):.2f} dB")
    ax.axvline(np.percentile(per_ev_rmse, 90), color="orange", linestyle=":",
               label=f"p90={np.percentile(per_ev_rmse, 90):.2f} dB")
    ax.set_xlabel("Per-event spectral RMSE (dB)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of per-event spectral RMSE across 19 bands")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _fig_save(fig, out / "05_event_rmse_distribution.png")


def plot_total_rms_scatter(res: dict, out: Path) -> None:
    """Plot 6: Predicted vs measured total RMS."""
    t, p = res["true_total"], res["pred_total"]
    tr = res["total_rms"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t, p, alpha=0.4, s=14, color="steelblue", edgecolors="none")
    lo = min(t.min(), p.min()) * 0.9; hi = max(t.max(), p.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="1:1")
    ax.set_xlabel("Measured total RMS (mm/s)")
    ax.set_ylabel("Predicted total RMS (mm/s)")
    ax.set_title(f"Total RMS  r={tr['pearson_r']:.3f}  "
                 f"R²={tr['r2']:.3f}  RMSE={tr['rmse_mms']:.4f} mm/s")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _fig_save(fig, out / "06_total_rms_scatter.png")


def plot_total_rms_residual_vs_amplitude(res: dict, out: Path) -> None:
    """Plot 7: Total-RMS residual vs measured amplitude."""
    t, p = res["true_total"], res["pred_total"]
    resid = p - t
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(t, resid, alpha=0.4, s=14, color="steelblue", edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Measured total RMS (mm/s)")
    ax.set_ylabel("Residual: predicted − measured (mm/s)")
    ax.set_title("Total-RMS residual vs measured amplitude")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _fig_save(fig, out / "07_total_rms_residual_vs_amplitude.png")


def plot_all_events_hexbin(res: dict, out: Path) -> None:
    """Plot 8: All event×band predicted vs measured (hexbin density)."""
    t = res["true_spec_db"]
    p = res["pred_spec_db"]
    if t is None or np.all(np.isnan(t)):
        return
    mask = np.isfinite(t) & np.isfinite(p)
    tv, pv = t[mask], p[mask]

    fig, ax = plt.subplots(figsize=(7, 6))
    lo = min(tv.min(), pv.min()) - 2; hi = max(tv.max(), pv.max()) + 2
    hb = ax.hexbin(tv, pv, gridsize=60, cmap="Blues",
                   extent=[lo, hi, lo, hi], bins="log")
    plt.colorbar(hb, ax=ax, label="log10(count)")
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.8, label="1:1")
    ax.set_xlabel("Measured level (dB re 1 nm/s)")
    ax.set_ylabel("Predicted level (dB re 1 nm/s)")
    ax.set_title("All event × band: predicted vs measured")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _fig_save(fig, out / "08_all_events_hexbin.png")


def plot_residual_heatmap(res: dict, out: Path) -> None:
    """Plot 9: Residual heatmap (events × frequency bands)."""
    t = res["true_spec_db"]
    p = res["pred_spec_db"]
    if t is None or np.all(np.isnan(t)):
        return
    residuals = t - p   # (N, 19)
    N = residuals.shape[0]

    # Sort events by total-RMS amplitude for readability
    sort_order = np.argsort(res["true_total"])
    resid_sorted = residuals[sort_order]

    vmax = float(np.nanpercentile(np.abs(resid_sorted), 95))

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(resid_sorted.T, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Residual (dB)  [true − predicted]")
    ax.set_yticks(range(len(PRIMARY_NOMINALS)))
    ax.set_yticklabels(_HZ_LABELS, fontsize=7)
    ax.set_xlabel("Events (sorted by measured total RMS, low→high)")
    ax.set_ylabel("Band frequency (Hz)")
    ax.set_title("Residual heatmap: events × frequency bands")
    plt.tight_layout()
    _fig_save(fig, out / "09_residual_heatmap.png")


def plot_mean_median_spectra(res: dict, out: Path) -> None:
    """Plot 10: Mean and median measured and predicted spectra."""
    t = res["true_spec_db"]
    p = res["pred_spec_db"]
    if t is None or np.all(np.isnan(t)):
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    x = _XPOS
    ax.plot(x, np.nanmean(t, axis=0),   "o-",  color="black",     lw=1.5, label="Measured mean")
    ax.plot(x, np.nanmedian(t, axis=0), "s-",  color="gray",      lw=1.2, ls="--", label="Measured median")
    ax.plot(x, np.nanmean(p, axis=0),   "o-",  color="steelblue", lw=1.5, label="Predicted mean")
    ax.plot(x, np.nanmedian(p, axis=0), "s-",  color="cornflowerblue", lw=1.2, ls="--", label="Predicted median")
    ax.set_xticks(x); ax.set_xticklabels(_HZ_LABELS, rotation=45, ha="right")
    ax.set_xlabel("Band nominal frequency (Hz)")
    ax.set_ylabel("Velocity band level (dB re 1 nm/s)")
    ax.set_title("Mean and median spectra: measured vs predicted")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _fig_save(fig, out / "10_mean_median_spectra.png")


def plot_spectral_quantile_bands(res: dict, out: Path) -> None:
    """Plot 11: Spectral quantile bands (p10/p25/p50/p75/p90)."""
    t = res["true_spec_db"]
    p = res["pred_spec_db"]
    if t is None or np.all(np.isnan(t)):
        return

    q10t, q25t = np.nanpercentile(t, 10, axis=0), np.nanpercentile(t, 25, axis=0)
    q50t        = np.nanmedian(t, axis=0)
    q75t, q90t = np.nanpercentile(t, 75, axis=0), np.nanpercentile(t, 90, axis=0)

    q10p, q25p = np.nanpercentile(p, 10, axis=0), np.nanpercentile(p, 25, axis=0)
    q50p        = np.nanmedian(p, axis=0)
    q75p, q90p = np.nanpercentile(p, 75, axis=0), np.nanpercentile(p, 90, axis=0)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = _XPOS
    ax.fill_between(x, q10t, q90t, alpha=0.12, color="black",     label="Meas. p10–p90")
    ax.fill_between(x, q25t, q75t, alpha=0.20, color="black",     label="Meas. p25–p75")
    ax.fill_between(x, q10p, q90p, alpha=0.12, color="steelblue", label="Pred. p10–p90")
    ax.fill_between(x, q25p, q75p, alpha=0.20, color="steelblue", label="Pred. p25–p75")
    ax.plot(x, q50t, "o-", color="black",     lw=1.5, label="Meas. median")
    ax.plot(x, q50p, "o-", color="steelblue", lw=1.5, label="Pred. median")
    ax.set_xticks(x); ax.set_xticklabels(_HZ_LABELS, rotation=45, ha="right")
    ax.set_xlabel("Band nominal frequency (Hz)")
    ax.set_ylabel("dB re 1 nm/s")
    ax.set_title("Spectral quantile bands: measured vs predicted")
    ax.legend(fontsize=7, ncol=3); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _fig_save(fig, out / "11_spectral_quantile_bands.png")


def plot_representative_events(res: dict, out: Path) -> None:
    """Plot 12: Spectra for low-, medium- and high-vibration representative events."""
    t = res["true_spec_db"]
    p = res["pred_spec_db"]
    if t is None or np.all(np.isnan(t)):
        return
    tot = res["true_total"]

    pcts = [(0, 25, "Low"), (37, 63, "Medium"), (75, 100, "High")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, (lo, hi, label) in zip(axes, pcts):
        th_lo = np.percentile(tot, lo)
        th_hi = np.percentile(tot, hi)
        idx   = np.where((tot >= th_lo) & (tot <= th_hi))[0]
        if len(idx) == 0:
            continue
        # Plot up to 15 events
        chosen = idx[:15]
        for i in chosen:
            ax.plot(_XPOS, t[i], color="gray",      alpha=0.4, linewidth=0.7)
            ax.plot(_XPOS, p[i], color="steelblue", alpha=0.4, linewidth=0.7, linestyle="--")
        ax.plot(_XPOS, np.nanmean(t[chosen], axis=0), "o-", color="black",     lw=1.5, label="Mean meas.")
        ax.plot(_XPOS, np.nanmean(p[chosen], axis=0), "o-", color="steelblue", lw=1.5, label="Mean pred.")
        ax.set_title(f"{label} vibration (n={len(chosen)})")
        ax.set_xticks(_XPOS[::3]); ax.set_xticklabels(_HZ_LABELS[::3], rotation=45, ha="right")
        ax.set_xlabel("Band (Hz)"); ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("dB re 1 nm/s")
            ax.legend(fontsize=7)

    plt.suptitle("Representative event spectra: measured (gray) vs predicted (blue)")
    plt.tight_layout()
    _fig_save(fig, out / "12_representative_events.png")


def plot_training_curves(history: List[dict], out: Path) -> None:
    """Plot 13: Training and validation loss curves."""
    if not history:
        return
    epochs   = [h["epoch"]       for h in history]
    tr_loss  = [h["train_loss"]  for h in history]
    val_loss = [h["val_loss"]    for h in history]
    val_rmse = [h["val_macro_rmse_std"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, tr_loss,  label="Train loss", color="steelblue")
    axes[0].plot(epochs, val_loss, label="Val loss",   color="tomato")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and validation loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_rmse, label="Val macro-RMSE (std)", color="tomato")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("RMSE (standardized)")
    axes[1].set_title("Validation macro-RMSE (standardized targets)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _fig_save(fig, out / "13_training_curves.png")


def plot_error_by_subgroup(res: dict, out: Path) -> None:
    """Plot 14: Error distributions by track and major train family."""
    sub_df  = res["sub_df"]
    ev_rmse = res["per_ev_rmse"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # By track
    ax = axes[0]
    tracks = sorted(sub_df["track_number"].unique())
    data_track = [ev_rmse[sub_df["track_number"].values == t] for t in tracks]
    ax.violinplot(data_track, positions=range(len(tracks)),
                  showmedians=True, widths=0.5)
    ax.set_xticks(range(len(tracks)))
    ax.set_xticklabels([f"Track {t}" for t in tracks])
    ax.set_ylabel("Per-event spectral RMSE (dB)")
    ax.set_title("Error by track")
    ax.grid(True, alpha=0.3)

    # By train family
    ax = axes[1]
    fams = [f for f in sorted(sub_df["train_family"].unique())
            if (sub_df["train_family"].values == f).sum() >= 10]
    data_fam = [ev_rmse[sub_df["train_family"].values == f] for f in fams]
    ax.violinplot(data_fam, positions=range(len(fams)),
                  showmedians=True, widths=0.5)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(fams, rotation=45, ha="right")
    ax.set_ylabel("Per-event spectral RMSE (dB)")
    ax.set_title("Error by train family (n≥10)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _fig_save(fig, out / "14_error_by_subgroup.png")


# ── Master output saver ───────────────────────────────────────────────────────

def save_eval_outputs(
    res:       dict,
    out_dir:   Path,
    model_name: str,
    split_name: str,
    history:   Optional[List[dict]] = None,
) -> None:
    """Save all metrics, predictions, and plots to out_dir."""
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Metrics JSON ──────────────────────────────────────────────────────────
    metrics = {
        "model":   model_name,
        "split":   split_name,
        "macro":   res["macro"],
        "total_rms": res["total_rms"],
        "per_band": res["per_band"],
        "pgv_metrics": res.get("pgv_metrics", {}),
    }
    (out_dir / f"metrics_{split_name}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # ── Per-band CSV ──────────────────────────────────────────────────────────
    if res["per_band"]:
        pd.DataFrame(res["per_band"]).to_csv(
            out_dir / f"metrics_per_band_{split_name}.csv", index=False)

    # ── Long-format predictions ───────────────────────────────────────────────
    if not res["predictions_df"].empty:
        res["predictions_df"].to_parquet(
            out_dir / f"predictions_{split_name}.parquet", index=False)

    # ── Event-level metrics ───────────────────────────────────────────────────
    if not res["ev_metrics_df"].empty:
        res["ev_metrics_df"].to_parquet(
            out_dir / f"event_metrics_{split_name}.parquet", index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_per_band_metrics(res, plots_dir)
    plot_improvement_over_baseline(res, plots_dir)
    plot_residual_violins(res, plots_dir)
    plot_absolute_error_violins(res, plots_dir)
    plot_event_rmse_distribution(res, plots_dir)
    plot_total_rms_scatter(res, plots_dir)
    plot_total_rms_residual_vs_amplitude(res, plots_dir)
    plot_all_events_hexbin(res, plots_dir)
    plot_residual_heatmap(res, plots_dir)
    plot_mean_median_spectra(res, plots_dir)
    plot_spectral_quantile_bands(res, plots_dir)
    plot_representative_events(res, plots_dir)
    if history:
        plot_training_curves(history, plots_dir)
    plot_error_by_subgroup(res, plots_dir)

    print(f"  Saved metrics + 14 plots → {plots_dir}")


def print_summary(res: dict, cfg_name: str, split_name: str) -> None:
    """Print a compact evaluation summary to stdout."""
    m   = res["macro"]
    tr  = res["total_rms"]
    print(f"\n  {'─'*60}")
    print(f"  {cfg_name} | {split_name.upper()}")
    print(f"  Macro RMSE = {m['rmse_db']:.3f} dB  "
          f"MAE = {m['mae_db']:.3f} dB  R² = {m['r2']:.3f}")
    print(f"  Total RMS  RMSE={tr['rmse_mms']:.4f} mm/s  "
          f"r={tr['pearson_r']:.3f}  R²={tr['r2']:.3f}  "
          f"compressed={tr['pred_compressed']}")
    if res.get("pgv_metrics"):
        pg = res["pgv_metrics"]
        print(f"  PGV direct  RMSE={pg['rmse_mms']:.4f} mm/s  "
              f"r={pg['pearson_r']:.3f}  R²={pg['r2']:.3f}")
    per_band = res["per_band"]
    if per_band:
        best  = max(per_band, key=lambda b: b["delta_rmse_db"])
        worst = min(per_band, key=lambda b: b["delta_rmse_db"])
        print(f"  Best  band: {best['band_hz']:5.4g} Hz  "
              f"ΔRMSE=+{best['delta_rmse_db']:.2f} dB")
        print(f"  Worst band: {worst['band_hz']:5.4g} Hz  "
              f"ΔRMSE={worst['delta_rmse_db']:.2f} dB")
    print(f"  {'─'*60}")
