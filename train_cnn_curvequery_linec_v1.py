"""
Train query-conditioned curve-prior + residual model on line-C side -1 subset.

Predicts log-PGV at arbitrary receiver distances via a continuous distance query.

Architecture:
    FO waveform (21 ch) + metadata  →  2D CNN  →  event embedding
    [event embedding + query(r)]    →  residual head  →  epsilon_hat(r)
    y_pred(r) = c_hat - n_track * log(r / r0) + epsilon_hat(r)

Variants (--variant):
    Q1  curve-only        (epsilon_hat = 0)
    Q2  curve + residual  (lambda_eps = 0.05)
    Q3  curve + residual + MP4 weighting  (mp4_weight = 2.0)
    Q4  curve + residual + stronger regularisation  (lambda_eps = 0.20)

Holdout modes (--holdout_sensor):
    None            all-sensor training
    MP4 | MP8 | MP10 | MP1 | MP2   held-out sensor generalisation test

Run:
    python train_cnn_curvequery_linec_v1.py --variant Q3
    python train_cnn_curvequery_linec_v1.py --variant Q3 --holdout_sensor MP4
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
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ml.cnn.config_cnn_curvequery_linec_v1 import Config, get_variant_config
from src.ml.cnn.cnn_curvequery_linec_utils import (
    CurvePriorCNN2D_Query,
    CurveDataset_Query,
    huber_loss_log,
    mse_scalar,
    compute_sample_weights,
)
from src.utils.geometry_utils import apply_corrected_distances


# ===========================================================================
# PATH HELPERS
# ===========================================================================

def _find_latest_waveform_build(cfg: Config) -> Path:
    if cfg.data.waveform_build_dir:
        return Path(cfg.data.waveform_build_dir)
    root = Path(cfg.data.waveform_root)
    builds = sorted(root.glob(cfg.data.waveform_glob), key=lambda p: p.name)
    if not builds:
        raise FileNotFoundError(f"No {cfg.data.waveform_glob} builds in {root}")
    return builds[-1]


def _find_latest_v2(cfg: Config) -> Path:
    if cfg.data.parquet_v2_path:
        return Path(cfg.data.parquet_v2_path)
    root = Path(cfg.data.parquet_root)
    builds = sorted(root.glob("parquet_v002_*"), key=lambda p: p.name)
    if not builds:
        raise FileNotFoundError(f"No parquet_v002_* builds in {root}")
    return builds[-1] / "dataset.parquet"


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_linec_data(
    cfg: Config,
) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """Load line-C side -1 subset with waveforms.
    Mirrors train_cnn_curveprior_linec_v1.py::load_linec_data exactly.
    """
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    v2_path = _find_latest_v2(cfg)
    print(f"Parquet v2: {v2_path}")
    df = pd.read_parquet(v2_path)

    if "effective_distance_to_active_track_m" not in df.columns:
        df = apply_corrected_distances(df)

    df_linec = df[df[cfg.data.sensor_col].isin(cfg.data.line_c_sensors)].copy()
    print(f"Line-C subset: {len(df_linec):,} rows")

    df_linec = df_linec.dropna(subset=[cfg.data.pgv_col, cfg.data.track_col])
    df_linec = df_linec[
        (df_linec[cfg.data.pgv_col] > 0) &
        (df_linec[cfg.data.track_col].isin([1, 2]))
    ]
    print(f"After cleaning: {len(df_linec):,} rows, {df_linec[cfg.data.event_col].nunique():,} events")

    wave_dir = _find_latest_waveform_build(cfg)
    print(f"Waveform build: {wave_dir}")

    waveforms = np.load(wave_dir / "waveforms.npy")
    if waveforms.dtype != np.float32:
        waveforms = waveforms.astype(np.float32)

    if waveforms.shape[1] == 51:
        waveforms = waveforms[:, 19:40, :]
        print("Sliced from ch51 (1165-1215) to ch21 (1184-1204, line-C window)")
        print("  selected slice: 19:40  |  center channel: 1194")
    elif waveforms.shape[1] != 21:
        raise ValueError(f"Unexpected waveform shape: {waveforms.shape}")

    waveforms *= np.float32(cfg.data.waveform_scale)
    print(f"Waveforms: {waveforms.shape} {waveforms.dtype}")

    index_df = pd.read_parquet(wave_dir / "event_index.parquet")
    index_ok = index_df[index_df["build_status"] == "ok"].copy()
    index_ok["event_id"] = index_ok["event_id"].astype(str)
    event_map = dict(zip(index_ok["event_id"], index_ok["waveform_row_idx"].astype(int)))
    print(f"Events with waveforms: {len(event_map):,}")

    df_linec[cfg.data.event_col] = df_linec[cfg.data.event_col].astype(str)
    df_linec = df_linec[df_linec[cfg.data.event_col].isin(event_map.keys())].copy()
    print(f"After waveform matching: {len(df_linec):,} rows, "
          f"{df_linec[cfg.data.event_col].nunique():,} events")

    return df_linec, waveforms, event_map


# ===========================================================================
# EVENT ARRAY BUILDING
# ===========================================================================

def build_event_arrays(
    df_sensor: pd.DataFrame,
    waveforms: np.ndarray,
    event_map: Dict,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert sensor-level DataFrame to event-level arrays.

    Returns:
        waveforms_arr  (N, C, T)
        metadata_arr   (N, 4)    [speed, speed_missing, train_type, track_norm]
        targets_log    (N, 5)    log(PGV_z) per sensor
        distances_arr  (N, 5)    effective distance to active track (m)
        tracks_arr     (N,)      1 or 2
        event_ids_arr  (N,)
    """
    print("\n" + "=" * 80)
    print("BUILDING EVENT-LEVEL ARRAYS")
    print("=" * 80)

    sensors = cfg.data.line_c_sensors
    wf_l, meta_l, tgt_l, dist_l, track_l, eid_l = [], [], [], [], [], []

    for event_id in df_sensor[cfg.data.event_col].unique():
        df_ev = df_sensor[df_sensor[cfg.data.event_col] == event_id]

        if len(df_ev) != 5:
            continue

        tracks_unique = df_ev[cfg.data.track_col].unique()
        if len(tracks_unique) != 1:
            continue
        track = int(tracks_unique[0])
        if track not in [1, 2]:
            continue

        if event_id not in event_map:
            continue
        wf = waveforms[event_map[event_id]]

        pgv_vec, dist_vec = [], []
        for sensor in sensors:
            row = df_ev[df_ev[cfg.data.sensor_col] == sensor]
            if len(row) != 1:
                break
            pgv  = row[cfg.data.pgv_col].values[0]
            dist = row["effective_distance_to_active_track_m"].values[0]
            pgv_vec.append(np.log(np.clip(pgv, 1e-6, None)))
            dist_vec.append(dist)
        else:
            # All 5 sensors found
            wf_l.append(wf)
            tgt_l.append(pgv_vec)
            dist_l.append(dist_vec)
            track_l.append(track)
            eid_l.append(event_id)

            row0 = df_ev.iloc[0]
            speed_raw = row0.get(cfg.data.speed_col, np.nan)
            speed = float(speed_raw) if np.isfinite(speed_raw) else 0.0
            speed_missing = float(not np.isfinite(speed_raw))
            train_type = float(row0.get("train_type_code", -1))
            if not np.isfinite(train_type):
                train_type = -1.0
            meta_l.append([speed, speed_missing, train_type, float(track) / 2.0])

    n = len(wf_l)
    print(f"Event-level samples: {n:,}")

    return (
        np.array(wf_l,    dtype=np.float32),   # (N, C, T)
        np.array(meta_l,  dtype=np.float32),   # (N, 4)
        np.array(tgt_l,   dtype=np.float32),   # (N, 5)
        np.array(dist_l,  dtype=np.float32),   # (N, 5)
        np.array(track_l, dtype=np.int64),     # (N,)
        np.array(eid_l),                        # (N,)
    )


# ===========================================================================
# SPLITS
# ===========================================================================

def make_event_splits(
    n_events:   int,
    train_frac: float,
    val_frac:   float,
    seed:       int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    idx = np.arange(n_events)
    np.random.shuffle(idx)
    n_train = int(n_events * train_frac)
    n_val   = int(n_events * val_frac)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


# ===========================================================================
# ATTENUATION EXPONENT FITTING
# ===========================================================================

def fit_attenuation_exponents_corrected(
    targets_log: np.ndarray,           # (N_events, S)  S ≤ 5
    distances:   np.ndarray,           # (N_events, S)
    tracks:      np.ndarray,           # (N_events,)
    r0: float = 10.0,
    holdout_sensor_idx: Optional[int] = None,
    sensor_names:       Optional[List[str]] = None,
) -> Tuple[float, float]:
    """Fit track-specific n values via event-intercept corrected method.

    For holdout mode, holdout_sensor_idx column is excluded before fitting.
    """
    print("\n" + "=" * 80)
    print("FITTING ATTENUATION EXPONENTS (Corrected Event-Intercept Method)")
    print("=" * 80)

    if holdout_sensor_idx is not None:
        targets_log = np.delete(targets_log, holdout_sensor_idx, axis=1)
        distances   = np.delete(distances,   holdout_sensor_idx, axis=1)
        excl = (sensor_names[holdout_sensor_idx]
                if sensor_names else str(holdout_sensor_idx))
        print(f"Holdout mode: excluding sensor {excl} from n fitting")

    results: Dict[int, float] = {}
    for track_id in [1, 2]:
        mask = tracks == track_id
        if not mask.any():
            print(f"Track {track_id}: no events — using oracle default")
            results[track_id] = 1.0777 if track_id == 1 else 1.3300
            continue

        y = targets_log[mask]                 # (n_t, S)
        x = np.log(distances[mask] / r0)      # (n_t, S)

        # Centre within each event (removes per-event intercept)
        x_c = x - x.mean(axis=1, keepdims=True)
        y_c = y - y.mean(axis=1, keepdims=True)

        denom = np.sum(x_c ** 2)
        n_global = (-np.sum(x_c * y_c) / denom) if denom > 1e-10 else 1.0

        print(f"\nTrack {track_id}:")
        print(f"  Events: {mask.sum():,}  |  Sensors: {y.shape[1]}")
        print(f"  Fitted n: {n_global:.4f}")
        results[track_id] = float(n_global)

    n_t1, n_t2 = results[1], results[2]
    print(f"\nFitted exponents:  Track 1 = {n_t1:.4f}  |  Track 2 = {n_t2:.4f}")
    return n_t1, n_t2


# ===========================================================================
# SCALERS
# ===========================================================================

def fit_meta_scaler(metadata_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(metadata_train)
    return scaler


# ===========================================================================
# TRAINING
# ===========================================================================

def train_epoch(
    model:      CurvePriorCNN2D_Query,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    cfg:        Config,
    device:     torch.device,
    n_track1:   float,
    n_track2:   float,
    r0:         float = 10.0,
    first_batch_debug: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0
    debug_done = False

    n1 = torch.tensor(n_track1, dtype=torch.float32, device=device)
    n2 = torch.tensor(n_track2, dtype=torch.float32, device=device)

    for batch in loader:
        wf      = batch["waveform"].to(device)      # (B, 1, C, T)
        meta    = batch["metadata"].to(device)       # (B, n_meta)
        tgt_log = batch["target_log"].to(device)    # (B,)
        dist    = batch["distance"].to(device)       # (B,)
        track   = batch["track"].to(device)          # (B,)  long
        snames  = batch["sensor_name"]               # list[str]

        n_vec = torch.where(track == 1, n1, n2).float()   # (B,)

        c_hat, eps = model(wf, meta, dist, track, r0=r0)

        log_r  = torch.log(dist / r0)                     # (B,)
        y_pred = c_hat - n_vec * log_r
        if eps is not None:
            y_pred = y_pred + eps

        # Per-sample intensity target  c_target = log_pgv + n * log(r/r0)
        c_target = (tgt_log + n_vec * log_r).detach()     # (B,)

        weights = compute_sample_weights(snames, cfg.model.mp4_weight, device)

        if first_batch_debug and not debug_done:
            print(f"[DEBUG] Output weights: {weights[:5].cpu().numpy()}")
            print(f"[DEBUG] MP4 weight = {cfg.model.mp4_weight}")
            debug_done = True

        L_profile = huber_loss_log(y_pred, tgt_log,
                                   delta=cfg.train.huber_delta, weights=weights)
        L_c       = mse_scalar(c_hat, c_target)
        loss      = L_profile + cfg.model.alpha_intensity * L_c

        if eps is not None and cfg.model.lambda_residual > 0:
            loss = loss + cfg.model.lambda_residual * (eps ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


# ===========================================================================
# EVALUATION
# ===========================================================================

@torch.no_grad()
def evaluate(
    model:    CurvePriorCNN2D_Query,
    loader:   DataLoader,
    device:   torch.device,
    cfg:      Config,
    n_track1: float,
    n_track2: float,
    r0: float = 10.0,
) -> Tuple[Dict, Dict, Dict]:
    """Evaluate model.  Returns (metrics_dict, arrays_dict, per_sensor_dict)."""
    model.eval()

    n1 = torch.tensor(n_track1, dtype=torch.float32, device=device)
    n2 = torch.tensor(n_track2, dtype=torch.float32, device=device)

    pl_l, tl_l, ch_l, ct_l, ep_l, dl_l, tk_l, sn_l, ei_l = (
        [], [], [], [], [], [], [], [], []
    )

    for batch in loader:
        wf      = batch["waveform"].to(device)
        meta    = batch["metadata"].to(device)
        tgt_log = batch["target_log"].to(device)
        dist    = batch["distance"].to(device)
        track   = batch["track"].to(device)

        n_vec = torch.where(track == 1, n1, n2).float()

        c_hat, eps = model(wf, meta, dist, track, r0=r0)
        log_r  = torch.log(dist / r0)
        y_pred = c_hat - n_vec * log_r
        if eps is not None:
            y_pred = y_pred + eps
        c_target = tgt_log + n_vec * log_r

        pl_l.append(y_pred.cpu().numpy())
        tl_l.append(tgt_log.cpu().numpy())
        ch_l.append(c_hat.cpu().numpy())
        ct_l.append(c_target.cpu().numpy())
        ep_l.append(eps.cpu().numpy() if eps is not None
                    else np.zeros(len(c_hat), dtype=np.float32))
        dl_l.append(dist.cpu().numpy())
        tk_l.append(track.cpu().numpy())
        sn_l.extend(batch["sensor_name"])
        ei_l.extend(list(batch["event_id"]) if not isinstance(batch["event_id"], list)
                    else batch["event_id"])

    preds_log = np.clip(np.concatenate(pl_l),
                        cfg.train.pred_log_clamp_min,
                        cfg.train.pred_log_clamp_max)
    tgts_log  = np.concatenate(tl_l)
    c_hat_arr = np.concatenate(ch_l)
    c_tgt_arr = np.concatenate(ct_l)
    eps_arr   = np.concatenate(ep_l)
    dist_arr  = np.concatenate(dl_l)
    track_arr = np.concatenate(tk_l)
    sname_arr = np.array(sn_l)
    eid_arr   = np.array(ei_l)

    preds_pgv = np.exp(preds_log)
    tgts_pgv  = np.exp(tgts_log)

    def _rmse(a, b):  return float(np.sqrt(np.mean((a - b) ** 2)))
    def _mae(a, b):   return float(np.mean(np.abs(a - b)))
    def _bias(a, b):  return float(np.mean(a - b))
    def _r2(a, b):
        ss_res = np.sum((b - a) ** 2)
        ss_tot = np.sum((b - b.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    m: Dict = {}
    m["rmse_pgv"]  = _rmse(preds_pgv, tgts_pgv)
    m["mae_pgv"]   = _mae(preds_pgv, tgts_pgv)
    m["rmse_log"]  = _rmse(preds_log, tgts_log)
    m["mae_log"]   = _mae(preds_log, tgts_log)
    m["r2_log"]    = _r2(preds_log, tgts_log)
    m["bias_log"]  = _bias(preds_log, tgts_log)

    for tid in [1, 2]:
        mask = track_arr == tid
        if mask.any():
            m[f"t{tid}_rmse_log"] = _rmse(preds_log[mask], tgts_log[mask])
            m[f"t{tid}_r2_log"]   = _r2(preds_log[mask], tgts_log[mask])
            m[f"t{tid}_rmse_pgv"] = _rmse(preds_pgv[mask], tgts_pgv[mask])

    sensor_order = ["MP4", "MP8", "MP10", "MP1", "MP2"]
    per_sensor: Dict = {}
    for s in sensor_order:
        mask = sname_arr == s
        if mask.any():
            per_sensor[s] = {
                "count":    int(mask.sum()),
                "rmse_pgv": _rmse(preds_pgv[mask], tgts_pgv[mask]),
                "rmse_log": _rmse(preds_log[mask], tgts_log[mask]),
                "bias_pgv": _bias(preds_pgv[mask], tgts_pgv[mask]),
                "bias_log": _bias(preds_log[mask], tgts_log[mask]),
                "eps_rms":  float(np.sqrt(np.mean(eps_arr[mask] ** 2))),
            }

    mp4_mask = sname_arr == "MP4"
    if mp4_mask.any():
        m["mp4_rmse_pgv"] = _rmse(preds_pgv[mp4_mask], tgts_pgv[mp4_mask])
        m["mp4_bias_pgv"] = _bias(preds_pgv[mp4_mask], tgts_pgv[mp4_mask])
        m["mp4_rmse_log"] = _rmse(preds_log[mp4_mask], tgts_log[mp4_mask])

    hi_mask = tgts_pgv > 4.0
    if hi_mask.any():
        m["hi_pgv_n"]        = int(hi_mask.sum())
        m["hi_pgv_rmse_pgv"] = _rmse(preds_pgv[hi_mask], tgts_pgv[hi_mask])
        m["hi_pgv_bias_pgv"] = _bias(preds_pgv[hi_mask], tgts_pgv[hi_mask])

    m["eps_rms"] = float(np.sqrt(np.mean(eps_arr ** 2)))

    if len(c_hat_arr) > 1:
        corr = float(np.corrcoef(c_hat_arr, c_tgt_arr)[0, 1])
        m["c_hat_corr"] = corr
        m["c_hat_r2"]   = _r2(c_hat_arr, c_tgt_arr)

    # Monotonicity check on first 200 events with all 5 sensors
    n_sensors_present = len(np.unique(sname_arr))
    if n_sensors_present == 5:
        evs = np.unique(eid_arr)[:200]
        violations, total_pairs = 0, 0
        for eid in evs:
            ev_m = eid_arr == eid
            if ev_m.sum() != 5:
                continue
            p_ev = {s: preds_log[ev_m & (sname_arr == s)][0]
                    for s in sensor_order if (ev_m & (sname_arr == s)).any()}
            if len(p_ev) < 5:
                continue
            pv = [p_ev[s] for s in sensor_order]
            for i in range(len(pv) - 1):
                total_pairs += 1
                if pv[i] < pv[i + 1]:
                    violations += 1
        m["mono_viol_rate"] = violations / total_pairs if total_pairs > 0 else 0.0

    arrays = {
        "preds_log": preds_log, "tgts_log": tgts_log,
        "preds_pgv": preds_pgv, "tgts_pgv": tgts_pgv,
        "c_hat": c_hat_arr, "c_target": c_tgt_arr,
        "epsilon": eps_arr, "distances": dist_arr,
        "tracks": track_arr, "sensors": sname_arr, "event_ids": eid_arr,
    }
    return m, arrays, per_sensor


# ===========================================================================
# PLOTS
# ===========================================================================

def save_plots(
    output_dir:      Path,
    arrays:          Dict,
    per_sensor:      Dict,
    train_history:   List[Dict],
    holdout_sensor:  Optional[str] = None,
    holdout_arrays:  Optional[Dict] = None,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    pl = arrays["preds_log"]; tl = arrays["tgts_log"]
    pp = arrays["preds_pgv"]; tp = arrays["tgts_pgv"]
    da = arrays["distances"];  tr = arrays["tracks"]
    sa = arrays["sensors"];    ea = arrays["epsilon"]
    ch = arrays["c_hat"];      ct = arrays["c_target"]
    ei = arrays["event_ids"]

    # 1. Learning curve
    if train_history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ep = [h["epoch"] for h in train_history]
        ax.plot(ep, [h["train_loss"] for h in train_history], label="train_loss")
        ax.plot(ep, [h["val_rmse_log"] for h in train_history], label="val_rmse_log")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss / RMSE(log)")
        ax.set_title("Learning Curve"); ax.legend(); ax.grid(True)
        fig.tight_layout()
        fig.savefig(plots_dir / "learning_curve.png", dpi=120)
        plt.close(fig)

    # 2. Measured vs predicted log-log
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(tp, pp, s=5, alpha=0.4, rasterized=True)
    lim = max(tp.max(), pp.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Measured PGV (mm/s)"); ax.set_ylabel("Predicted PGV (mm/s)")
    ax.set_title("Measured vs Predicted"); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "measured_vs_predicted_loglog.png", dpi=120)
    plt.close(fig)

    res = pp - tp

    # 3. Residuals vs PGV
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(tp, res, s=5, alpha=0.4, rasterized=True)
    ax.axhline(0, color="r", lw=1)
    ax.set_xlabel("Measured PGV (mm/s)"); ax.set_ylabel("Residual (mm/s)")
    ax.set_title("Residuals vs PGV"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "residuals_vs_pgv.png", dpi=120)
    plt.close(fig)

    # 4. Residuals vs distance
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(da, res, s=5, alpha=0.4, rasterized=True)
    ax.axhline(0, color="r", lw=1)
    ax.set_xlabel("Distance (m)"); ax.set_ylabel("Residual (mm/s)")
    ax.set_title("Residuals vs Distance"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "residuals_vs_distance.png", dpi=120)
    plt.close(fig)

    # 5. Per-sensor residual boxplot
    s_order = ["MP4", "MP8", "MP10", "MP1", "MP2"]
    data_box  = [res[sa == s]  for s in s_order if (sa == s).any()]
    lbls_box  = [s             for s in s_order if (sa == s).any()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data_box, labels=lbls_box, showfliers=False)
    ax.axhline(0, color="r", lw=1)
    ax.set_ylabel("Residual (mm/s)"); ax.set_title("Per-Sensor Residuals")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "per_sensor_residual_boxplot.png", dpi=120)
    plt.close(fig)

    # 6 & 7. Profile examples per track
    for tid, fname in [(1, "profile_examples_track1.png"),
                       (2, "profile_examples_track2.png")]:
        evs = np.unique(ei[tr == tid])[:8]
        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        axes = axes.flatten()
        for ax_i, eid in enumerate(evs[:8]):
            ev_m = (ei == eid) & (tr == tid)
            if ev_m.sum() < 2:
                continue
            d_ev = da[ev_m]; p_ev = pp[ev_m]; t_ev = tp[ev_m]
            order = np.argsort(d_ev)
            axes[ax_i].plot(d_ev[order], t_ev[order], "ko-", ms=4, label="meas")
            axes[ax_i].plot(d_ev[order], p_ev[order], "r^--", ms=4, label="pred")
            axes[ax_i].set_title(str(eid)[-8:], fontsize=8)
            axes[ax_i].set_xlabel("d (m)"); axes[ax_i].grid(True, alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Profile Examples — Track {tid}")
        fig.tight_layout()
        fig.savefig(plots_dir / fname, dpi=120)
        plt.close(fig)

    # 8. Epsilon vs distance
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(da, ea, s=5, alpha=0.4, rasterized=True)
    ax.axhline(0, color="r", lw=1)
    ax.set_xlabel("Distance (m)"); ax.set_ylabel("ε_hat (log units)")
    ax.set_title("Residual Head Output vs Distance"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "epsilon_vs_distance.png", dpi=120)
    plt.close(fig)

    # 9. Epsilon distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ea, bins=60, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="r", lw=1)
    ax.set_xlabel("ε_hat"); ax.set_title("Residual Distribution"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "epsilon_distribution.png", dpi=120)
    plt.close(fig)

    # 10. c_hat vs c_target
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ct, ch, s=5, alpha=0.4, rasterized=True)
    lim = [min(ct.min(), ch.min()), max(ct.max(), ch.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("c_target (log)"); ax.set_ylabel("c_hat (log)")
    ax.set_title("Event Intensity: Predicted vs Target"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "c_hat_vs_c_target.png", dpi=120)
    plt.close(fig)

    # 11. Held-out sensor predictions
    if holdout_sensor and holdout_arrays:
        hp = holdout_arrays["preds_pgv"]; ht = holdout_arrays["tgts_pgv"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].scatter(ht, hp, s=8, alpha=0.5)
        lim = max(ht.max(), hp.max()) * 1.05
        axes[0].plot([0, lim], [0, lim], "r--", lw=1)
        axes[0].set_xlabel("Measured"); axes[0].set_ylabel("Predicted")
        axes[0].set_title(f"Held-out {holdout_sensor} — Scatter")
        axes[0].grid(True, alpha=0.3)
        hr = hp - ht
        axes[1].hist(hr, bins=40); axes[1].axvline(0, color="r")
        axes[1].set_xlabel("Residual (mm/s)")
        axes[1].set_title(f"Held-out {holdout_sensor} — Residuals")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "heldout_sensor_predictions.png", dpi=120)
        plt.close(fig)

    print(f"Plots saved to: {plots_dir}")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant",        choices=["Q1", "Q2", "Q3", "Q4"], default="Q1")
    parser.add_argument("--holdout_sensor", default=None)
    parser.add_argument("--lambda_epsilon", type=float, default=None)
    args = parser.parse_args()

    cfg     = get_variant_config(args.variant, holdout_sensor=args.holdout_sensor)
    holdout = args.holdout_sensor

    if args.lambda_epsilon is not None:
        cfg.model.lambda_residual = args.lambda_epsilon

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu"
    )

    print("\n" + "=" * 80)
    print(f"QUERY-CONDITIONED CURVE-PRIOR CNN — Variant {args.variant}")
    print("=" * 80)
    print(f"Device:         {device}")
    print(f"Holdout sensor: {holdout}")
    print(f"Lambda epsilon: {cfg.model.lambda_residual}")
    print(f"MP4 weight:     {cfg.model.mp4_weight}")

    # ── Data ────────────────────────────────────────────────────────────────
    df, waveforms, event_map = load_linec_data(cfg)
    wf_arr, meta_arr, tgt_log, dist_arr, track_arr, eid_arr = build_event_arrays(
        df, waveforms, event_map, cfg
    )
    n_events = len(wf_arr)
    print(f"\nDataset: {n_events} events")

    # ── Splits ───────────────────────────────────────────────────────────────
    train_idx, val_idx, test_idx = make_event_splits(
        n_events, cfg.data.train_fraction, cfg.data.val_fraction, cfg.data.seed_split
    )
    print(f"Train: {len(train_idx)} events  |  Val: {len(val_idx)}  |  Test: {len(test_idx)}")

    # ── Scaler (fit on training metadata only) ────────────────────────────────
    meta_scaler     = fit_meta_scaler(meta_arr[train_idx])
    meta_arr_scaled = meta_scaler.transform(meta_arr)

    # ── Fit attenuation exponents on training split only ─────────────────────
    sensor_names  = cfg.data.line_c_sensors
    holdout_idx   = sensor_names.index(holdout) if holdout is not None else None

    n_track1, n_track2 = fit_attenuation_exponents_corrected(
        tgt_log[train_idx], dist_arr[train_idx], track_arr[train_idx],
        r0=cfg.features.r0,
        holdout_sensor_idx=holdout_idx,
        sensor_names=sensor_names,
    )

    # ── Datasets ─────────────────────────────────────────────────────────────
    def _ds(indices, holdout_sensor=None, holdout_only=False):
        return CurveDataset_Query(
            waveforms=wf_arr[indices],
            metadata=meta_arr_scaled[indices],
            targets_log=tgt_log[indices],
            distances=dist_arr[indices],
            tracks=track_arr[indices],
            event_ids=eid_arr[indices],
            sensor_names=sensor_names,
            holdout_sensor=holdout_sensor,
            include_holdout_only=holdout_only,
        )

    train_ds    = _ds(train_idx, holdout_sensor=holdout)
    val_ds      = _ds(val_idx,   holdout_sensor=holdout)
    test_ds     = _ds(test_idx,  holdout_sensor=holdout)
    held_ds     = _ds(test_idx, holdout_sensor=holdout, holdout_only=True)                   if holdout else None

    kw = dict(batch_size=cfg.train.batch_size, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)
    held_loader  = DataLoader(held_ds,  shuffle=False, **kw) if held_ds else None

    print(f"Train samples: {len(train_ds):,}  |  Val: {len(val_ds):,}  |  Test: {len(test_ds):,}")
    if holdout:
        print(f"Held-out test samples: {len(held_ds):,}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = CurvePriorCNN2D_Query(
        n_metadata=meta_arr.shape[1],
        conv_channels=cfg.model.conv_channels,
        kernel_ch=cfg.model.kernel_ch,
        kernel_time=cfg.model.kernel_time,
        stride_time=cfg.model.stride_time,
        use_batchnorm=cfg.model.use_batchnorm,
        metadata_hidden=cfg.model.metadata_hidden,
        intensity_hidden=cfg.model.intensity_hidden,
        query_hidden=cfg.model.query_hidden,
        residual_hidden=cfg.model.residual_hidden,
        enable_residual_head=cfg.model.enable_residual_head,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: CurvePriorCNN2D_Query  |  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_val  = float("inf")
    best_epoch = 0
    best_state = None
    patience   = 0
    history    = []

    for epoch in range(1, cfg.train.epochs + 1):
        tr_loss = train_epoch(
            model, train_loader, optimizer, cfg, device,
            n_track1, n_track2, cfg.features.r0,
            first_batch_debug=(epoch == 1),
        )
        val_m, _, _ = evaluate(model, val_loader, device, cfg,
                               n_track1, n_track2, cfg.features.r0)
        val_rmse = val_m["rmse_log"]
        val_r2   = val_m["r2_log"]

        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "val_rmse_log": val_rmse, "val_r2": val_r2})

        if epoch % 5 == 0 or epoch <= 5:
            print(f"Epoch {epoch:3d}: train_loss={tr_loss:.4f}"
                  f"  val_rmse_log={val_rmse:.4f}  val_r2={val_r2:.4f}")

        if val_rmse < best_val:
            best_val   = val_rmse
            best_epoch = epoch
            patience   = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            if epoch > 5:
                print(f"  ✓ Best epoch: {epoch} (val_rmse_log={val_rmse:.4f})")
        else:
            patience += 1

        if patience >= cfg.train.patience_early_stopping:
            print(f"Early stopping at epoch {epoch} (best was epoch {best_epoch})")
            break

    print(f"\n[CHECKPOINT] Restoring best model from epoch {best_epoch}")
    model.load_state_dict(best_state)

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL EVALUATION (using best checkpoint)")
    print("=" * 80)

    test_m, test_arr, per_sensor = evaluate(
        model, test_loader, device, cfg, n_track1, n_track2, cfg.features.r0
    )
    print(f"\nTEST METRICS (non-held sensors):")
    print(f"  RMSE(PGV): {test_m['rmse_pgv']:.4f} mm/s")
    print(f"  RMSE(log): {test_m['rmse_log']:.4f}")
    print(f"  R²(log):   {test_m['r2_log']:.4f}")
    if "mp4_rmse_pgv" in test_m:
        print(f"  MP4 RMSE(PGV): {test_m['mp4_rmse_pgv']:.4f} mm/s")
    print("\nPer-sensor RMSE(PGV) | bias(PGV):")
    for s, sm in per_sensor.items():
        print(f"  {s:5s}: RMSE={sm['rmse_pgv']:.3f}  bias={sm['bias_pgv']:+.3f}")

    held_m, held_arr, held_per = None, None, None
    if held_loader and holdout:
        held_m, held_arr, held_per = evaluate(
            model, held_loader, device, cfg, n_track1, n_track2, cfg.features.r0
        )
        print(f"\nHELD-OUT SENSOR ({holdout}) TEST METRICS:")
        print(f"  RMSE(PGV): {held_m['rmse_pgv']:.4f} mm/s")
        print(f"  RMSE(log): {held_m['rmse_log']:.4f}")
        print(f"  bias(log): {held_m.get('bias_log', 0.0):+.4f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    holdout_tag = f"holdout_{holdout}" if holdout else "all"
    output_dir  = Path(cfg.output.output_root) / (
        f"cnn_curvequery_linec_v001_v{args.variant}_{holdout_tag}_{ts}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    snap = {
        "variant": args.variant,
        "holdout_sensor": holdout,
        "n_mode": "fit_corrected",
        "n_track1": n_track1, "n_track2": n_track2,
        "lambda_epsilon": cfg.model.lambda_residual,
        "mp4_weight": cfg.model.mp4_weight,
        "alpha_intensity": cfg.model.alpha_intensity,
        "enable_residual_head": cfg.model.enable_residual_head,
        "train_events": int(len(train_idx)),
        "val_events":   int(len(val_idx)),
        "test_events":  int(len(test_idx)),
        "best_epoch":   best_epoch,
        "val_rmse_log": float(best_val),
    }
    with open(output_dir / "config_snapshot.json", "w") as f:
        json.dump(snap, f, indent=2)

    metrics_out = {"test": test_m, "per_sensor": per_sensor,
                   "best_epoch": best_epoch,
                   "n_track1": n_track1, "n_track2": n_track2}
    if held_m:
        metrics_out[f"held_{holdout}"] = held_m
        metrics_out[f"held_{holdout}_per_sensor"] = held_per
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    pred_df = pd.DataFrame({
        "event_id":   test_arr["event_ids"], "sensor": test_arr["sensors"],
        "track":      test_arr["tracks"],    "distance": test_arr["distances"],
        "pred_log":   test_arr["preds_log"], "target_log": test_arr["tgts_log"],
        "pred_pgv":   test_arr["preds_pgv"], "target_pgv": test_arr["tgts_pgv"],
        "epsilon":    test_arr["epsilon"],   "c_hat": test_arr["c_hat"],
        "c_target":   test_arr["c_target"],
    })
    pred_df.to_parquet(output_dir / "predictions.parquet", index=False)

    if held_arr:
        pd.DataFrame({
            "event_id":  held_arr["event_ids"], "sensor": held_arr["sensors"],
            "track":     held_arr["tracks"],    "distance": held_arr["distances"],
            "pred_log":  held_arr["preds_log"], "target_log": held_arr["tgts_log"],
            "pred_pgv":  held_arr["preds_pgv"], "target_pgv": held_arr["tgts_pgv"],
        }).to_parquet(output_dir / "held_predictions.parquet", index=False)

    pd.DataFrame(
        [{"sensor": s, **sm} for s, sm in per_sensor.items()]
    ).to_csv(output_dir / "per_sensor_metrics.csv", index=False)

    pd.DataFrame([
        {"track": t, "rmse_log": test_m.get(f"t{t}_rmse_log"),
         "r2_log": test_m.get(f"t{t}_r2_log"),
         "rmse_pgv": test_m.get(f"t{t}_rmse_pgv")}
        for t in [1, 2]
    ]).to_csv(output_dir / "per_track_metrics.csv", index=False)

    torch.save(model.state_dict(), output_dir / "model.pth")
    np.save(output_dir / "meta_scaler_mean.npy",  meta_scaler.mean_)
    np.save(output_dir / "meta_scaler_scale.npy", meta_scaler.scale_)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    save_plots(output_dir, test_arr, per_sensor, history,
               holdout_sensor=holdout, holdout_arrays=held_arr)

    print(f"\nOutput saved to: {output_dir}")
    print(f"Query model {args.variant} ({holdout_tag}) complete")


if __name__ == "__main__":
    main()
