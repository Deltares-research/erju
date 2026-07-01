"""Train track-conditioned curve-prior + residual model on line-C subset.

SAVEPRED VARIANT — identical to train_cnn_curveprior_linec_v1.py but additionally
saves model.pth, predictions.parquet (test split), and per_sensor_metrics.csv.
Do NOT merge this file back into the validated original.

Physics-informed prediction:
  y_pred = c_hat - n_track * log(r/r0) + epsilon_hat

Supports variants:
  P1 — curve-only (epsilon_hat = 0)
  P2 — curve + residual (lambda_residual > 0)
  P3 — curve + residual + MP4 weighting
  P4 — curve + residual + MP4 weighting + monotonicity
"""

from __future__ import annotations

import argparse
import json
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ml.cnn.config_cnn_curveprior_linec_v1 import Config, get_variant_config
from src.ml.cnn.cnn_curveprior_linec_utils import (
    CurveDataset,
    CurvePriorCNN2D,
    huber_loss_log,
    residual_regularization,
    curve_intensity_loss,
    monotonicity_penalty,
    compute_output_weights,
    compute_curve_target,
)
from src.utils.geometry_utils import apply_corrected_distances


# ==================================================================================
# HELPERS
# ==================================================================================

def _find_latest_waveform_build(cfg) -> Path:
    """Auto-discover latest waveform build."""
    if cfg.data.waveform_build_dir:
        return Path(cfg.data.waveform_build_dir)
    root = Path(cfg.data.waveform_root)
    builds = sorted(root.glob(cfg.data.waveform_glob), key=lambda p: p.name)
    if not builds:
        builds_ch51 = sorted(root.glob("holten_waveform_v003_ch51_*"), key=lambda p: p.name)
        if builds_ch51:
            return builds_ch51[-1]
    return builds[-1] if builds else None


def _find_latest_v2(cfg) -> Path:
    """Auto-discover latest parquet v002 build."""
    if cfg.data.parquet_v2_path:
        return Path(cfg.data.parquet_v2_path)
    root = Path(cfg.data.parquet_root)
    builds = sorted(root.glob("parquet_v002_*"), key=lambda p: p.name)
    if not builds:
        raise FileNotFoundError("No parquet_v002_* builds found")
    return builds[-1] / "dataset.parquet"


# ==================================================================================
# DATA LOADING
# ==================================================================================

def load_linec_data(cfg: Config) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """Load line-C side -1 subset with waveforms."""
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    # Parquet
    v2_path = _find_latest_v2(cfg)
    print(f"Parquet v2: {v2_path}")
    df = pd.read_parquet(v2_path)
    
    # Apply geometry corrections
    if "effective_distance_to_active_track_m" not in df.columns:
        df = apply_corrected_distances(df)
    
    # Filter to line-C side -1
    df_linec = df[df[cfg.data.sensor_col].isin(cfg.data.line_c_sensors)].copy()
    print(f"Line-C subset: {len(df_linec):,} rows")
    
    # Clean
    df_linec = df_linec.dropna(subset=[cfg.data.pgv_col, cfg.data.track_col])
    df_linec = df_linec[(df_linec[cfg.data.pgv_col] > 0) & 
                        (df_linec[cfg.data.track_col].isin([1, 2]))]
    print(f"After cleaning: {len(df_linec):,} rows, {df_linec[cfg.data.event_col].nunique():,} events")
    
    # Load waveforms
    wave_dir = _find_latest_waveform_build(cfg)
    print(f"Waveform build: {wave_dir}")
    
    waveforms = np.load(wave_dir / "waveforms.npy")
    if waveforms.dtype != np.float32:
        waveforms = waveforms.astype(np.float32)
    
    # Slice if ch51
    if waveforms.shape[1] == 51:
        waveforms = waveforms[:, 19:40, :]
        print(f"Sliced from ch51 (1165-1215) to ch21 (1184-1204, line-C window)")
        print(f"  ch51 build channels: 1165-1215")
        print(f"  selected slice: 19:40")
        print(f"  selected channel IDs: 1184-1204")
        print(f"  center channel: 1194")
    elif waveforms.shape[1] != 21:
        raise ValueError(f"Unexpected waveform shape: {waveforms.shape}")
    
    waveforms *= np.float32(cfg.data.waveform_scale)
    print(f"Waveforms: {waveforms.shape} {waveforms.dtype}")
    
    # Load event index
    index_df = pd.read_parquet(wave_dir / "event_index.parquet")
    index_ok = index_df[index_df["build_status"] == "ok"].copy()
    index_ok["event_id"] = index_ok["event_id"].astype(str)
    event_map = dict(zip(index_ok["event_id"], index_ok["waveform_row_idx"].astype(int)))
    print(f"Events with waveforms: {len(event_map):,}")
    
    # Map events
    df_linec[cfg.data.event_col] = df_linec[cfg.data.event_col].astype(str)
    df_linec = df_linec[df_linec[cfg.data.event_col].isin(event_map.keys())].copy()
    print(f"After waveform matching: {len(df_linec):,} rows, {df_linec[cfg.data.event_col].nunique():,} events")
    
    return df_linec, waveforms, event_map


# ==================================================================================
# DATASET CONSTRUCTION
# ==================================================================================

def build_event_level_dataset(
    df_sensor: pd.DataFrame,
    waveforms: np.ndarray,
    event_map: Dict,
    cfg: Config,
) -> Tuple[CurveDataset, pd.DataFrame]:
    """Convert sensor-level DataFrame to event-level dataset with distances."""
    print("\n" + "=" * 80)
    print("BUILDING EVENT-LEVEL DATASET")
    print("=" * 80)
    
    waveforms_list = []
    targets_list = []
    distances_list = []
    tracks_list = []
    event_ids_list = []
    metadata_list = []
    
    sensors = cfg.data.line_c_sensors
    
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
        
        # Get waveform
        if event_id not in event_map:
            continue
        wf_idx = event_map[event_id]
        wf = waveforms[wf_idx]
        
        # Extract target vector and distances
        pgv_vec = []
        dist_vec = []
        for sensor in sensors:
            df_sens = df_ev[df_ev[cfg.data.sensor_col] == sensor]
            if len(df_sens) != 1:
                break
            pgv = df_sens[cfg.data.pgv_col].values[0]
            dist = df_sens["effective_distance_to_active_track_m"].values[0]
            pgv_vec.append(np.log(np.clip(pgv, 1e-6, None)))
            dist_vec.append(dist)
        else:
            # All 5 sensors found
            waveforms_list.append(wf)
            targets_list.append(pgv_vec)
            distances_list.append(dist_vec)
            tracks_list.append(track)
            event_ids_list.append(event_id)
            
            # Metadata
            row = df_ev.iloc[0]
            meta_vec = []
            speed = row.get(cfg.features.speed_col, np.nan)
            if not np.isfinite(speed):
                speed = 0.0
            meta_vec.append(speed)
            
            train_type = row.get("train_type_code", -1)
            if not np.isfinite(train_type):
                train_type = -1
            meta_vec.append(float(train_type))
            
            track_norm = float(track) / 2.0
            meta_vec.append(track_norm)
            
            metadata_list.append(meta_vec)
    
    print(f"Event-level samples: {len(waveforms_list):,}")
    
    metadata_arr = np.array(metadata_list, dtype=np.float32)
    waveforms_arr = np.array(waveforms_list, dtype=np.float32)
    targets_arr = np.array(targets_list, dtype=np.float32)
    distances_arr = np.array(distances_list, dtype=np.float32)
    tracks_arr = np.array(tracks_list, dtype=np.int64)
    event_ids_arr = np.array(event_ids_list)
    
    dataset = CurveDataset(
        waveforms=waveforms_arr,
        metadata=metadata_arr,
        targets=targets_arr,
        distances=distances_arr,
        tracks=tracks_arr,
        event_ids=event_ids_arr,
    )
    
    event_df = pd.DataFrame({
        "event_id": event_ids_arr,
        "track": tracks_arr,
    })
    
    return dataset, event_df


# ==================================================================================
# TRAIN / VAL / TEST SPLITS
# ==================================================================================

def make_event_splits(
    n_events: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Event-level train/val/test split."""
    np.random.seed(seed)
    idx = np.arange(n_events)
    np.random.shuffle(idx)
    
    n_train = int(n_events * train_frac)
    n_val = int(n_events * val_frac)
    
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val:]
    
    return train_idx, val_idx, test_idx


def fit_attenuation_exponents_corrected(
    targets_log: np.ndarray,  # (n_events, 5)
    distances: np.ndarray,     # (n_events, 5)
    tracks: np.ndarray,        # (n_events,)
    r0: float = 10.0,
) -> Tuple[float, float]:
    """Fit track-specific attenuation exponents using event-intercept corrected method.
    
    Model: y_ij = c_i - n_track * log(r_j / r0)
    
    where:
      y_ij = log(PGV) at event i, sensor j
      c_i = event-specific intercept (intensity)
      n_track = global attenuation exponent for track
      r_j = distance to sensor j
      r0 = reference distance
    
    For each event, center the predictors and targets, then fit global n using least squares.
    """
    print("\n" + "=" * 80)
    print("FITTING ATTENUATION EXPONENTS (Corrected Event-Intercept Method)")
    print("=" * 80)
    
    n_track1 = []
    n_track2 = []
    
    for track_id in [1, 2]:
        mask = tracks == track_id
        if not mask.any():
            print(f"Track {track_id}: No events")
            continue
        
        targets_t = targets_log[mask]  # (n_events_t, 5)
        distances_t = distances[mask]  # (n_events_t, 5)
        
        n_events = len(targets_t)
        
        # Design matrix: x = log(r / r0)
        x = np.log(distances_t / r0)  # (n_events, 5)
        
        # Center x within each event
        x_mean = x.mean(axis=1, keepdims=True)  # (n_events, 1)
        x_c = x - x_mean  # (n_events, 5) centered
        
        # Targets y = log(PGV)
        y = targets_t  # (n_events, 5)
        
        # Center y within each event
        y_mean = y.mean(axis=1, keepdims=True)  # (n_events, 1)
        y_c = y - y_mean  # (n_events, 5) centered
        
        # Global slope: sum over all events and sensors
        # Model: y = c - n*x  →  y_c = -n*x_c
        numerator = np.sum(x_c * y_c)
        denominator = np.sum(x_c ** 2)
        
        beta_global = numerator / denominator if denominator > 1e-10 else 0.0
        n_global = -beta_global
        
        print(f"\nTrack {track_id}:")
        print(f"  Complete events: {n_events}")
        print(f"  Total sensors: {n_events * 5}")
        print(f"  Fitted n: {n_global:.4f}")
        
        if track_id == 1:
            n_track1.append(n_global)
        else:
            n_track2.append(n_global)
    
    n_t1 = float(n_track1[0]) if n_track1 else 1.0777
    n_t2 = float(n_track2[0]) if n_track2 else 1.3300
    
    print(f"\nFitted exponents:")
    print(f"  Track 1: {n_t1:.4f}")
    print(f"  Track 2: {n_t2:.4f}")
    
    return n_t1, n_t2


# ==================================================================================
# TRAINING
# ==================================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    device: str,
    n_track1: float,
    n_track2: float,
    r0: float = 10.0,
    first_batch_debug: bool = False,
) -> float:
    """Train for one epoch with curve-prior + residual losses."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    debug_logged = False
    
    for batch in train_loader:
        wf, meta, targets, distances, tracks, event_ids = batch
        wf = wf.to(device)
        meta = meta.to(device)
        targets = targets.to(device)
        distances = distances.to(device)
        tracks = tracks.to(device)
        
        batch_size = wf.shape[0]
        
        # Compute per-output weights
        if cfg.model.use_pgv_weighting:
            batch_weights = compute_output_weights(batch_size, cfg.model.mp4_weight, device)
        else:
            batch_weights = torch.ones(batch_size, 5, device=device, dtype=torch.float32)
        
        # Debug: print weights on first batch
        if first_batch_debug and not debug_logged:
            print(f"[DEBUG] Output weights: {batch_weights[0].cpu().numpy()}")
            print(f"[DEBUG] MP4 weight = {cfg.model.mp4_weight}")
            debug_logged = True
        
        # Forward
        c_hat, epsilon_hat = model(wf, meta)
        
        # Compute curve predictions
        # Determine n_track per sample
        n_track_vec = torch.where(
            tracks == 1,
            torch.full_like(tracks, n_track1, dtype=torch.float32),
            torch.full_like(tracks, n_track2, dtype=torch.float32),
        )  # (B,)
        
        # Base curve: -n_track * log(r / r0)
        base_curve = -n_track_vec.unsqueeze(1) * torch.log(distances / r0)  # (B, 5)
        
        # Predicted log-PGV
        y_pred = c_hat.unsqueeze(1) + base_curve  # (B, 5)
        if cfg.model.lambda_residual > 0:
            y_pred = y_pred + epsilon_hat
        
        # Main loss: Huber on predictions
        mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        loss_profile = huber_loss_log(y_pred, targets, mask, delta=cfg.train.huber_delta, weights=batch_weights)
        
        # Curve intensity auxiliary loss
        c_target = (targets + n_track_vec.unsqueeze(1) * torch.log(distances / r0)).mean(dim=1)
        loss_curve = curve_intensity_loss(c_hat, c_target, mask)
        
        # Residual regularization
        loss_residual = residual_regularization(epsilon_hat, mask) if cfg.model.lambda_residual > 0 else torch.tensor(0.0, device=device)
        
        # Monotonicity penalty
        loss_mono = monotonicity_penalty(y_pred, mask) if cfg.model.lambda_monotonic > 0 else torch.tensor(0.0, device=device)
        
        # Total loss
        loss = loss_profile + cfg.model.alpha_intensity * loss_curve + cfg.model.lambda_residual * loss_residual + cfg.model.lambda_monotonic * loss_mono
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    cfg: Config,
    n_track1: float,
    n_track2: float,
    r0: float = 10.0,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model and return metrics."""
    model.eval()
    
    all_preds_log = []
    all_targets_log = []
    all_c_hat = []
    all_c_target = []
    all_distances = []
    all_tracks = []
    all_event_ids = []   # SAVEPRED addition
    all_epsilon = []     # SAVEPRED addition
    total_loss = 0.0
    n_batches = 0
    
    for batch in data_loader:
        wf, meta, targets, distances, tracks, event_ids = batch
        wf = wf.to(device)
        meta = meta.to(device)
        targets = targets.to(device)
        distances = distances.to(device)
        tracks = tracks.to(device)
        
        batch_size = wf.shape[0]
        
        # Forward
        c_hat, epsilon_hat = model(wf, meta)
        
        # Compute curve predictions
        n_track_vec = torch.where(
            tracks == 1,
            torch.full_like(tracks, n_track1, dtype=torch.float32),
            torch.full_like(tracks, n_track2, dtype=torch.float32),
        )
        
        base_curve = -n_track_vec.unsqueeze(1) * torch.log(distances / r0)
        y_pred = c_hat.unsqueeze(1) + base_curve
        if cfg.model.lambda_residual > 0:
            y_pred = y_pred + epsilon_hat
        
        # Loss
        mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        batch_weights = compute_output_weights(batch_size, cfg.model.mp4_weight, device) if cfg.model.use_pgv_weighting else torch.ones(batch_size, 5, device=device)
        loss = huber_loss_log(y_pred, targets, mask, delta=cfg.train.huber_delta, weights=batch_weights)
        total_loss += loss.item()
        n_batches += 1
        
        # Collect predictions
        all_preds_log.append(y_pred.cpu().numpy())
        all_targets_log.append(targets.cpu().numpy())
        all_c_hat.append(c_hat.cpu().numpy())
        
        # Target intensity
        c_target = (targets + n_track_vec.unsqueeze(1) * torch.log(distances / r0)).mean(dim=1)
        all_c_target.append(c_target.cpu().numpy())
        all_distances.append(distances.cpu().numpy())
        all_tracks.append(tracks.cpu().numpy())
        all_event_ids.extend(list(event_ids))        # SAVEPRED addition
        all_epsilon.append(epsilon_hat.cpu().numpy()) # SAVEPRED addition
    
    preds_log = np.vstack(all_preds_log)
    targets_log = np.vstack(all_targets_log)
    c_hat_all = np.hstack(all_c_hat)
    c_target_all = np.hstack(all_c_target)
    distances_all = np.vstack(all_distances)
    tracks_all = np.hstack(all_tracks)
    event_ids_arr = np.array(all_event_ids)  # SAVEPRED addition
    epsilon_arr   = np.vstack(all_epsilon)   # SAVEPRED addition
    
    # Clamp for stability
    preds_log = np.clip(preds_log, cfg.train.pred_log_clamp_min, cfg.train.pred_log_clamp_max)
    
    # Convert to linear
    preds_pgv = np.exp(preds_log)
    targets_pgv = np.exp(targets_log)
    
    # Combined metrics
    rmse_pgv = np.sqrt(np.mean((preds_pgv - targets_pgv) ** 2))
    mae_pgv = np.mean(np.abs(preds_pgv - targets_pgv))
    rmse_log = np.sqrt(np.mean((preds_log - targets_log) ** 2))
    mae_log = np.mean(np.abs(preds_log - targets_log))
    
    ss_res = np.sum((targets_log - preds_log) ** 2)
    ss_tot = np.sum((targets_log - targets_log.mean()) ** 2)
    r2_log = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    ss_res_pgv = np.sum((targets_pgv - preds_pgv) ** 2)
    ss_tot_pgv = np.sum((targets_pgv - targets_pgv.mean()) ** 2)
    r2_pgv = 1.0 - (ss_res_pgv / ss_tot_pgv) if ss_tot_pgv > 0 else 0.0
    
    metrics = {
        "loss": total_loss / n_batches if n_batches > 0 else 0.0,
        "rmse_pgv": rmse_pgv,
        "mae_pgv": mae_pgv,
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "r2_log": r2_log,
        "r2_pgv": r2_pgv,
    }
    
    # Per-track
    for track_id in [1, 2]:
        mask = tracks_all == track_id
        if mask.any():
            pred_t = preds_pgv[mask]
            tgt_t = targets_pgv[mask]
            pred_log_t = preds_log[mask]
            tgt_log_t = targets_log[mask]
            
            metrics[f"t{track_id}_rmse_pgv"] = np.sqrt(np.mean((pred_t - tgt_t) ** 2))
            metrics[f"t{track_id}_mae_pgv"] = np.mean(np.abs(pred_t - tgt_t))
            metrics[f"t{track_id}_rmse_log"] = np.sqrt(np.mean((pred_log_t - tgt_log_t) ** 2))
            metrics[f"t{track_id}_mae_log"] = np.mean(np.abs(pred_log_t - tgt_log_t))
            ss_res_t = np.sum((tgt_log_t - pred_log_t) ** 2)
            ss_tot_t = np.sum((tgt_log_t - tgt_log_t.mean()) ** 2)
            metrics[f"t{track_id}_r2_log"] = 1.0 - (ss_res_t / ss_tot_t) if ss_tot_t > 0 else 0.0
    
    # SAVEPRED: return extra arrays needed for predictions.parquet
    return metrics, preds_log, targets_log, c_hat_all, c_target_all, distances_all, tracks_all, event_ids_arr, epsilon_arr


# ==================================================================================
# MAIN
# ==================================================================================

def main():
    parser = argparse.ArgumentParser(description="Train curve-prior + residual model")
    parser.add_argument("--variant", type=str, default="P1", choices=["P1", "P2", "P3", "P4"])
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--n_mode", type=str, default="fit_corrected", 
                        choices=["fit_corrected", "fixed_oracle", "fit_current_old"])
    parser.add_argument("--lambda_residual", type=float, default=0.0)
    args = parser.parse_args()
    
    cfg = get_variant_config(args.variant, n_mode=args.n_mode)
    if args.epochs > 0:
        cfg.train.epochs = args.epochs
    if args.lambda_residual > 0:
        cfg.model.lambda_residual = args.lambda_residual
    
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu")
    
    print("\n" + "=" * 80)
    print(f"TRACK-CONDITIONED CURVE-PRIOR + RESIDUAL CNN — Variant {args.variant}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Epochs: {cfg.train.epochs}")
    print(f"LR: {cfg.train.learning_rate}")
    print(f"n_mode: {cfg.train.n_mode}")
    print(f"lambda_residual: {cfg.model.lambda_residual}")
    if cfg.model.use_pgv_weighting:
        print(f"MP4_weight: {cfg.model.mp4_weight}")
    
    # Load data
    df_sensor, waveforms, event_map = load_linec_data(cfg)
    
    # Build dataset
    dataset, event_df = build_event_level_dataset(df_sensor, waveforms, event_map, cfg)
    print(f"Dataset: {len(dataset)} events")
    
    # Splits
    train_idx, val_idx, test_idx = make_event_splits(
        len(dataset),
        cfg.data.train_fraction,
        cfg.data.val_fraction,
        cfg.data.test_fraction,
        cfg.data.seed_split,
    )
    print(f"Train: {len(train_idx)} events")
    print(f"Val:   {len(val_idx)} events")
    print(f"Test:  {len(test_idx)} events")
    
    # Fit or load attenuation exponents based on n_mode
    train_targets = dataset.targets[train_idx]
    train_distances = dataset.distances[train_idx]
    train_tracks = dataset.tracks[train_idx]
    
    if cfg.train.n_mode == "fit_corrected":
        n_track1, n_track2 = fit_attenuation_exponents_corrected(
            train_targets, train_distances, train_tracks, cfg.features.r0
        )
        n_mode_label = "fit_corrected"
    elif cfg.train.n_mode == "fixed_oracle":
        n_track1 = cfg.features.n_track1_init
        n_track2 = cfg.features.n_track2_init
        print(f"\nUsing fixed oracle n values:")
        print(f"  Track 1: {n_track1:.4f}")
        print(f"  Track 2: {n_track2:.4f}")
        n_mode_label = "fixed_oracle"
    elif cfg.train.n_mode == "fit_current_old":
        # Legacy method for diagnostic comparison only
        n_track1, n_track2 = fit_attenuation_exponents(
            train_targets, train_distances, train_tracks, cfg.features.r0
        )
        n_mode_label = "fit_current_old (BROKEN - DIAGNOSTIC ONLY)"
    else:
        raise ValueError(f"Unknown n_mode: {cfg.train.n_mode}")
    
    # Data loaders
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True, 
                              num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory)
    test_loader = DataLoader(test_set, batch_size=cfg.train.batch_size, shuffle=False,
                             num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory)
    
    # Model
    model = CurvePriorCNN2D(
        n_metadata=dataset.metadata.shape[1],
        conv_channels=cfg.model.conv_channels,
        kernel_ch=cfg.model.kernel_ch,
        kernel_time=cfg.model.kernel_time,
        stride_ch=cfg.model.stride_ch,
        stride_time=cfg.model.stride_time,
        use_batchnorm=cfg.model.use_batchnorm,
        conv_dropout=cfg.model.conv_dropout,
        metadata_hidden=cfg.model.metadata_hidden,
        metadata_dropout=cfg.model.metadata_dropout,
        intensity_hidden=cfg.model.intensity_hidden,
        intensity_dropout=cfg.model.intensity_dropout,
        residual_hidden=cfg.model.residual_hidden,
        residual_dropout=cfg.model.residual_dropout,
        n_outputs=cfg.model.n_outputs,
        activation=cfg.model.activation,
    ).to(device)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    scheduler = StepLR(optimizer, step_size=cfg.train.learning_rate_decay_steps, gamma=cfg.train.learning_rate_decay)
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    best_val_rmse_log = float("inf")
    best_epoch = -1
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(cfg.train.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, cfg, device, n_track1, n_track2,
            cfg.features.r0, first_batch_debug=(epoch == 0)
        )
        
        val_metrics, _, _, _, _, _, _, _, _ = evaluate(
            model, val_loader, device, cfg, n_track1, n_track2, cfg.features.r0
        )
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}  val_rmse_log={val_metrics['rmse_log']:.4f}  val_r2={val_metrics['r2_log']:.4f}")
        
        if val_metrics["rmse_log"] < best_val_rmse_log:
            best_val_rmse_log = val_metrics["rmse_log"]
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            if epoch > 0:
                print(f"  ✓ Best epoch: {best_epoch} (val_rmse_log={best_val_rmse_log:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= cfg.train.patience_early_stopping:
            print(f"Early stopping at epoch {epoch+1} (best was epoch {best_epoch})")
            break
        
        scheduler.step()
    
    # Restore best model
    print(f"\n[CHECKPOINT] Restoring best model from epoch {best_epoch}")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION (using best checkpoint)")
    print("=" * 80)
    
    train_metrics, train_preds_log, train_targets_log, train_c_hat, train_c_target, _, _, _, _ = evaluate(
        model, train_loader, device, cfg, n_track1, n_track2, cfg.features.r0
    )
    val_metrics, val_preds_log, val_targets_log, val_c_hat, val_c_target, _, _, _, _ = evaluate(
        model, val_loader, device, cfg, n_track1, n_track2, cfg.features.r0
    )
    # SAVEPRED: unpack extra arrays for prediction saving
    (test_metrics, test_preds_log, test_targets_log, test_c_hat, test_c_target,
     test_distances, test_tracks, test_event_ids, test_epsilon) = evaluate(
        model, test_loader, device, cfg, n_track1, n_track2, cfg.features.r0
    )
    
    print("\nTEST METRICS:")
    print(f"  RMSE(PGV): {test_metrics['rmse_pgv']:.4f} mm/s")
    print(f"  RMSE(log): {test_metrics['rmse_log']:.4f}")
    print(f"  R²(log):   {test_metrics['r2_log']:.4f}")
    
    # SAVEPRED: use a different output tag so outputs don't overwrite originals
    output_dir = cfg.output.output_root / f"cnn_curveprior_p3_savepred_linec_v001_v{args.variant}_{cfg.train.n_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    (output_dir / "config_snapshot.json").write_text(json.dumps({
        "variant": args.variant,
        "n_mode": n_mode_label,
        "n_track1": n_track1,
        "n_track2": n_track2,
        "lambda_residual": cfg.model.lambda_residual,
        "mp4_weight": cfg.model.mp4_weight if cfg.model.use_pgv_weighting else 1.0,
        "model": cfg.model.__dict__,
        "train": cfg.train.__dict__,
    }, indent=2, default=str))
    
    # Save metrics
    (output_dir / "metrics.json").write_text(json.dumps({
        "best_epoch": best_epoch,
        "best_val_rmse_log": float(best_val_rmse_log),
        "train": {k: float(v) for k, v in train_metrics.items()},
        "val": {k: float(v) for k, v in val_metrics.items()},
        "test": {k: float(v) for k, v in test_metrics.items()},
    }, indent=2))
    
    # SAVEPRED: save model checkpoint
    torch.save(best_model_state, output_dir / "model.pth")
    print(f"Saved model.pth")

    # SAVEPRED: build predictions.parquet (test split, flattened to per-sensor rows)
    sensor_names = cfg.data.line_c_sensors  # ["MP4","MP8","MP10","MP1","MP2"]
    _dist_map = {
        1: {"MP4": 2.5,  "MP8": 4.0,  "MP10": 8.0,  "MP1": 16.0, "MP2": 23.0},
        2: {"MP4": 6.5,  "MP8": 8.0,  "MP10": 12.0, "MP1": 20.0, "MP2": 27.0},
    }
    _rows = []
    for ev_i in range(len(test_event_ids)):
        track = int(test_tracks[ev_i])
        n_used = n_track1 if track == 1 else n_track2
        for s_j, sensor in enumerate(sensor_names):
            dist = _dist_map[track][sensor]
            _rows.append({
                "split":      "test",
                "event_id":   str(test_event_ids[ev_i]),
                "sensor":     sensor,
                "track":      track,
                "distance":   float(dist),
                "target_log": float(test_targets_log[ev_i, s_j]),
                "pred_log":   float(test_preds_log[ev_i, s_j]),
                "target_pgv": float(np.exp(test_targets_log[ev_i, s_j])),
                "pred_pgv":   float(np.exp(test_preds_log[ev_i, s_j])),
                "c_hat":      float(test_c_hat[ev_i]),
                "c_target":   float(test_c_target[ev_i]),
                "epsilon":    float(test_epsilon[ev_i, s_j]),
                "n_used":     float(n_used),
            })
    pred_df = pd.DataFrame(_rows)
    pred_df.to_parquet(output_dir / "predictions.parquet", index=False)
    print(f"Saved predictions.parquet ({len(pred_df)} rows)")

    # SAVEPRED: per-sensor metrics CSV
    _ps_rows = []
    for s_j, sensor in enumerate(sensor_names):
        p_s = test_preds_log[:, s_j]
        t_s = test_targets_log[:, s_j]
        _ps_rows.append({
            "sensor":   sensor,
            "count":    int(len(p_s)),
            "rmse_log": float(np.sqrt(np.mean((p_s - t_s) ** 2))),
            "rmse_pgv": float(np.sqrt(np.mean((np.exp(p_s) - np.exp(t_s)) ** 2))),
            "bias_pgv": float(np.mean(np.exp(p_s) - np.exp(t_s))),
            "bias_log": float(np.mean(p_s - t_s)),
        })
    pd.DataFrame(_ps_rows).to_csv(output_dir / "per_sensor_metrics.csv", index=False)
    print(f"Saved per_sensor_metrics.csv")

    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
