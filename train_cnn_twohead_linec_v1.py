"""Train track-conditioned two-head multi-output CNN on line-C subset.

Event-level prediction:
  FO waveform (21 channels) + metadata → 5 log-PGV values at track-specific distances

Supports variants:
  A — direct two-head, unweighted
  B — direct two-head, MP4-weighted
  C — monotonic two-head, unweighted
  D — monotonic two-head, MP4-weighted
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
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ml.cnn.config_cnn_twohead_linec_v1 import Config, get_variant_config
from src.ml.cnn.cnn_twohead_linec_utils import (
    TwoHeadDataset,
    TrackConditionedCNN2D,
    mse_loss_log,
    huber_loss_log,
    monotonicity_penalty_loss,
    compute_sample_weights,
)
from src.utils.geometry_utils import apply_corrected_distances


# ==================================================================================
# HELPERS
# ==================================================================================

def _find_latest_waveform_build(cfg) -> Path:
    """Auto-discover latest waveform build matching glob."""
    if cfg.data.waveform_build_dir:
        return Path(cfg.data.waveform_build_dir)
    root = Path(cfg.data.waveform_root)
    # Try to find 21-channel line-C build first
    builds = sorted(root.glob(cfg.data.waveform_glob), key=lambda p: p.name)
    if builds:
        return builds[-1]
    # Fallback: use ch51 and slice locally
    builds_ch51 = sorted(root.glob("holten_waveform_v003_ch51_*"), key=lambda p: p.name)
    if builds_ch51:
        return builds_ch51[-1]
    raise FileNotFoundError(f"No waveform builds found in {root}")


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
    """Load line-C side -1 subset with waveforms.
    
    Returns:
      df_sensor: sensor-level DataFrame with cleaned data
      waveforms_event: (N_events, 21, 7500) waveform array
      event_map: dict mapping event_id → waveform row index
    """
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
    
    # Clean: remove NaNs and invalid targets
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
    
    # If ch51 build (51 channels), slice to 21-channel local window for line C
    # Line C center = 1194; window = [1194-10, 1194+10] = 21 channels
    # ch51 spans [1184, 1204], so slice is [0:21] or middle 21 of 51
    if waveforms.shape[1] == 51:
        # ch51: keep center 21 channels (indices 15:36 of 51)
        waveforms = waveforms[:, 15:36, :]
        print(f"Sliced from ch51 to ch21 (local line-C window)")
    elif waveforms.shape[1] != 21:
        raise ValueError(f"Unexpected waveform shape: {waveforms.shape}. Expected (N, 21, T) or (N, 51, T)")
    
    waveforms *= np.float32(cfg.data.waveform_scale)
    print(f"Waveforms: {waveforms.shape} {waveforms.dtype}")
    
    # Load event index
    index_df = pd.read_parquet(wave_dir / "event_index.parquet")
    index_ok = index_df[index_df["build_status"] == "ok"].copy()
    index_ok["event_id"] = index_ok["event_id"].astype(str)
    event_map = dict(zip(index_ok["event_id"], index_ok["waveform_row_idx"].astype(int)))
    print(f"Events with waveforms: {len(event_map):,}")
    
    # Map events in df_linec
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
) -> Tuple[TwoHeadDataset, pd.DataFrame]:
    """Convert sensor-level DataFrame to event-level dataset.
    
    For each event:
      - Gather the 5 sensors (MP4, MP8, MP10, MP1, MP2)
      - Extract their log-PGV values
      - Determine track_number
      - Load waveform
      - Extract metadata features
      - Package as event
    
    Returns:
      dataset: TwoHeadDataset
      event_df: (N_events,) with event_id, track, etc.
    """
    print("\n" + "=" * 80)
    print("BUILDING EVENT-LEVEL DATASET")
    print("=" * 80)
    
    events_list = []
    waveforms_list = []
    targets_list = []
    tracks_list = []
    event_ids_list = []
    metadata_list = []
    
    sensors = cfg.data.line_c_sensors  # [MP4, MP8, MP10, MP1, MP2]
    
    for event_id in df_sensor[cfg.data.event_col].unique():
        df_ev = df_sensor[df_sensor[cfg.data.event_col] == event_id]
        
        # Must have all 5 sensors, same track
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
        wf = waveforms[wf_idx]  # (21, 7500)
        
        # Extract target vector: log(PGV) in sensor order
        pgv_vec = []
        for sensor in sensors:
            df_sens = df_ev[df_ev[cfg.data.sensor_col] == sensor]
            if len(df_sens) != 1:
                break
            pgv = df_sens[cfg.data.pgv_col].values[0]
            pgv_vec.append(np.log(np.clip(pgv, 1e-6, None)))
        else:
            # All 5 sensors found
            waveforms_list.append(wf)
            targets_list.append(pgv_vec)
            tracks_list.append(track)
            event_ids_list.append(event_id)
            
            # Extract metadata from first sensor row
            row = df_ev.iloc[0]
            
            # Build metadata vector
            meta_vec = []
            
            # Train speed
            speed = row.get(cfg.features.speed_col, np.nan)
            if not np.isfinite(speed):
                speed = 0.0
            meta_vec.append(speed)
            
            # Train type code
            train_type = row.get("train_type_code", -1)
            if not np.isfinite(train_type):
                train_type = -1
            meta_vec.append(float(train_type))
            
            # Track number (normalized to [-1, 1] or [0, 1])
            track_norm = float(track) / 2.0
            meta_vec.append(track_norm)
            
            metadata_list.append(meta_vec)
            
            # Store event info for later
            events_list.append({
                "event_id": event_id,
                "track": track,
                "speed": speed,
                "train_type": train_type,
            })
    
    print(f"Event-level samples: {len(waveforms_list):,}")
    
    # Build metadata array
    metadata_arr = np.array(metadata_list, dtype=np.float32)
    print(f"Metadata shape: {metadata_arr.shape}")
    
    waveforms_arr = np.array(waveforms_list, dtype=np.float32)
    targets_arr = np.array(targets_list, dtype=np.float32)
    tracks_arr = np.array(tracks_list, dtype=np.int64)
    event_ids_arr = np.array(event_ids_list)
    
    dataset = TwoHeadDataset(
        waveforms=waveforms_arr,
        metadata=metadata_arr,
        targets=targets_arr,
        tracks=tracks_arr,
        event_ids=event_ids_arr,
    )
    
    event_df = pd.DataFrame(events_list)
    
    return dataset, event_df


# ==================================================================================
# TRAIN / VAL / TEST SPLITS
# ==================================================================================

def make_event_splits(
    event_ids: np.ndarray,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Event-level train/val/test split."""
    np.random.seed(seed)
    n_events = len(event_ids)
    
    idx = np.arange(n_events)
    np.random.shuffle(idx)
    
    n_train = int(n_events * train_frac)
    n_val = int(n_events * val_frac)
    
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val:]
    
    return train_idx, val_idx, test_idx


# ==================================================================================
# MODEL TRAINING
# ==================================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: str,
    device: str,
    weights: Optional[np.ndarray] = None,
    mono_weight: float = 0.0,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in train_loader:
        wf, meta, targets, tracks, event_ids = batch
        wf = wf.to(device)
        meta = meta.to(device)
        targets = targets.to(device)
        tracks = tracks.to(device)
        
        # Forward
        logits_t1, logits_t2, mask_t1, mask_t2 = model(wf, meta, tracks)
        
        # Loss
        if loss_fn == "mse_log":
            loss_t1 = mse_loss_log(logits_t1, targets, mask_t1, weights=None)
            loss_t2 = mse_loss_log(logits_t2, targets, mask_t2, weights=None)
        else:  # huber_log
            loss_t1 = huber_loss_log(logits_t1, targets, mask_t1, delta=0.5)
            loss_t2 = huber_loss_log(logits_t2, targets, mask_t2, delta=0.5)
        
        loss = loss_t1 + loss_t2
        
        # Monotonicity penalty
        if mono_weight > 0:
            mono_loss = monotonicity_penalty_loss(logits_t1, mask_t1) + \
                        monotonicity_penalty_loss(logits_t2, mask_t2)
            loss = loss + mono_weight * mono_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    loss_fn: str,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on a split.
    
    Returns:
      metrics: dict with RMSE, MAE, R² (combined and per-track)
      preds_log: (n_samples, 5) log-PGV predictions
      targets_log: (n_samples, 5) log-PGV ground truth
      tracks: (n_samples,) track numbers
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_tracks = []
    total_loss = 0.0
    n_batches = 0
    
    for batch in data_loader:
        wf, meta, targets, tracks, event_ids = batch
        wf = wf.to(device)
        meta = meta.to(device)
        targets = targets.to(device)
        
        logits_t1, logits_t2, mask_t1, mask_t2 = model(wf, meta, tracks)
        
        # Combine predictions by track
        preds = torch.zeros_like(targets)
        preds[mask_t1] = logits_t1[mask_t1]
        preds[mask_t2] = logits_t2[mask_t2]
        
        # Loss
        if loss_fn == "mse_log":
            loss_t1 = mse_loss_log(logits_t1, targets, mask_t1)
            loss_t2 = mse_loss_log(logits_t2, targets, mask_t2)
        else:
            loss_t1 = huber_loss_log(logits_t1, targets, mask_t1, delta=0.5)
            loss_t2 = huber_loss_log(logits_t2, targets, mask_t2, delta=0.5)
        
        loss = loss_t1 + loss_t2
        total_loss += loss.item()
        n_batches += 1
        
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_tracks.append(tracks.cpu().numpy())
    
    preds_log = np.vstack(all_preds)
    targets_log = np.vstack(all_targets)
    tracks_arr = np.hstack(all_tracks)
    
    # Compute metrics
    # Convert from log to linear for RMSE/MAE in PGV space
    preds_pgv = np.exp(np.clip(preds_log, -30, 30))
    targets_pgv = np.exp(targets_log)
    
    # Combined metrics
    rmse_pgv = np.sqrt(np.mean((preds_pgv - targets_pgv) ** 2))
    mae_pgv = np.mean(np.abs(preds_pgv - targets_pgv))
    rmse_log = np.sqrt(np.mean((preds_log - targets_log) ** 2))
    mae_log = np.mean(np.abs(preds_log - targets_log))
    
    # R² on log scale
    ss_res = np.sum((targets_log - preds_log) ** 2)
    ss_tot = np.sum((targets_log - targets_log.mean()) ** 2)
    r2_log = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    metrics = {
        "loss": total_loss / n_batches if n_batches > 0 else 0.0,
        "rmse_pgv": rmse_pgv,
        "mae_pgv": mae_pgv,
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "r2_log": r2_log,
    }
    
    # Per-track metrics
    for track_id in [1, 2]:
        mask = tracks_arr == track_id
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
    
    return metrics, preds_log, targets_log, tracks_arr


# ==================================================================================
# MAIN
# ==================================================================================

def main():
    parser = argparse.ArgumentParser(description="Train track-conditioned two-head CNN")
    parser.add_argument("--variant", type=str, default="A", choices=["A", "B", "C", "D"])
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--epochs", type=int, default=0, help="Override epochs (0 = use config)")
    args = parser.parse_args()
    
    # Load config
    cfg = get_variant_config(args.variant)
    if args.epochs > 0:
        cfg.train.epochs = args.epochs
    
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu")
    
    print("\n" + "=" * 80)
    print(f"TRACK-CONDITIONED TWO-HEAD MULTI-OUTPUT CNN — Variant {args.variant}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Epochs: {cfg.train.epochs}")
    print(f"Batch size: {cfg.train.batch_size}")
    print(f"LR: {cfg.train.learning_rate}")
    
    # Load data
    df_sensor, waveforms, event_map = load_linec_data(cfg)
    
    # Build event-level dataset
    dataset, event_df = build_event_level_dataset(df_sensor, waveforms, event_map, cfg)
    print(f"Dataset: {len(dataset)} events")
    
    # Splits
    train_idx, val_idx, test_idx = make_event_splits(
        dataset.event_ids,
        cfg.data.train_fraction,
        cfg.data.val_fraction,
        cfg.data.test_fraction,
        cfg.data.seed_split,
    )
    
    print(f"Train: {len(train_idx)} events")
    print(f"Val:   {len(val_idx)} events")
    print(f"Test:  {len(test_idx)} events")
    
    # Data loaders
    from torch.utils.data import Subset
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
    model = TrackConditionedCNN2D(
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
        head_hidden=cfg.model.head_hidden,
        head_dropout=cfg.model.head_dropout,
        n_outputs=cfg.model.n_outputs,
        activation=cfg.model.activation,
    ).to(device)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    if cfg.train.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=cfg.train.learning_rate, 
                        weight_decay=cfg.train.weight_decay)
    elif cfg.train.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=cfg.train.learning_rate,
                         weight_decay=cfg.train.weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=cfg.train.learning_rate,
                       weight_decay=cfg.train.weight_decay)
    
    scheduler = StepLR(optimizer, step_size=cfg.train.learning_rate_decay_steps,
                      gamma=cfg.train.learning_rate_decay)
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    history = {"train_loss": [], "val_loss": [], "val_rmse": []}
    best_val_rmse = float("inf")
    patience_counter = 0
    
    for epoch in range(cfg.train.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, cfg.train.loss_fn, device,
            weights=None,
            mono_weight=cfg.model.monotonic_weight if cfg.model.enforce_monotonicity else 0.0,
        )
        
        val_metrics, _, _, _ = evaluate(model, val_loader, device, cfg.train.loss_fn)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_rmse"].append(val_metrics["rmse_pgv"])
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  val_rmse={val_metrics['rmse_pgv']:.4f}  val_r2={val_metrics['r2_log']:.4f}")
        
        # Early stopping
        if val_metrics["rmse_pgv"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse_pgv"]
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= cfg.train.patience_early_stopping:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        scheduler.step()
    
    # Evaluate
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    train_metrics, train_preds_log, train_targets_log, train_tracks = evaluate(
        model, train_loader, device, cfg.train.loss_fn
    )
    val_metrics, val_preds_log, val_targets_log, val_tracks = evaluate(
        model, val_loader, device, cfg.train.loss_fn
    )
    test_metrics, test_preds_log, test_targets_log, test_tracks = evaluate(
        model, test_loader, device, cfg.train.loss_fn
    )
    
    print("\nTRAIN:")
    print(f"  RMSE(PGV): {train_metrics['rmse_pgv']:.4f} mm/s")
    print(f"  MAE(PGV):  {train_metrics['mae_pgv']:.4f} mm/s")
    print(f"  RMSE(log): {train_metrics['rmse_log']:.4f}")
    print(f"  R²(log):   {train_metrics['r2_log']:.4f}")
    
    print("\nVALIDATION:")
    print(f"  RMSE(PGV): {val_metrics['rmse_pgv']:.4f} mm/s")
    print(f"  MAE(PGV):  {val_metrics['mae_pgv']:.4f} mm/s")
    print(f"  RMSE(log): {val_metrics['rmse_log']:.4f}")
    print(f"  R²(log):   {val_metrics['r2_log']:.4f}")
    
    print("\nTEST:")
    print(f"  RMSE(PGV): {test_metrics['rmse_pgv']:.4f} mm/s")
    print(f"  MAE(PGV):  {test_metrics['mae_pgv']:.4f} mm/s")
    print(f"  RMSE(log): {test_metrics['rmse_log']:.4f}")
    print(f"  R²(log):   {test_metrics['r2_log']:.4f}")
    
    # Per-track
    for track_id in [1, 2]:
        print(f"\nTEST TRACK {track_id}:")
        print(f"  RMSE(PGV): {test_metrics.get(f't{track_id}_rmse_pgv', np.nan):.4f} mm/s")
        print(f"  MAE(PGV):  {test_metrics.get(f't{track_id}_mae_pgv', np.nan):.4f} mm/s")
        print(f"  RMSE(log): {test_metrics.get(f't{track_id}_rmse_log', np.nan):.4f}")
        print(f"  R²(log):   {test_metrics.get(f't{track_id}_r2_log', np.nan):.4f}")
    
    # Save output
    output_dir = cfg.output.output_root / f"{cfg.output.output_tag}_v{args.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    (output_dir / "config.json").write_text(json.dumps({
        "variant": args.variant,
        "model": cfg.model.__dict__,
        "train": cfg.train.__dict__,
    }, indent=2, default=str))
    
    # Save metrics
    (output_dir / "metrics.json").write_text(json.dumps({
        "train": {k: float(v) for k, v in train_metrics.items()},
        "val": {k: float(v) for k, v in val_metrics.items()},
        "test": {k: float(v) for k, v in test_metrics.items()},
    }, indent=2))
    
    # Save predictions
    pred_df = pd.DataFrame({
        "split": ["test"] * len(test_preds_log),
        "track": test_tracks,
    })
    for i in range(5):
        pred_df[f"pred_log_{i}"] = test_preds_log[:, i]
        pred_df[f"true_log_{i}"] = test_targets_log[:, i]
    pred_df.to_parquet(output_dir / "predictions.parquet")
    
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
