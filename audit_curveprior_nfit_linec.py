"""Audit attenuation exponent fitting in curve-prior model.

Compare three fitting approaches:
  A. Event-intercept corrected (proposed fix)
  B. Current curve-prior method
  C. Oracle historical method (if available)

Goal: Identify why current method produces n ≈ 0.24/0.51 instead of ≈1.08/1.33
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.ml.cnn.config_cnn_curveprior_linec_v1 import get_variant_config
from train_cnn_curveprior_linec_v1 import (
    load_linec_data,
    build_event_level_dataset,
    make_event_splits,
)

# ============================================================================
# CONSTANTS
# ============================================================================

R_TRACK1 = np.array([2.5, 4.0, 8.0, 16.0, 23.0], dtype=np.float32)
R_TRACK2 = np.array([6.5, 8.0, 12.0, 20.0, 27.0], dtype=np.float32)
R0 = 10.0

ORACLE_N_TRACK1 = 1.0777
ORACLE_N_TRACK2 = 1.3300

# ============================================================================
# METHOD A: Event-Intercept Corrected Fitting
# ============================================================================

def fit_n_corrected(
    targets_log: np.ndarray,  # (n_events, 5)
    distances: np.ndarray,     # (n_events, 5)
    tracks: np.ndarray,        # (n_events,)
    r0: float = 10.0,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Fit track-specific n using event-intercept corrected formula.
    
    Model: y_ij = c_i - n * log(r_j / r0)
    
    For each track:
      1. Compute design matrix (centered by event mean)
      2. Use all complete events to fit global n
      3. Also compute per-event n_i for diagnostics
    
    Returns: (n_track1, n_track2, n_per_event_t1, n_per_event_t2)
    """
    print("\n[METHOD A] EVENT-INTERCEPT CORRECTED FITTING")
    print("=" * 80)
    
    n_track1_global = None
    n_track2_global = None
    n_per_event = {1: [], 2: []}
    
    for track_id in [1, 2]:
        mask = tracks == track_id
        if not mask.any():
            print(f"  Track {track_id}: No events")
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
        
        # Per-event n_i
        for i in range(n_events):
            x_c_i = x_c[i]  # (5,)
            y_c_i = y_c[i]  # (5,)
            
            num_i = np.sum(x_c_i * y_c_i)
            den_i = np.sum(x_c_i ** 2)
            
            beta_i = num_i / den_i if den_i > 1e-10 else 0.0
            n_i = -beta_i
            n_per_event[track_id].append(n_i)
        
        # Stats
        n_per_event_arr = np.array(n_per_event[track_id])
        
        print(f"\n  Track {track_id}:")
        print(f"    Global n_fit:     {n_global:8.4f}")
        print(f"    Mean(n_i):        {n_per_event_arr.mean():8.4f}")
        print(f"    Median(n_i):      {np.median(n_per_event_arr):8.4f}")
        print(f"    Std(n_i):         {n_per_event_arr.std():8.4f}")
        print(f"    Min(n_i):         {n_per_event_arr.min():8.4f}")
        print(f"    Max(n_i):         {n_per_event_arr.max():8.4f}")
        print(f"    Complete events:  {n_events}")
        print(f"    Total sensors:    {n_events * 5}")
        
        if track_id == 1:
            n_track1_global = n_global
        else:
            n_track2_global = n_global
    
    return (
        n_track1_global or ORACLE_N_TRACK1,
        n_track2_global or ORACLE_N_TRACK2,
        np.array(n_per_event.get(1, [])),
        np.array(n_per_event.get(2, [])),
    )


# ============================================================================
# METHOD B: Current Curve-Prior Fitting (Reproduce Issue)
# ============================================================================

def fit_n_current_old(
    targets_log: np.ndarray,  # (n_events, 5)
    distances: np.ndarray,     # (n_events, 5)
    tracks: np.ndarray,        # (n_events,)
    r0: float = 10.0,
) -> Tuple[float, float]:
    """Reproduce the current curve-prior fitting method.
    
    This is the method that produced suspicious values: 0.24, 0.51
    """
    print("\n[METHOD B] CURRENT CURVE-PRIOR FITTING (Old Implementation)")
    print("=" * 80)
    
    n_track1 = []
    n_track2 = []
    
    for track_id in [1, 2]:
        mask = tracks == track_id
        if not mask.any():
            print(f"  Track {track_id}: No events")
            continue
        
        targets_t = targets_log[mask]  # (n_events_t, 5)
        distances_t = distances[mask]  # (n_events_t, 5)
        
        # Set up least squares problem using per-sensor slope estimates
        n_estimates = []
        for i in range(len(targets_t)):
            for j in range(len(targets_t[i]) - 1):
                dy = targets_t[i, j+1] - targets_t[i, j]
                dr = np.log(distances_t[i, j+1] / distances_t[i, j])
                if abs(dr) > 1e-6:
                    n_est = dy / dr
                    if n_est > 0:
                        n_estimates.append(n_est)
        
        # Median n
        if n_estimates:
            n_fit = float(np.median(n_estimates))
        else:
            n_fit = ORACLE_N_TRACK1 if track_id == 1 else ORACLE_N_TRACK2
        
        print(f"\n  Track {track_id}:")
        print(f"    Slope estimates:  {len(n_estimates)}")
        print(f"    Median n:         {n_fit:8.4f}")
        if n_estimates:
            n_arr = np.array(n_estimates)
            print(f"    Mean:             {n_arr.mean():8.4f}")
            print(f"    Std:              {n_arr.std():8.4f}")
            print(f"    Min:              {n_arr.min():8.4f}")
            print(f"    Max:              {n_arr.max():8.4f}")
        
        if track_id == 1:
            n_track1.append(n_fit)
        else:
            n_track2.append(n_fit)
    
    return float(n_track1[0]) if n_track1 else ORACLE_N_TRACK1, \
           float(n_track2[0]) if n_track2 else ORACLE_N_TRACK2


# ============================================================================
# DIAGNOSTIC: Print Example Events
# ============================================================================

def print_example_events(
    targets_log: np.ndarray,
    distances: np.ndarray,
    tracks: np.ndarray,
    n_examples: int = 3,
    r0: float = 10.0,
):
    """Print detailed diagnostics for example events."""
    print("\n" + "=" * 80)
    print("EXAMPLE EVENTS (5 per track)")
    print("=" * 80)
    
    sensor_names = ["MP4", "MP8", "MP10", "MP1", "MP2"]
    
    for track_id in [1, 2]:
        print(f"\nTRACK {track_id}:")
        print(f"  Distance vector r: {[R_TRACK1, R_TRACK2][track_id-1]}")
        print(f"  log(r/r0):         {np.log([R_TRACK1, R_TRACK2][track_id-1] / r0)}")
        
        mask = tracks == track_id
        event_indices = np.where(mask)[0][:n_examples]
        
        for event_num, event_idx in enumerate(event_indices, 1):
            y_log = targets_log[event_idx]
            r = distances[event_idx]
            
            print(f"\n  Event {event_num}:")
            print(f"    Sensor    | r(m)  | log(r/r0) | log-PGV | PGV(mm/s)")
            print(f"    " + "-" * 50)
            for j, sensor in enumerate(sensor_names):
                pgv = np.exp(y_log[j])
                print(f"    {sensor:8s} | {r[j]:5.1f} | {np.log(r[j]/r0):9.4f} | {y_log[j]:7.4f} | {pgv:9.4f}")
            
            # Per-event n estimate (corrected method)
            x = np.log(r / r0)
            x_c = x - x.mean()
            y_c = y_log - y_log.mean()
            
            num = np.sum(x_c * y_c)
            den = np.sum(x_c ** 2)
            beta = num / den if den > 1e-10 else 0.0
            n_i = -beta
            
            # Event intensity (average intercept)
            c_i = (y_log + n_i * x).mean()
            
            print(f"    Per-event n_i:    {n_i:8.4f}")
            print(f"    Intensity c_i:    {c_i:8.4f}")


# ============================================================================
# DIAGNOSTIC: Distance Verification
# ============================================================================

def verify_distances(dataset):
    """Verify distance vectors match expected physical values."""
    print("\n" + "=" * 80)
    print("DISTANCE VERIFICATION")
    print("=" * 80)
    
    for track_id in [1, 2]:
        mask = dataset.tracks == track_id
        if mask.any():
            r_sample = dataset.distances[mask][0]
            expected_r = R_TRACK1 if track_id == 1 else R_TRACK2
            
            print(f"\nTrack {track_id}:")
            print(f"  Expected r: {expected_r}")
            print(f"  Got r:      {r_sample}")
            print(f"  Match:      {np.allclose(r_sample, expected_r)}")


# ============================================================================
# MAIN AUDIT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Audit attenuation exponent fitting")
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("CURVE-PRIOR ATTENUATION EXPONENT FITTING AUDIT")
    print("=" * 80)
    print(f"Oracle reference:")
    print(f"  Track 1: n = {ORACLE_N_TRACK1}")
    print(f"  Track 2: n = {ORACLE_N_TRACK2}")
    
    # Load and prepare data
    cfg = get_variant_config("P1")
    print("\nLoading data...")
    df_sensor, waveforms, event_map = load_linec_data(cfg)
    
    print("\nBuilding event-level dataset...")
    dataset, event_df = build_event_level_dataset(df_sensor, waveforms, event_map, cfg)
    
    # Create splits (same as training)
    train_idx, val_idx, test_idx = make_event_splits(
        len(dataset), 0.65, 0.15, 0.20, seed=42
    )
    
    train_targets = dataset.targets[train_idx]
    train_distances = dataset.distances[train_idx]
    train_tracks = dataset.tracks[train_idx]
    
    print(f"\nTrain split: {len(train_idx)} events")
    
    # Verify distances
    verify_distances(dataset)
    
    # Run audits
    print("\n" + "=" * 80)
    print("FITTING COMPARISON")
    print("=" * 80)
    
    # Method A: Corrected
    n1_corrected, n2_corrected, n_per_t1_corrected, n_per_t2_corrected = fit_n_corrected(
        train_targets, train_distances, train_tracks, R0
    )
    
    # Method B: Current (old)
    n1_old, n2_old = fit_n_current_old(
        train_targets, train_distances, train_tracks, R0
    )
    
    # Print example events
    print_example_events(train_targets, train_distances, train_tracks, n_examples=3)
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("FITTING SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<30} {'Track 1 n':<12} {'Track 2 n':<12}")
    print("-" * 54)
    print(f"{'Oracle reference':<30} {ORACLE_N_TRACK1:<12.4f} {ORACLE_N_TRACK2:<12.4f}")
    print(f"{'Corrected (event-intercept)':<30} {n1_corrected:<12.4f} {n2_corrected:<12.4f}")
    print(f"{'Current/Old (median slopes)':<30} {n1_old:<12.4f} {n2_old:<12.4f}")
    
    # Percent differences
    print(f"\n{'Difference vs Oracle':<30} {'Track 1':<12} {'Track 2':<12}")
    print("-" * 54)
    pct_corrected_t1 = 100 * (n1_corrected - ORACLE_N_TRACK1) / ORACLE_N_TRACK1
    pct_corrected_t2 = 100 * (n2_corrected - ORACLE_N_TRACK2) / ORACLE_N_TRACK2
    print(f"{'Corrected':<30} {pct_corrected_t1:>10.1f}% {pct_corrected_t2:>10.1f}%")
    
    pct_old_t1 = 100 * (n1_old - ORACLE_N_TRACK1) / ORACLE_N_TRACK1
    pct_old_t2 = 100 * (n2_old - ORACLE_N_TRACK2) / ORACLE_N_TRACK2
    print(f"{'Current/Old':<30} {pct_old_t1:>10.1f}% {pct_old_t2:>10.1f}%")
    
    # Diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    if np.abs(pct_old_t1) > 50 or np.abs(pct_old_t2) > 50:
        print("\n[!] CRITICAL: Current method produces n values 50%+ different from oracle!")
        print("\nProbable causes:")
        print("  1. Per-sensor slope method (current) treats consecutive sensors as independent")
        print("  2. Per-sensor slopes: dy = y_j+1 - y_j, dr = log(r_j+1 / r_j)")
        print("  3. This is NOT the same as fitting: y = c - n*log(r/r0)")
        print("  4. Consecutive differences emphasize noise at short baselines")
        print("  5. Fitted n should use event intercepts (corrected method)")
    
    if np.abs(pct_corrected_t1) < 20 and np.abs(pct_corrected_t2) < 20:
        print("\n[OK] CORRECTED METHOD: Within 20% of oracle, acceptable accuracy.")
    else:
        print(f"\n[!] CORRECTED METHOD: {max(np.abs(pct_corrected_t1), np.abs(pct_corrected_t2)):.1f}% off oracle.")
        print("   Check if distance vectors or PGV data are as expected.")
    
    print("\nRECOMMENDATION:")
    print("  Use 'n_mode = \"fit_corrected\"' in next training run.")
    print("  Also test 'n_mode = \"fixed_oracle\"' for comparison.")


if __name__ == "__main__":
    main()
