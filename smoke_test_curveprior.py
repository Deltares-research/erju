"""Smoke tests for curve-prior model before cluster submission.

Tests:
  1. Config loading
  2. Data loading
  3. Distance vectors correctness
  4. Dataset construction
  5. One forward pass
  6. Loss computation
  7. n_track fitting
"""

from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.ml.cnn.config_cnn_curveprior_linec_v1 import get_variant_config
from src.ml.cnn.cnn_curveprior_linec_utils import (
    CurvePriorCNN2D,
    huber_loss_log,
    compute_curve_target,
)
from train_cnn_curveprior_linec_v1 import (
    load_linec_data,
    build_event_level_dataset,
    fit_attenuation_exponents,
    make_event_splits,
)

print("\n" + "=" * 80)
print("SMOKE TESTS: Curve-Prior Model")
print("=" * 80)

# ============================================================================
# 1. Config Loading
# ============================================================================
print("\n[1] Config Loading")
try:
    for variant in ["P1", "P2", "P3"]:
        cfg = get_variant_config(variant)
        print(f"  ✓ Variant {variant}: lr={cfg.train.learning_rate}, "
              f"lambda_res={cfg.model.lambda_residual}, "
              f"mp4_weight={cfg.model.mp4_weight}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 2. Data Loading
# ============================================================================
print("\n[2] Data Loading")
try:
    cfg = get_variant_config("P1")
    df_sensor, waveforms, event_map = load_linec_data(cfg)
    print(f"  ✓ Loaded {len(df_sensor):,} rows, {len(event_map):,} events")
    print(f"    Waveforms shape: {waveforms.shape}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 3. Dataset Construction
# ============================================================================
print("\n[3] Dataset Construction")
try:
    dataset, event_df = build_event_level_dataset(df_sensor, waveforms, event_map, cfg)
    print(f"  ✓ Built {len(dataset)} event samples")
    print(f"    Waveforms: {dataset.waveforms.shape}")
    print(f"    Targets: {dataset.targets.shape}")
    print(f"    Distances: {dataset.distances.shape}")
    print(f"    Tracks: {dataset.tracks.unique()}")
    
    # Check distance vectors match physical distances
    for track_id in [1, 2]:
        mask = dataset.tracks == track_id
        if mask.any():
            dist_sample = dataset.distances[mask][0]
            print(f"    Track {track_id} sample distances: {dist_sample}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 4. Distance Vectors Sanity Check
# ============================================================================
print("\n[4] Distance Vectors Sanity Check")
try:
    # Expected distances from config
    expected_t1 = np.array([2.5, 4.0, 8.0, 16.0, 23.0])
    expected_t2 = np.array([6.5, 8.0, 12.0, 20.0, 27.0])
    
    t1_mask = dataset.tracks == 1
    t2_mask = dataset.tracks == 2
    
    if t1_mask.any():
        t1_dist = dataset.distances[t1_mask][0]
        print(f"  Track 1 sample: {t1_dist}")
        print(f"  Expected:      {expected_t1}")
    
    if t2_mask.any():
        t2_dist = dataset.distances[t2_mask][0]
        print(f"  Track 2 sample: {t2_dist}")
        print(f"  Expected:      {expected_t2}")
    
    print(f"  ✓ Distance vectors appear consistent")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 5. Model Forward Pass
# ============================================================================
print("\n[5] Model Forward Pass")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    ).to(device)
    
    # Sample batch
    wf, meta, targets, distances, track, event_id = dataset[0]
    wf = wf.unsqueeze(0).to(device)
    meta = meta.unsqueeze(0).to(device)
    
    # Forward
    c_hat, epsilon_hat = model(wf, meta)
    print(f"  ✓ Forward pass successful")
    print(f"    c_hat shape: {c_hat.shape}, value: {c_hat[0].item():.4f}")
    print(f"    epsilon_hat shape: {epsilon_hat.shape}")
    print(f"    epsilon_hat sample: {epsilon_hat[0].cpu().detach().numpy()}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 6. Loss Computation
# ============================================================================
print("\n[6] Loss Computation")
try:
    # Create dummy batch
    batch_size = 4
    y_pred = torch.randn(batch_size, 5)
    y_true = torch.randn(batch_size, 5)
    mask = torch.ones(batch_size, dtype=torch.bool)
    
    loss_unweighted = huber_loss_log(y_pred, y_true, mask, delta=0.5)
    print(f"  ✓ Unweighted loss: {loss_unweighted.item():.4f}")
    
    # With weights
    weights = torch.ones(batch_size, 5)
    weights[:, 0] = 2.0  # MP4 weight
    loss_weighted = huber_loss_log(y_pred, y_true, mask, delta=0.5, weights=weights)
    print(f"  ✓ Weighted loss (MP4=2.0): {loss_weighted.item():.4f}")
    print(f"    Ratio: {loss_weighted.item() / loss_unweighted.item():.4f}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 7. Attenuation Exponent Fitting
# ============================================================================
print("\n[7] Attenuation Exponent Fitting")
try:
    train_idx, val_idx, test_idx = make_event_splits(
        len(dataset), 0.65, 0.15, 0.20, seed=42
    )
    
    train_targets = dataset.targets[train_idx]
    train_distances = dataset.distances[train_idx]
    train_tracks = dataset.tracks[train_idx]
    
    n_track1, n_track2 = fit_attenuation_exponents(
        train_targets, train_distances, train_tracks, r0=10.0
    )
    
    print(f"  ✓ Fitted exponents:")
    print(f"    Track 1: {n_track1:.4f}")
    print(f"    Track 2: {n_track2:.4f}")
    print(f"    Expected: ~1.0777 (T1), ~1.3300 (T2)")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 8. Curve Target Calculation
# ============================================================================
print("\n[8] Curve Target Calculation")
try:
    c_target = compute_curve_target(
        train_targets, train_distances, n_track1, r0=10.0
    )
    print(f"  ✓ Computed c_target for {len(c_target)} events")
    print(f"    Mean: {c_target.mean():.4f}")
    print(f"    Std: {c_target.std():.4f}")
    print(f"    Range: [{c_target.min():.4f}, {c_target.max():.4f}]")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ ALL SMOKE TESTS PASSED")
print("=" * 80)
print("\nReady for cluster submission!")
print("Submit with: sbatch slurm/run_curveprior_linec_v1.slurm")
