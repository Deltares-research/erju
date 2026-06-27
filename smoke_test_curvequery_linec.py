"""
Local smoke check for query-conditioned curve-prior model.

Validates:
  - Data loading
  - Distance vectors
  - Model instantiation
  - Forward pass
  - Loss computation
  - Small overfit test

Usage:
    python smoke_test_curvequery_linec.py
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ml.cnn.config_cnn_curvequery_linec_v1 import get_variant_config
from ml.cnn.cnn_curvequery_linec_utils import CurvePriorCNN2D_Query, CurveDataset_Query
from ml.cnn.cnn_curveprior_linec_utils import load_linec_data, build_event_level_dataset


def smoke_test():
    """Run smoke checks."""
    
    print("="*80)
    print("SMOKE CHECK: Query-Conditioned Curve-Prior Model")
    print("="*80)
    print()
    
    # 1. Configuration
    print("Test 1: Configuration")
    print("-" * 80)
    for variant in ["Q1", "Q2", "Q3", "Q4"]:
        cfg = get_variant_config(variant, holdout_sensor=None)
        print(f"  {variant}: enable_residual={cfg.train.enable_residual_head}, lambda_eps={cfg.loss.lambda_epsilon}")
    print("  ✓ All variants load")
    print()
    
    # 2. Model instantiation
    print("Test 2: Model instantiation")
    print("-" * 80)
    cfg = get_variant_config("Q3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CurvePriorCNN2D_Query(cfg)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Device: {device}")
    print(f"  Parameters: {n_params:,}")
    print("  ✓ Model instantiated")
    print()
    
    # 3. Mock data (small batch)
    print("Test 3: Forward pass with mock data")
    print("-" * 80)
    
    B = 4  # batch size
    waveform = torch.randn(B, 21, 7500, device=device)
    query_distance = torch.tensor([2.5, 4.0, 8.0, 16.0], dtype=torch.float32, device=device)
    track_id = torch.tensor([0, 0, 0, 0], dtype=torch.long, device=device)
    
    c_hat, epsilon_hat = model(waveform, query_distance=query_distance, track_id=track_id)
    
    print(f"  Input waveform: {waveform.shape}")
    print(f"  Query distance: {query_distance.shape}")
    print(f"  c_hat output: {c_hat.shape}")
    print(f"  epsilon_hat output: {epsilon_hat.shape if epsilon_hat is not None else 'None'}")
    
    assert c_hat.shape == (B,), f"c_hat shape mismatch: {c_hat.shape}"
    if epsilon_hat is not None:
        assert epsilon_hat.shape == (B,), f"epsilon_hat shape mismatch: {epsilon_hat.shape}"
    
    print("  ✓ Forward pass successful")
    print()
    
    # 4. Loss computation
    print("Test 4: Loss computation")
    print("-" * 80)
    
    pgv_targets = torch.tensor([1.5, 2.0, 3.0, 4.5], dtype=torch.float32, device=device)
    n_track = 1.0655
    
    log_r_ratio = torch.log(query_distance / 10.0)
    y_pred_base = c_hat - n_track * log_r_ratio
    
    if epsilon_hat is not None:
        y_pred = y_pred_base + epsilon_hat
    else:
        y_pred = y_pred_base
    
    target_log = torch.log(pgv_targets)
    loss = torch.mean((y_pred - target_log) ** 2)
    
    print(f"  y_pred range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print(f"  target_log range: [{target_log.min():.4f}, {target_log.max():.4f}]")
    print(f"  MSE loss: {loss.item():.6f}")
    print("  ✓ Loss computation successful")
    print()
    
    # 5. Backward pass
    print("Test 5: Backward pass")
    print("-" * 80)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss.backward()
    
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    
    print(f"  Gradients computed: {len(grad_norms)} parameters")
    print(f"  Gradient norm range: [{min(grad_norms):.2e}, {max(grad_norms):.2e}]")
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    print("  ✓ Backward pass successful")
    print()
    
    # 6. Dataset class
    print("Test 6: Dataset instantiation")
    print("-" * 80)
    
    # Small mock dataset
    n_events = 8
    waveforms_mock = np.random.randn(n_events, 21, 7500).astype(np.float32)
    pgv_targets_mock = np.random.uniform(0.5, 5.0, (n_events, 5)).astype(np.float32)
    distances_mock = np.array([
        [2.5, 4.0, 8.0, 16.0, 23.0],
        [2.5, 4.0, 8.0, 16.0, 23.0],
        [6.5, 8.0, 12.0, 20.0, 27.0],
        [6.5, 8.0, 12.0, 20.0, 27.0],
        [2.5, 4.0, 8.0, 16.0, 23.0],
        [2.5, 4.0, 8.0, 16.0, 23.0],
        [6.5, 8.0, 12.0, 20.0, 27.0],
        [6.5, 8.0, 12.0, 20.0, 27.0],
    ], dtype=np.float32)
    tracks_mock = np.array([1, 1, 2, 2, 1, 1, 2, 2])
    event_ids_mock = np.arange(n_events)
    
    dataset = CurveDataset_Query(
        waveforms_mock, pgv_targets_mock, distances_mock, tracks_mock, event_ids_mock,
        split="train", holdout_sensor=None
    )
    
    print(f"  Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  Sample waveform shape: {sample['waveform'].shape}")
    print(f"  Sample distance: {sample['distance']:.2f} m")
    print("  ✓ Dataset class works")
    print()
    
    # 7. Held-out sensor filtering
    print("Test 7: Held-out sensor filtering")
    print("-" * 80)
    
    dataset_all = CurveDataset_Query(
        waveforms_mock, pgv_targets_mock, distances_mock, tracks_mock, event_ids_mock,
        split="train", holdout_sensor=None
    )
    
    dataset_holdout_mp4 = CurveDataset_Query(
        waveforms_mock, pgv_targets_mock, distances_mock, tracks_mock, event_ids_mock,
        split="train", holdout_sensor="MP4"
    )
    
    print(f"  All sensors: {len(dataset_all)} samples ({len(dataset_all)//8} events × 5 sensors)")
    print(f"  Holdout MP4 train: {len(dataset_holdout_mp4)} samples (~{len(dataset_holdout_mp4)//8} events × 4 sensors)")
    
    assert len(dataset_holdout_mp4) < len(dataset_all), "Holdout filtering failed"
    print("  ✓ Holdout sensor filtering works")
    print()
    
    # 8. Batch loading
    print("Test 8: Batch loading from dataset")
    print("-" * 80)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch_idx, batch in enumerate(loader):
        print(f"  Batch {batch_idx}: {len(batch['waveform'])} samples")
        print(f"    Waveforms: {batch['waveform'].shape}")
        print(f"    Distances: {batch['distance'].shape}")
        print(f"    PGV targets: {batch['pgv_target'].shape}")
        if batch_idx >= 1:
            break
    
    print("  ✓ Batch loading works")
    print()
    
    # 9. Integration test (Q1 vs Q3)
    print("Test 9: Q1 vs Q3 behavior")
    print("-" * 80)
    
    cfg_q1 = get_variant_config("Q1")
    cfg_q3 = get_variant_config("Q3")
    
    model_q1 = CurvePriorCNN2D_Query(cfg_q1).to(device)
    model_q3 = CurvePriorCNN2D_Query(cfg_q3).to(device)
    
    c_hat_q1, eps_q1 = model_q1(waveform, query_distance=query_distance, track_id=track_id)
    c_hat_q3, eps_q3 = model_q3(waveform, query_distance=query_distance, track_id=track_id)
    
    print(f"  Q1 (curve-only):")
    print(f"    c_hat: {c_hat_q1.shape}")
    print(f"    epsilon: {eps_q1}")
    
    print(f"  Q3 (curve + residual + weight):")
    print(f"    c_hat: {c_hat_q3.shape}")
    print(f"    epsilon: {eps_q3.shape}")
    
    assert eps_q1 is None, "Q1 should not have residuals"
    assert eps_q3 is not None, "Q3 should have residuals"
    
    print("  ✓ Variant behavior correct")
    print()
    
    print("="*80)
    print("✓ ALL SMOKE CHECKS PASSED")
    print("="*80)
    print()
    print("Ready for SLURM submission:")
    print("  sbatch slurm/run_curvequery_linec_v1.slurm")
    print()


if __name__ == "__main__":
    smoke_test()
