"""
Local smoke check for query-conditioned curve-prior model.

Validates:
  - Config loading (Q1-Q4)
  - Model instantiation (Q1 and Q3)
  - Forward pass shapes
  - Loss computation
  - Backward pass / gradient flow
  - CurveDataset_Query split logic and holdout exclusion
  - n-fitting (corrected method)
  - Scaler fit
  - Tiny overfit test (8 synthetic events, Q3, 50 mini-epochs)

Usage:
    python smoke_test_curvequery_linec.py
"""

from __future__ import annotations

import sys
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from src.ml.cnn.config_cnn_curvequery_linec_v1 import get_variant_config
from src.ml.cnn.cnn_curvequery_linec_utils import (
    CurvePriorCNN2D_Query, CurveDataset_Query,
    huber_loss_log, mse_scalar, compute_sample_weights,
)
from train_cnn_curvequery_linec_v1 import (
    fit_attenuation_exponents_corrected, make_event_splits, fit_meta_scaler,
)

PASS = "  ✓"
FAIL = "  ✗ FAIL"


def _make_model(cfg, device):
    return CurvePriorCNN2D_Query(
        n_metadata=4,
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


def smoke_test():
    print("=" * 80)
    print("SMOKE CHECK: Query-Conditioned Curve-Prior Model")
    print("=" * 80)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # ── Test 1: Config loading ──────────────────────────────────────────────
    print("Test 1: Configuration (Q1-Q4)")
    print("-" * 60)
    for variant in ["Q1", "Q2", "Q3", "Q4"]:
        cfg = get_variant_config(variant)
        rh  = cfg.model.enable_residual_head
        lam = cfg.model.lambda_residual
        mp4 = cfg.model.mp4_weight
        print(f"  {variant}: residual={rh}  lambda={lam}  mp4_w={mp4}")
    assert not get_variant_config("Q1").model.enable_residual_head, "Q1 should have no residual"
    assert get_variant_config("Q3").model.mp4_weight == 2.0, "Q3 should have mp4_weight=2.0"
    print(f"{PASS} All variants load with correct settings")
    print()

    # ── Test 2: Model instantiation and parameter count ─────────────────────
    print("Test 2: Model instantiation")
    print("-" * 60)
    for variant in ["Q1", "Q3"]:
        cfg   = get_variant_config(variant)
        model = _make_model(cfg, device)
        n_p   = sum(p.numel() for p in model.parameters())
        print(f"  {variant}: {n_p:,} parameters  residual_head={model.residual_head is not None}")
    print(f"{PASS} Models instantiated")
    print()

    # ── Test 3: Forward pass shapes ─────────────────────────────────────────
    print("Test 3: Forward pass (Q1 and Q3)")
    print("-" * 60)
    B = 8
    wf    = torch.randn(B, 1, 21, 7500, device=device)
    meta  = torch.randn(B, 4,  device=device)
    dist  = torch.tensor([2.5, 4.0, 8.0, 16.0, 23.0, 6.5, 12.0, 20.0],
                         dtype=torch.float32, device=device)
    track = torch.tensor([1, 1, 1, 1, 1, 2, 2, 2], dtype=torch.long, device=device)

    for variant in ["Q1", "Q3"]:
        cfg   = get_variant_config(variant)
        model = _make_model(cfg, device)
        model.eval()
        with torch.no_grad():
            c_hat, eps = model(wf, meta, dist, track)
        assert c_hat.shape == (B,), f"c_hat shape: {c_hat.shape}"
        if variant == "Q1":
            assert eps is None, "Q1 should produce eps=None"
        else:
            assert eps.shape == (B,), f"eps shape: {eps.shape}"
        print(f"  {variant}: c_hat={tuple(c_hat.shape)}  eps={'None' if eps is None else tuple(eps.shape)}")
    print(f"{PASS} Forward pass shapes correct")
    print()

    # ── Test 4: Loss computation ─────────────────────────────────────────────
    print("Test 4: Loss computation (Q3)")
    print("-" * 60)
    cfg   = get_variant_config("Q3")
    model = _make_model(cfg, device)
    n1, n2 = 1.0655, 1.3246
    r0      = 10.0

    n_vec   = torch.where(track == 1,
                          torch.tensor(n1, device=device),
                          torch.tensor(n2, device=device)).float()
    tgt_log = torch.log(torch.tensor(
        [6.0, 3.0, 1.2, 0.7, 0.4, 4.5, 2.1, 0.6],
        dtype=torch.float32, device=device))

    c_hat, eps = model(wf, meta, dist, track)
    log_r      = torch.log(dist / r0)
    y_pred     = c_hat - n_vec * log_r + (eps if eps is not None else 0)
    c_target   = (tgt_log + n_vec * log_r).detach()

    snames  = ["MP4", "MP8", "MP10", "MP1", "MP2", "MP4", "MP8", "MP10"]
    weights = compute_sample_weights(snames, cfg.model.mp4_weight, device)
    L_p     = huber_loss_log(y_pred, tgt_log, weights=weights)
    L_c     = mse_scalar(c_hat, c_target)
    L_eps   = (eps ** 2).mean() if eps is not None else torch.tensor(0.0)
    L       = L_p + 0.2 * L_c + cfg.model.lambda_residual * L_eps

    assert L.item() > 0, "Loss should be positive"
    print(f"  L_profile={L_p.item():.4f}  L_c={L_c.item():.4f}  L_eps={L_eps.item():.4f}")
    print(f"  Total loss={L.item():.4f}")
    L.backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  Gradients on {grad_count} parameter tensors")
    print(f"{PASS} Loss and backward pass OK")
    print()

    # ── Test 5: CurveDataset_Query split and holdout logic ───────────────────
    print("Test 5: CurveDataset_Query split and holdout")
    print("-" * 60)
    N = 20
    sensor_names = ["MP4", "MP8", "MP10", "MP1", "MP2"]
    wf_d  = np.zeros((N, 21, 7500), dtype=np.float32)
    mt_d  = np.zeros((N, 4),        dtype=np.float32)
    tg_d  = np.zeros((N, 5),        dtype=np.float32)
    di_d  = np.ones ((N, 5),        dtype=np.float32) * 10.0
    tk_d  = np.ones (N,             dtype=np.int64)
    ei_d  = np.array([f"evt_{i}" for i in range(N)])

    ds_all   = CurveDataset_Query(wf_d, mt_d, tg_d, di_d, tk_d, ei_d, sensor_names)
    ds_noheld = CurveDataset_Query(wf_d, mt_d, tg_d, di_d, tk_d, ei_d, sensor_names,
                                   holdout_sensor="MP4")
    ds_held   = CurveDataset_Query(wf_d, mt_d, tg_d, di_d, tk_d, ei_d, sensor_names,
                                   holdout_sensor="MP4", include_holdout_only=True)

    assert len(ds_all)    == N * 5,     f"all: {len(ds_all)} != {N*5}"
    assert len(ds_noheld) == N * 4,     f"no-held: {len(ds_noheld)} != {N*4}"
    assert len(ds_held)   == N * 1,     f"held-only: {len(ds_held)} != {N*1}"

    # Verify holdout sensor only appears in held dataset
    for _, item in [(i, ds_noheld[i]) for i in range(min(20, len(ds_noheld)))]:
        assert item["sensor_name"] != "MP4", "MP4 should not appear in no-held dataset"
    for _, item in [(i, ds_held[i]) for i in range(min(20, len(ds_held)))]:
        assert item["sensor_name"] == "MP4", "Only MP4 should appear in held-only dataset"

    # Verify item has correct keys
    item = ds_all[0]
    for key in ["waveform", "metadata", "target_log", "distance", "track",
                "event_id", "sensor_name", "sensor_idx"]:
        assert key in item, f"Missing key: {key}"
    assert item["waveform"].shape == (1, 21, 7500), f"waveform shape: {item['waveform'].shape}"

    print(f"  all-sensor:  {len(ds_all)} samples")
    print(f"  no-holdout:  {len(ds_noheld)} samples (MP4 excluded)")
    print(f"  held-only:   {len(ds_held)} samples (MP4 only)")
    print(f"{PASS} Dataset holdout logic correct")
    print()

    # ── Test 6: Corrected n-fitting ──────────────────────────────────────────
    print("Test 6: Corrected n-fitting")
    print("-" * 60)
    # Generate synthetic data where we know the true n
    np.random.seed(0)
    n_true_1, n_true_2 = 1.0777, 1.3300
    r0_test  = 10.0
    n_ev     = 200
    r_track1 = np.array([2.5, 4.0, 8.0, 16.0, 23.0])
    r_track2 = np.array([6.5, 8.0, 12.0, 20.0, 27.0])

    tracks_syn  = np.random.choice([1, 2], size=n_ev)
    c_events    = np.random.randn(n_ev) * 0.5 + 0.0
    tgt_syn     = np.zeros((n_ev, 5), dtype=np.float32)
    dist_syn    = np.zeros((n_ev, 5), dtype=np.float32)
    for i, tr in enumerate(tracks_syn):
        r_arr      = r_track1 if tr == 1 else r_track2
        n_true     = n_true_1  if tr == 1 else n_true_2
        dist_syn[i] = r_arr
        tgt_syn[i]  = (c_events[i] - n_true * np.log(r_arr / r0_test)
                       + np.random.randn(5) * 0.02).astype(np.float32)

    # Use training split only
    idx_tr = np.arange(int(n_ev * 0.65))
    n1_fit, n2_fit = fit_attenuation_exponents_corrected(
        tgt_syn[idx_tr], dist_syn[idx_tr], tracks_syn[idx_tr], r0=r0_test
    )
    assert abs(n1_fit - n_true_1) < 0.05, f"Track 1 n mismatch: {n1_fit:.4f} vs {n_true_1}"
    assert abs(n2_fit - n_true_2) < 0.05, f"Track 2 n mismatch: {n2_fit:.4f} vs {n_true_2}"

    # Holdout exclusion: n fitted without MP4
    n1_h, n2_h = fit_attenuation_exponents_corrected(
        tgt_syn[idx_tr], dist_syn[idx_tr], tracks_syn[idx_tr], r0=r0_test,
        holdout_sensor_idx=0, sensor_names=sensor_names,
    )
    print(f"  All-sensor:   n1={n1_fit:.4f}  n2={n2_fit:.4f}")
    print(f"  No-MP4:       n1={n1_h:.4f}  n2={n2_h:.4f}")
    print(f"{PASS} n-fitting recovers correct exponents")
    print()

    # ── Test 7: Scaler fit ───────────────────────────────────────────────────
    print("Test 7: Metadata scaler")
    print("-" * 60)
    meta_train = np.random.randn(100, 4).astype(np.float32)
    scaler     = fit_meta_scaler(meta_train)
    meta_scaled = scaler.transform(meta_train)
    assert abs(meta_scaled.mean()) < 0.1, "Scaled mean should be ~0"
    assert abs(meta_scaled.std() - 1.0) < 0.2, "Scaled std should be ~1"
    print(f"  Train mean post-scale: {meta_scaled.mean():.4f}")
    print(f"  Train std  post-scale: {meta_scaled.std():.4f}")
    print(f"{PASS} Scaler fits and applies correctly")
    print()

    # ── Test 8: Tiny overfit (Q3, 8 events, 50 epochs) ───────────────────────
    print("Test 8: Tiny overfit test (Q3, 8 synthetic events, 50 mini-epochs)")
    print("-" * 60)
    N_tiny = 8
    cfg_q3 = get_variant_config("Q3")
    np.random.seed(42); torch.manual_seed(42)

    wf_tiny  = np.random.randn(N_tiny, 21, 7500).astype(np.float32)
    mt_tiny  = np.random.randn(N_tiny, 4).astype(np.float32)
    tk_tiny  = np.array([1] * 4 + [2] * 4, dtype=np.int64)
    di_tiny  = np.tile(r_track1, (N_tiny, 1)).astype(np.float32)
    di_tiny[:4]  = r_track1
    di_tiny[4:]  = r_track2
    c_tiny   = np.random.randn(N_tiny) * 0.3
    tg_tiny  = np.zeros((N_tiny, 5), dtype=np.float32)
    for i in range(N_tiny):
        n_t = 1.0655 if tk_tiny[i] == 1 else 1.3246
        r_t = r_track1 if tk_tiny[i] == 1 else r_track2
        tg_tiny[i] = c_tiny[i] - n_t * np.log(r_t / r0_test)
    ei_tiny = np.array([f"e{i}" for i in range(N_tiny)])

    ds_tiny = CurveDataset_Query(wf_tiny, mt_tiny, tg_tiny, di_tiny, tk_tiny, ei_tiny,
                                 sensor_names)
    ld_tiny = DataLoader(ds_tiny, batch_size=40, shuffle=True)

    model_tiny = _make_model(cfg_q3, device)
    opt_tiny   = torch.optim.AdamW(model_tiny.parameters(), lr=3e-3)
    n1_t, n2_t = 1.0655, 1.3246
    n1_ts = torch.tensor(n1_t, dtype=torch.float32, device=device)
    n2_ts = torch.tensor(n2_t, dtype=torch.float32, device=device)

    losses = []
    model_tiny.train()
    for ep in range(50):
        ep_loss = 0.0
        for batch in ld_tiny:
            wfb  = batch["waveform"].to(device)
            mtb  = batch["metadata"].to(device)
            tgb  = batch["target_log"].to(device)
            dsb  = batch["distance"].to(device)
            tkb  = batch["track"].to(device)
            sn   = batch["sensor_name"]
            n_v  = torch.where(tkb == 1, n1_ts, n2_ts).float()
            c, e = model_tiny(wfb, mtb, dsb, tkb)
            lr_  = torch.log(dsb / r0_test)
            yp   = c - n_v * lr_ + (e if e is not None else 0)
            ct   = (tgb + n_v * lr_).detach()
            w    = compute_sample_weights(sn, cfg_q3.model.mp4_weight, device)
            loss = (huber_loss_log(yp, tgb, weights=w)
                    + 0.2 * mse_scalar(c, ct)
                    + cfg_q3.model.lambda_residual * (e ** 2).mean())
            opt_tiny.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_tiny.parameters(), 1.0)
            opt_tiny.step()
            ep_loss += loss.item()
        losses.append(ep_loss)

    first_loss = losses[0]
    final_loss = losses[-1]
    ratio      = final_loss / first_loss
    print(f"  Epoch 1  loss: {first_loss:.4f}")
    print(f"  Epoch 50 loss: {final_loss:.4f}")
    print(f"  Reduction:     {ratio:.2%}")
    assert ratio < 0.5, f"Loss did not decrease by at least 50%: ratio={ratio:.2%}"
    print(f"{PASS} Training loss decreased by {1 - ratio:.0%} over 50 epochs")
    print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 80)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 80)
    print()
    print("Ready for cluster submission:")
    print("  sbatch slurm/run_curvequery_linec_v1.slurm")


if __name__ == "__main__":
    smoke_test()



