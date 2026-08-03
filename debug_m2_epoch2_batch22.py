#!/usr/bin/env python3
"""debug_m2_epoch2_batch22.py
==============================
Isolated, read-only diagnostic for the M2 non-finite-gradient failure that
reproduces deterministically at epoch=2 batch_idx=22 in
`trunk.encoder.encoder.0.conv1.weight` (seed=42, full data, batch_size=32).

This script does NOT modify src/ml/linec_multisensor/{models,losses,engine}.py
or src/ml/spectral/models_spectral.py. It replicates the exact training
trajectory (same seed -> same weight init -> same DataLoader shuffle order)
up to epoch 2 batch 22, then instruments that single batch:

  1. Saves model + optimizer state immediately before the forward pass.
  2. Recomputes the M2 forward pass manually (calling the model's own
     submodules, not duplicating their internals) so intermediate tensors
     (raw head output, t_hat, shape_logits, s_hat, c_hat, pred, and the
     ResNet2DEncoder's masked-statistics-pool mean/var/std) can be retained.
  3. Runs backward separately, from a freshly recomputed forward each time,
     for: spectral huber, total-level huber, ccc, shape huber, and pred.sum().
  4. Wraps each attempt in torch.autograd.set_detect_anomaly(True) to get
     the exact backward op + traceback that first produces a NaN/Inf.
  5. Never calls opt.step() or advances past this batch.

Usage (mirrors run_linec_multisensor_v1.py's M2 full-data config exactly):
    python debug_m2_epoch2_batch22.py --seed 42 --output-root <dir>
"""

from __future__ import annotations

import argparse
import json
import socket
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.ml.linec_multisensor.data import build_datasets, make_loader
from src.ml.linec_multisensor.models import build_model
from src.ml.linec_multisensor.losses import (
    composite_loss, total_level_db, _per_sensor_total_db, _ccc_loss,
    SPECTRAL_HUBER_DELTA, TOTAL_LEVEL_HUBER_DELTA, SHAPE_HUBER_DELTA,
)
from src.ml.linec_multisensor.engine import TRAIN_HP, _evaluate
from src.ml.spectral.models_spectral import _infer_raw_valid_mask, _resize_time_mask

TARGET_EPOCH = 2       # 1-indexed, matches engine.py's `for epoch in range(1, epochs+1)`
TARGET_BATCH_IDX = 22  # 0-indexed, matches engine.py's `enumerate(train_loader)`


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _stats(t: torch.Tensor) -> dict:
    tf = t.detach().float()
    finite = torch.isfinite(tf)
    return {
        "shape": list(t.shape), "n_nan": int(torch.isnan(tf).sum()), "n_inf": int(torch.isinf(tf).sum()),
        "min_finite": float(tf[finite].min()) if bool(finite.any()) else None,
        "max_finite": float(tf[finite].max()) if bool(finite.any()) else None,
    }


def _retain(t: torch.Tensor, interm: dict, name: str) -> None:
    """retain_grad() only if the tensor is actually part of the graph;
    records it either way so _stats() can still report its forward value."""
    if t.requires_grad:
        t.retain_grad()
    interm[name] = t


def manual_forward(model, wf, meta, nv, r, track):
    """Re-derive M2Model.forward() by calling the model's own submodules
    (trunk, head, n_o0 buffer) so every intermediate can retain_grad().
    Numerically identical to M2Model.forward -- no logic is duplicated for
    the encoder/head themselves, only the top-level assembly is unrolled.
    """
    interm = {}

    # -- shared trunk, unrolled to expose the ResNet2DEncoder's internal pooling --
    encoder = model.trunk.encoder  # ResNet2DEncoder
    raw_mask = _infer_raw_valid_mask(wf, nv, encoder.padding_zero_tol)
    z_conv = encoder.encoder(wf)  # (B, C, H, T) pre-pool conv stack output
    _retain(z_conv, interm, "z_conv")

    # MaskedStatisticsPool internals, replicated read-only for instrumentation
    x = z_conv if z_conv.dim() == 4 else z_conv.unsqueeze(2)
    B, C, H, T = x.shape
    time_mask = _resize_time_mask(raw_mask, T)
    mask = time_mask[:, None, None, :].to(dtype=x.dtype)
    count = (time_mask.sum(dim=1).to(dtype=x.dtype) * H).clamp_min(1.0)[:, None]
    summed = (x * mask).sum(dim=(2, 3))
    mean = summed / count
    second = (x.square() * mask).sum(dim=(2, 3)) / count
    _retain(mean, interm, "pool_mean"); _retain(second, interm, "pool_second")

    var = (second - mean.square()).clamp_min(0.0)
    _retain(var, interm, "pool_var")
    std = var.sqrt()
    _retain(std, interm, "pool_std")

    neg_inf = torch.finfo(x.dtype).min
    maxv = x.masked_fill(~time_mask[:, None, None, :], neg_inf).amax(dim=(2, 3))
    maxv = torch.where(torch.isfinite(maxv), maxv, torch.zeros_like(maxv))
    enc = torch.cat([mean, std, maxv], dim=1)
    _retain(enc, interm, "encoder_out")

    meta_emb = model.trunk.meta_net(meta)
    amp = model.trunk.raw_amp(wf, raw_mask).to(dtype=enc.dtype)
    _retain(amp, interm, "raw_amp_out")
    trunk_out = torch.cat([enc, meta_emb, amp], dim=1)

    # -- M2 head + physics decode, unrolled from M2Model.forward --
    out = model.head(trunk_out).float()
    _retain(out, interm, "head_out")
    t_hat = out[:, 0]; _retain(t_hat, interm, "t_hat")
    shape_logits = out[:, 1:]; _retain(shape_logits, interm, "shape_logits")

    import math
    ln10_10 = math.log(10.0) / 10.0
    log_probs = F.log_softmax(shape_logits * ln10_10, dim=1)
    s_hat = log_probs / ln10_10
    _retain(s_hat, interm, "s_hat")
    c_hat = t_hat.unsqueeze(1) + s_hat
    _retain(c_hat, interm, "c_hat")

    n_idx = (track - 1).clamp(0, 1)
    n_sel = model.n_o0[n_idx]
    corr = -20.0 * n_sel.unsqueeze(1) * torch.log10(r.unsqueeze(-1) / model.r0)
    pred = c_hat.unsqueeze(1) + corr
    _retain(pred, interm, "pred")

    return pred, interm


def run_backward_case(model, batch, stats, out_dir: Path, case_name: str, loss_fn) -> dict:
    """Fresh forward + backward for one loss branch. Returns diagnostic dict.
    Never steps the optimizer; never mutates model state permanently beyond
    the gradients this backward call itself populates (caller must zero_grad
    before/after as needed).
    """
    wf, meta, nv, tgt, r, track = batch
    model.zero_grad(set_to_none=True)
    print(f"     [DIAG] pre-forward torch.is_grad_enabled()={torch.is_grad_enabled()}")

    anomaly_error = None
    interm = {}
    try:
        with torch.autograd.set_detect_anomaly(True):
            pred, interm = manual_forward(model, wf, meta, nv, r, track)
            loss = loss_fn(pred, tgt)
            loss.backward()
    except RuntimeError as e:
        anomaly_error = str(e)

    conv1 = model.trunk.encoder.encoder[0].conv1
    conv1_grad = conv1.weight.grad
    result = {
        "case": case_name,
        "anomaly_traceback": anomaly_error,
        "loss_value": float(loss.detach().item()) if anomaly_error is None else None,
        "conv1_weight_grad": _stats(conv1_grad) if conv1_grad is not None else None,
        "intermediates": {k: _stats(v) for k, v in interm.items()},
        "intermediate_grads": {
            k: _stats(v.grad) for k, v in interm.items() if v.grad is not None
        },
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("[FATAL] this debug script requires CUDA, no CPU fallback.")
    device = torch.device("cuda")

    root = args.output_root or (Path(r"P:\11210978-erju-ai") if os.name == "nt" else Path("/p/11210978-erju-ai")) \
        / "holten_models" / "outputs" / "linec_multisensor_v1" / "M2_debug"
    root.mkdir(parents=True, exist_ok=True)

    print(f"hostname={socket.gethostname()}  SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', 'N/A')}  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}  GPU={torch.cuda.get_device_name(0)}")

    # -- exact same call sequence as run_linec_multisensor_v1.py::main() up to train_model() --
    set_seed(args.seed)
    train_ds, val_ds, test_ds, stats, events_df = build_datasets(smoke_n=None)
    batch_size = 32
    train_loader = make_loader(train_ds, batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_ds, batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_model("M2", stats.n_meta, n_o0=stats.n_o0).to(device)
    print(f"Events: train={len(train_ds)} val={len(val_ds)}  n_meta={stats.n_meta}")

    t_mean = torch.as_tensor(stats.target_mean, dtype=torch.float32, device=device)
    t_std = torch.as_tensor(stats.target_std, dtype=torch.float32, device=device)
    hp = TRAIN_HP
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    def train_step(batch):
        wf, meta, nv, tgt, r, track = (t.to(device) for t in batch[:6])
        opt.zero_grad(set_to_none=True)
        pred = model(wf, meta, nv, r, track)
        loss, comps = composite_loss(pred, tgt, t_mean, t_std)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
        opt.step()
        return comps

    # -- epoch 1: full, identical to engine.train_model's loop --
    print("Replaying epoch 1 (full) ...")
    model.train()
    for batch_idx, (wf, meta, nv, tgt, r, track, _) in enumerate(train_loader):
        train_step((wf, meta, nv, tgt, r, track))
    val_rmse = _evaluate(model, val_loader, device, root, epoch=1)
    sched.step(val_rmse)
    print(f"  epoch 1 done, val_macro_rmse={val_rmse:.4f} dB")

    # -- epoch 2: batches 0..21 normally, then instrument batch 22 --
    print("Replaying epoch 2 batches 0..21 ...")
    model.train()
    target_batch = None
    for batch_idx, (wf, meta, nv, tgt, r, track, _) in enumerate(train_loader):
        if batch_idx < TARGET_BATCH_IDX:
            train_step((wf, meta, nv, tgt, r, track))
            continue
        if batch_idx == TARGET_BATCH_IDX:
            target_batch = (wf.to(device), meta.to(device), nv.to(device),
                             tgt.to(device), r.to(device), track.to(device))
            break

    if target_batch is None:
        raise SystemExit(f"[FATAL] train_loader had fewer than {TARGET_BATCH_IDX + 1} batches this epoch.")

    print(f"Reached epoch={TARGET_EPOCH} batch_idx={TARGET_BATCH_IDX}. Instrumenting (no optimizer step will follow).")
    conv1_w = model.trunk.encoder.encoder[0].conv1.weight
    print(f"[DIAG] torch.is_grad_enabled()={torch.is_grad_enabled()}  model.training={model.training}  "
          f"conv1.weight.requires_grad={conv1_w.requires_grad}")
    wf, meta, nv, tgt, r, track = target_batch

    # 1. Save model + optimizer state immediately before the forward pass.
    torch.save(model.state_dict(), root / "pre_batch22_model_state.pt")
    torch.save(opt.state_dict(), root / "pre_batch22_optimizer_state.pt")
    torch.save({"wf": wf.cpu(), "meta": meta.cpu(), "nv": nv.cpu(), "tgt": tgt.cpu(),
                "r": r.cpu(), "track": track.cpu()}, root / "batch22_inputs.pt")

    cases = {
        "spectral_huber": lambda pred, tgt: F.huber_loss(
            (pred - t_mean) / t_std, (tgt - t_mean) / t_std, delta=SPECTRAL_HUBER_DELTA),
        "total_level_huber": lambda pred, tgt: F.huber_loss(
            total_level_db(pred), total_level_db(tgt), delta=TOTAL_LEVEL_HUBER_DELTA),
        "ccc": lambda pred, tgt: _ccc_loss(total_level_db(pred), total_level_db(tgt)),
        "shape_huber": lambda pred, tgt: F.huber_loss(
            pred - _per_sensor_total_db(pred).unsqueeze(-1),
            tgt - _per_sensor_total_db(tgt).unsqueeze(-1), delta=SHAPE_HUBER_DELTA),
        "pred_sum": lambda pred, tgt: pred.sum(),
    }

    results = {}
    for name, fn in cases.items():
        print(f"  -- case: {name} --")
        res = run_backward_case(model, (wf, meta, nv, tgt, r, track), stats, root, name, fn)
        results[name] = res
        conv1_grad = res["conv1_weight_grad"]
        n_nan = conv1_grad["n_nan"] if conv1_grad else None
        print(f"     conv1.weight.grad n_nan={n_nan}  anomaly={'YES: ' + res['anomaly_traceback'][:200] if res['anomaly_traceback'] else 'no'}")

    model.zero_grad(set_to_none=True)  # leave no stray grads behind; NO opt.step() is ever called

    (root / "debug_report.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nAll diagnostics saved to {root}")
    print("No optimizer step was taken after batch 22; run stops here as instructed.")


if __name__ == "__main__":
    main()
