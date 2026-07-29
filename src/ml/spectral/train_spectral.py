"""train_spectral.py
===================
Training loop, checkpointing, and early stopping for the MP8 spectral models.

Supports
--------
* Mixed precision (bfloat16) via torch.amp
* CosineAnnealingLR or ReduceLROnPlateau
* Gradient clipping
* Best-model checkpointing (by validation macro-RMSE on standardized targets)
* Periodic checkpoints every 25 epochs
* Graceful SIGTERM handling (saves last checkpoint before exit)
* Multi-task loss: MSE_spectral + λ * MSE_pgv
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ── Loss helpers ──────────────────────────────────────────────────────────────

def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def _huber(pred: torch.Tensor, target: torch.Tensor,
           delta: float = 1.5) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=delta)


import torch.nn.functional as F


def compute_loss(
    model_out,            # single tensor or (spec, pgv) tuple
    batch_target,         # standardized target(s)
    batch_pgv_aux,        # standardized log-PGV (always available)
    cfg_loss: str,
    huber_delta: float,
    pgv_aux_weight: float,
    is_multitask: bool,
    is_pgv: bool,
) -> Tuple[torch.Tensor, float, float]:
    """Compute primary loss (and optionally auxiliary PGV loss).

    Returns (total_loss, primary_loss_val, aux_loss_val).
    """
    criterion = _huber if cfg_loss == "huber" else _mse

    if is_multitask:
        spec_pred, pgv_pred = model_out
        l_spec = criterion(spec_pred, batch_target)
        l_pgv  = criterion(pgv_pred, batch_pgv_aux)
        total  = l_spec + pgv_aux_weight * l_pgv
        return total, l_spec.item(), l_pgv.item()
    elif is_pgv:
        l = criterion(model_out, batch_target.squeeze(-1))
        return l, l.item(), 0.0
    else:
        l = criterion(model_out, batch_target)
        return l, l.item(), 0.0


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg_loss: str,
    huber_delta: float,
    pgv_aux_weight: float,
    is_multitask: bool,
    is_pgv: bool,
    amp_ctx,
) -> Dict[str, float]:
    """Run inference on a DataLoader and return average losses + macro-RMSE."""
    model.eval()
    total_loss_sum = 0.0
    prim_loss_sum  = 0.0
    all_preds, all_targets = [], []
    n = 0

    for wf, meta, n_valid, tgt, pgv_aux, _ in loader:
        wf       = wf.to(device)
        meta     = meta.to(device)
        n_valid  = n_valid.to(device)
        tgt      = tgt.to(device)
        pgv_aux  = pgv_aux.to(device)

        with amp_ctx:
            out = model(wf, meta, n_valid)
        loss, prim, _ = compute_loss(
            out, tgt, pgv_aux, cfg_loss, huber_delta, pgv_aux_weight,
            is_multitask, is_pgv)

        bs = wf.size(0)
        total_loss_sum += loss.item() * bs
        prim_loss_sum  += prim * bs
        n += bs

        # Collect predictions for macro-RMSE
        if is_multitask:
            p = out[0]
        elif is_pgv:
            p = out.unsqueeze(-1)
        else:
            p = out
        all_preds.append(p.float().cpu().numpy())
        all_targets.append(tgt.float().cpu().numpy())

    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Macro-RMSE on standardized targets
    per_col_rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2, axis=0))
    macro_rmse   = float(per_col_rmse.mean())

    return {
        "total_loss":  total_loss_sum / max(n, 1),
        "prim_loss":   prim_loss_sum  / max(n, 1),
        "macro_rmse_std": macro_rmse,
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train_model(
    model:         nn.Module,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    device:        torch.device,
    cfg,                          # SpectralExperimentConfig
    out_dir:       Path,
) -> List[Dict[str, float]]:
    """Train model with early stopping; return history list.

    Saves
    -----
    best_model.pt          best validation macro-RMSE (standardized)
    checkpoint_ep{N}.pt    periodic checkpoint every 25 epochs
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_cfg = cfg.train
    is_multitask = cfg.use_pgv_aux
    is_pgv       = (cfg.target_type == "pgv") and not is_multitask

    # ── Optimizer + scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tr_cfg.lr, weight_decay=tr_cfg.weight_decay
    )
    if tr_cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tr_cfg.epochs, eta_min=tr_cfg.lr * 0.01
        )
    elif tr_cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=max(5, tr_cfg.patience // 3), min_lr=1e-6
        )
    else:
        scheduler = None

    # ── Mixed precision ───────────────────────────────────────────────────────
    use_amp = (tr_cfg.mixed_prec and device.type == "cuda"
               and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    amp_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                  enabled=use_amp)
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp and
                                   amp_dtype == torch.float16)

    model.to(device)

    # ── Early stopping state ──────────────────────────────────────────────────
    best_val_rmse  = float("inf")
    best_epoch     = -1
    epochs_no_imp  = 0
    best_path      = out_dir / "best_model.pt"
    history: List[Dict] = []

    # ── SIGTERM handler ───────────────────────────────────────────────────────
    _save_flag = {"kill": False}
    def _handle_sigterm(sig, frame):
        _save_flag["kill"] = True
    signal.signal(signal.SIGTERM, _handle_sigterm)

    t_start = time.time()

    for epoch in range(1, tr_cfg.epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        tr_loss_sum, n_tr = 0.0, 0
        for wf, meta, n_valid, tgt, pgv_aux, _ in train_loader:
            wf      = wf.to(device)
            meta    = meta.to(device)
            n_valid = n_valid.to(device)
            tgt     = tgt.to(device)
            pgv_aux = pgv_aux.to(device)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                out = model(wf, meta, n_valid)
                loss, _, _ = compute_loss(
                    out, tgt, pgv_aux, tr_cfg.loss, tr_cfg.huber_delta,
                    cfg.pgv_aux_weight, is_multitask, is_pgv)

            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                if tr_cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), tr_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if tr_cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), tr_cfg.grad_clip)
                optimizer.step()

            bs = wf.size(0)
            tr_loss_sum += loss.item() * bs
            n_tr += bs

        tr_loss = tr_loss_sum / max(n_tr, 1)

        # ── Validate ─────────────────────────────────────────────────────────
        val_metrics = evaluate_loader(
            model, val_loader, device,
            tr_cfg.loss, tr_cfg.huber_delta, cfg.pgv_aux_weight,
            is_multitask, is_pgv, amp_ctx,
        )
        val_rmse = val_metrics["macro_rmse_std"]

        lr_now = optimizer.param_groups[0]["lr"]
        rec = {
            "epoch":      epoch,
            "train_loss": round(tr_loss, 6),
            "val_loss":   round(val_metrics["total_loss"], 6),
            "val_macro_rmse_std": round(val_rmse, 6),
            "lr":         lr_now,
            "elapsed_s":  round(time.time() - t_start, 1),
        }
        history.append(rec)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:4d}/{tr_cfg.epochs}  "
                  f"tr={tr_loss:.4f}  val={val_metrics['total_loss']:.4f}  "
                  f"val_rmse_std={val_rmse:.4f}  lr={lr_now:.2e}")

        # ── Scheduler step ────────────────────────────────────────────────────
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_rmse)
        elif scheduler is not None:
            scheduler.step()

        # ── Best checkpoint ──────────────────────────────────────────────────
        if val_rmse < best_val_rmse - 1e-6:
            best_val_rmse = val_rmse
            best_epoch    = epoch
            epochs_no_imp = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_macro_rmse_std": val_rmse,
            }, best_path)
        else:
            epochs_no_imp += 1

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if epoch % 25 == 0:
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_macro_rmse_std": val_rmse,
            }, out_dir / f"checkpoint_ep{epoch:04d}.pt")

        # ── Early stopping ────────────────────────────────────────────────────
        if epochs_no_imp >= tr_cfg.patience:
            print(f"  Early stop at epoch {epoch} "
                  f"(best ep {best_epoch}, val_rmse_std={best_val_rmse:.4f})")
            break

        # ── SIGTERM graceful exit ─────────────────────────────────────────────
        if _save_flag["kill"]:
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_macro_rmse_std": val_rmse,
            }, out_dir / f"checkpoint_sigterm_ep{epoch:04d}.pt")
            print(f"  SIGTERM received — saved checkpoint at epoch {epoch}")
            break

    print(f"  Training done. Best epoch={best_epoch}, "
          f"best val_rmse_std={best_val_rmse:.4f}")

    # Save history
    (out_dir / "train_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    return history


def load_best_checkpoint(model: nn.Module, out_dir: Path,
                          device: torch.device) -> int:
    """Load the best_model.pt checkpoint into model. Returns best epoch."""
    ckpt_path = out_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best_model.pt in {out_dir}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return int(ckpt.get("epoch", -1))
