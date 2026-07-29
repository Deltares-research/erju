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
from contextlib import nullcontext
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


def _huber(
    pred: torch.Tensor, target: torch.Tensor, delta: float = 1.5
) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=delta)


import torch.nn.functional as F


def _autocast_context(device: torch.device, enabled: bool):
    """Return a fresh autocast context for one forward pass."""
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _iter_output_tensors(model_out):
    if isinstance(model_out, (tuple, list)):
        yield from model_out
    else:
        yield model_out


def _assert_finite_tensor(name: str, value: torch.Tensor, ev_idx=None) -> None:
    if not torch.isfinite(value).all():
        bad = int((~torch.isfinite(value)).sum().item())
        ids = ev_idx.detach().cpu().tolist() if ev_idx is not None else []
        raise FloatingPointError(f"Non-finite {name}: count={bad}, event_indices={ids}")


def _assert_stable_outputs(model_out, max_abs: float, ev_idx=None) -> None:
    for k, tensor in enumerate(_iter_output_tensors(model_out)):
        _assert_finite_tensor(f"model_output[{k}]", tensor, ev_idx)
        peak = float(tensor.detach().abs().max().item())
        if peak > max_abs:
            ids = ev_idx.detach().cpu().tolist() if ev_idx is not None else []
            raise FloatingPointError(
                f"Extreme standardized model output: max_abs={peak:.3g} > {max_abs}, "
                f"event_indices={ids}"
            )


def compute_loss(
    model_out,  # single tensor or (spec, pgv) tuple
    batch_target,  # standardized target(s)
    batch_pgv_aux,  # standardized log-PGV (always available)
    cfg_loss: str,
    huber_delta: float,
    pgv_aux_weight: float,
    spectral_amp_weight: float,
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
        l_pgv = criterion(pgv_pred, batch_pgv_aux)
        total = l_spec + pgv_aux_weight * l_pgv
        return total, l_spec.item(), l_pgv.item()
    elif is_pgv:
        l = criterion(model_out, batch_target.squeeze(-1))
        return l, l.item(), 0.0
    else:
        l_spec = criterion(model_out, batch_target)
        # The average standardized residual across bands is an inexpensive proxy
        # for the event-level common spectral amplitude.  This directly penalizes
        # the range compression observed in S2/S3/S5.
        if spectral_amp_weight > 0 and model_out.ndim == 2 and model_out.shape[1] > 1:
            l_amp = criterion(model_out.mean(dim=1), batch_target.mean(dim=1))
            total = l_spec + spectral_amp_weight * l_amp
            return total, l_spec.item(), l_amp.item()
        return l_spec, l_spec.item(), 0.0


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
    spectral_amp_weight: float = 0.0,
    max_abs_output: float = 25.0,
) -> Dict[str, float]:
    """Run inference on a DataLoader and return average losses + macro-RMSE."""
    model.eval()
    total_loss_sum = 0.0
    prim_loss_sum = 0.0
    all_preds, all_targets = [], []
    n = 0

    for wf, meta, n_valid, tgt, pgv_aux, ev_idx in loader:
        wf = wf.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        n_valid = n_valid.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        pgv_aux = pgv_aux.to(device, non_blocking=True)
        _assert_finite_tensor("validation waveform", wf, ev_idx)
        _assert_finite_tensor("validation metadata", meta, ev_idx)
        _assert_finite_tensor("validation target", tgt, ev_idx)

        # Always validate in float32.  The previous bfloat16 validation produced
        # intermittent enormous spikes even when training loss remained stable.
        with _autocast_context(device, enabled=False):
            out = model(wf.float(), meta.float(), n_valid)
        _assert_stable_outputs(out, max_abs_output, ev_idx)
        loss, prim, _ = compute_loss(
            out,
            tgt.float(),
            pgv_aux.float(),
            cfg_loss,
            huber_delta,
            pgv_aux_weight,
            spectral_amp_weight,
            is_multitask,
            is_pgv,
        )
        _assert_finite_tensor("validation loss", loss, ev_idx)

        bs = wf.size(0)
        total_loss_sum += loss.item() * bs
        prim_loss_sum += prim * bs
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

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Macro-RMSE on standardized targets
    per_col_rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2, axis=0))
    macro_rmse = float(per_col_rmse.mean())

    return {
        "total_loss": total_loss_sum / max(n, 1),
        "prim_loss": prim_loss_sum / max(n, 1),
        "macro_rmse_std": macro_rmse,
    }


# ── Training loop ─────────────────────────────────────────────────────────────


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg,  # SpectralExperimentConfig
    out_dir: Path,
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
    is_pgv = (cfg.target_type == "pgv") and not is_multitask

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
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(5, tr_cfg.patience // 3),
            min_lr=1e-6,
        )
    else:
        scheduler = None

    # ── Mixed precision ───────────────────────────────────────────────────────
    use_amp = (
        tr_cfg.mixed_prec and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    # bfloat16 does not need gradient scaling.  A fresh autocast context is
    # created for every forward pass; validation is always float32.
    scaler = torch.amp.GradScaler(device=device.type, enabled=False)

    model.to(device)

    # ── Early stopping state ──────────────────────────────────────────────────
    best_val_rmse = float("inf")
    best_epoch = -1
    epochs_no_imp = 0
    best_path = out_dir / "best_model.pt"
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
        for wf, meta, n_valid, tgt, pgv_aux, ev_idx in train_loader:
            wf = wf.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)
            n_valid = n_valid.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            pgv_aux = pgv_aux.to(device, non_blocking=True)
            _assert_finite_tensor("training waveform", wf, ev_idx)
            _assert_finite_tensor("training metadata", meta, ev_idx)
            _assert_finite_tensor("training target", tgt, ev_idx)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, enabled=use_amp):
                out = model(wf, meta, n_valid)
                _assert_stable_outputs(out, tr_cfg.max_abs_standardized_output, ev_idx)
                loss, _, _ = compute_loss(
                    out,
                    tgt,
                    pgv_aux,
                    tr_cfg.loss,
                    tr_cfg.huber_delta,
                    cfg.pgv_aux_weight,
                    tr_cfg.spectral_amp_weight,
                    is_multitask,
                    is_pgv,
                )
            _assert_finite_tensor("training loss", loss, ev_idx)

            loss.backward()
            for pname, parameter in model.named_parameters():
                if (
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ):
                    ids = ev_idx.detach().cpu().tolist()
                    raise FloatingPointError(
                        f"Non-finite gradient in {pname}; event_indices={ids}"
                    )
            if tr_cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), tr_cfg.grad_clip)
            optimizer.step()

            bs = wf.size(0)
            tr_loss_sum += loss.item() * bs
            n_tr += bs

        tr_loss = tr_loss_sum / max(n_tr, 1)

        # ── Validate ─────────────────────────────────────────────────────────
        val_metrics = evaluate_loader(
            model,
            val_loader,
            device,
            tr_cfg.loss,
            tr_cfg.huber_delta,
            cfg.pgv_aux_weight,
            is_multitask,
            is_pgv,
            spectral_amp_weight=tr_cfg.spectral_amp_weight,
            max_abs_output=tr_cfg.max_abs_standardized_output,
        )
        val_rmse = val_metrics["macro_rmse_std"]

        lr_now = optimizer.param_groups[0]["lr"]
        rec = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 6),
            "val_loss": round(val_metrics["total_loss"], 6),
            "val_macro_rmse_std": round(val_rmse, 6),
            "lr": lr_now,
            "elapsed_s": round(time.time() - t_start, 1),
        }
        history.append(rec)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Ep {epoch:4d}/{tr_cfg.epochs}  "
                f"tr={tr_loss:.4f}  val={val_metrics['total_loss']:.4f}  "
                f"val_rmse_std={val_rmse:.4f}  lr={lr_now:.2e}"
            )

        # ── Scheduler step ────────────────────────────────────────────────────
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_rmse)
        elif scheduler is not None:
            scheduler.step()

        # ── Best checkpoint ──────────────────────────────────────────────────
        if val_rmse < best_val_rmse - 1e-6:
            best_val_rmse = val_rmse
            best_epoch = epoch
            epochs_no_imp = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_macro_rmse_std": val_rmse,
                },
                best_path,
            )
        else:
            epochs_no_imp += 1

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if epoch % 25 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_macro_rmse_std": val_rmse,
                },
                out_dir / f"checkpoint_ep{epoch:04d}.pt",
            )

        # ── Early stopping ────────────────────────────────────────────────────
        if epochs_no_imp >= tr_cfg.patience:
            print(
                f"  Early stop at epoch {epoch} "
                f"(best ep {best_epoch}, val_rmse_std={best_val_rmse:.4f})"
            )
            break

        # ── SIGTERM graceful exit ─────────────────────────────────────────────
        if _save_flag["kill"]:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_macro_rmse_std": val_rmse,
                },
                out_dir / f"checkpoint_sigterm_ep{epoch:04d}.pt",
            )
            print(f"  SIGTERM received — saved checkpoint at epoch {epoch}")
            break

    print(
        f"  Training done. Best epoch={best_epoch}, "
        f"best val_rmse_std={best_val_rmse:.4f}"
    )

    # Save history
    (out_dir / "train_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    return history


def load_best_checkpoint(model: nn.Module, out_dir: Path, device: torch.device) -> int:
    """Load the best_model.pt checkpoint into model. Returns best epoch."""
    ckpt_path = out_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best_model.pt in {out_dir}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return int(ckpt.get("epoch", -1))
