"""engine.py
============
Training loop, early stopping, checkpointing and history for M0/M1/M2.

Clean rerun (2026-08-03): FP32 only -- no autocast/bfloat16 anywhere. Every
batch is checked for finiteness at each stage (inputs, predictions, loss
components, gradients); validation macro RMSE is checked too. The instant a
non-finite value appears, a diagnostic artifact is written and the run
aborts (see stability.py) -- batches are never skipped silently.

Fixed hyperparameters (documented, not swept):

    optimizer      AdamW(lr=1e-4, weight_decay=1e-4)
    scheduler      ReduceLROnPlateau(factor=0.5, patience=10) on val macro RMSE (dB)
    batch_size     32
    epochs         250 (full) / --epochs override for smoke
    patience       40 (full, early stopping) / --patience override for smoke
    grad_clip      1.0 (max grad norm)
    precision      float32 only
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.ml.linec_multisensor.losses import composite_loss
from src.ml.linec_multisensor.stability import save_diagnostic_and_abort

TRAIN_HP = dict(lr=1e-4, weight_decay=1e-4, batch_size=32, epochs=250, patience=40, grad_clip=1.0)


def _check_batch_finite(stage: str, epoch: int, batch_idx: int, out_dir: Path, **tensors) -> None:
    for name, t in tensors.items():
        if isinstance(t, torch.Tensor) and not bool(torch.isfinite(t).all()):
            save_diagnostic_and_abort(out_dir, stage, epoch, batch_idx, tensors,
                                       message=f"non-finite values in '{name}'")


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
              out_dir: Path, epoch: int) -> float:
    model.eval()
    se, n = 0.0, 0
    for batch_idx, (wf, meta, nv, tgt, r, track, _) in enumerate(loader):
        wf, meta, nv, tgt, r, track = (t.to(device) for t in (wf, meta, nv, tgt, r, track))
        _check_batch_finite("val_input", epoch, batch_idx, out_dir, wf=wf, meta=meta, tgt=tgt, r=r)
        pred = model(wf, meta, nv, r, track)
        _check_batch_finite("val_pred", epoch, batch_idx, out_dir, pred=pred)
        se += float(((pred - tgt) ** 2).sum().item())
        n += tgt.numel()
    val_rmse = float(np.sqrt(se / max(n, 1)))
    if not np.isfinite(val_rmse):
        save_diagnostic_and_abort(out_dir, "val_rmse", epoch, -1, {},
                                   message=f"val_macro_rmse_db is non-finite ({val_rmse})")
    return val_rmse


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 device: torch.device, out_dir: Path, target_mean: np.ndarray, target_std: np.ndarray,
                 epochs: int = None, patience: int = None) -> Tuple[List[Dict], int]:
    hp = TRAIN_HP
    epochs = epochs if epochs is not None else hp["epochs"]
    patience = patience if patience is not None else hp["patience"]
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    model.to(device)
    t_mean = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    t_std = torch.as_tensor(target_std, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    best_val = float("inf")
    best_epoch = -1
    epochs_since_best = 0
    history: List[Dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum, n_batches = 0.0, 0
        comp_sums: Dict[str, float] = {}
        grad_norm_sum, grad_norm_max = 0.0, 0.0

        for batch_idx, (wf, meta, nv, tgt, r, track, _) in enumerate(train_loader):
            wf, meta, nv, tgt, r, track = (t.to(device) for t in (wf, meta, nv, tgt, r, track))
            _check_batch_finite("train_input", epoch, batch_idx, out_dir, wf=wf, meta=meta, tgt=tgt, r=r)

            opt.zero_grad(set_to_none=True)
            pred = model(wf, meta, nv, r, track)
            _check_batch_finite("train_pred", epoch, batch_idx, out_dir, pred=pred)

            loss, comps = composite_loss(pred, tgt, t_mean, t_std)
            if not bool(torch.isfinite(loss)):
                save_diagnostic_and_abort(out_dir, "train_loss", epoch, batch_idx,
                                           {"pred": pred, "tgt": tgt}, message=f"loss is non-finite: {comps}")
            for k, v in comps.items():
                if not np.isfinite(v):
                    save_diagnostic_and_abort(out_dir, "train_loss_component", epoch, batch_idx,
                                               {"pred": pred, "tgt": tgt},
                                               message=f"loss component '{k}'={v} is non-finite")

            loss.backward()

            grad_norm_sq = 0.0
            for pname, p in model.named_parameters():
                if p.grad is not None:
                    if not bool(torch.isfinite(p.grad).all()):
                        save_diagnostic_and_abort(
                            out_dir, "train_grad", epoch, batch_idx,
                            {"pred": pred, "tgt": tgt, "r": r, "meta": meta, f"grad[{pname}]": p.grad},
                            message=f"non-finite gradient detected in parameter '{pname}'")
                    grad_norm_sq += float(p.grad.detach().float().pow(2).sum())
            if not np.isfinite(grad_norm_sq):
                save_diagnostic_and_abort(out_dir, "train_grad_norm", epoch, batch_idx, {},
                                           message=f"gradient norm is non-finite ({grad_norm_sq})")
            grad_norm = grad_norm_sq ** 0.5
            grad_norm_sum += grad_norm
            grad_norm_max = max(grad_norm_max, grad_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            opt.step()

            train_loss_sum += comps["total"]
            n_batches += 1
            for k, v in comps.items():
                comp_sums[k] = comp_sums.get(k, 0.0) + v

        train_loss = train_loss_sum / max(n_batches, 1)
        val_rmse_db = _evaluate(model, val_loader, device, out_dir, epoch)
        sched.step(val_rmse_db)
        elapsed = time.time() - t0

        row = {"epoch": epoch, "train_loss": train_loss, "val_macro_rmse_db": val_rmse_db,
               "lr": opt.param_groups[0]["lr"], "elapsed_s": round(elapsed, 2),
               "grad_norm_mean": grad_norm_sum / max(n_batches, 1), "grad_norm_max": grad_norm_max}
        row.update({f"train_{k}": v / max(n_batches, 1) for k, v in comp_sums.items()})
        history.append(row)
        print(f"  epoch {epoch:4d}  train_loss={train_loss:.4f}  val_macro_rmse={val_rmse_db:.4f} dB  "
              f"lr={row['lr']:.2e}  grad_norm(mean/max)={row['grad_norm_mean']:.3f}/{row['grad_norm_max']:.3f}  "
              f"({elapsed:.1f}s)")

        improved = val_rmse_db < best_val - 1e-4
        if improved:
            best_val = val_rmse_db
            best_epoch = epoch
            epochs_since_best = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_macro_rmse_db": val_rmse_db}, out_dir / "best_model.pt")
        else:
            epochs_since_best += 1
        if epochs_since_best >= patience:
            print(f"  early stopping at epoch {epoch} (best epoch {best_epoch}, val={best_val:.4f} dB)")
            break

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))

    if best_epoch == -1:
        (out_dir / "RUN_INVALID").write_text(
            "Run invalid: training loop completed without ever recording a finite "
            "validation checkpoint (best_epoch=-1).\n"
        )
    return history, best_epoch


def load_best_checkpoint(model: nn.Module, out_dir: Path, device: torch.device) -> int:
    ckpt_path = out_dir / "best_model.pt"
    if not ckpt_path.exists():
        (out_dir / "RUN_INVALID").write_text(
            f"Run invalid: no checkpoint found at {ckpt_path} -- training never produced a "
            f"finite validation result.\n"
        )
        raise RuntimeError(f"No valid checkpoint for this run: {ckpt_path} does not exist. Run marked INVALID.")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return int(ckpt["epoch"])
