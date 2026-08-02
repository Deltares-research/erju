"""engine.py
============
Training loop, early stopping, checkpointing and history for M0/M1/M2.
Fixed hyperparameters (documented, not swept):

    optimizer      AdamW(lr=1e-4, weight_decay=1e-4)
    scheduler      ReduceLROnPlateau(factor=0.5, patience=10) on val macro RMSE (dB)
    batch_size     32
    epochs         250 (full) / --epochs override for smoke
    patience       40 (full, early stopping) / --patience override for smoke
    grad_clip      1.0 (max grad norm)
    mixed_prec     bf16 autocast on CUDA only (identical for all 3 models)
"""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.ml.linec_multisensor.losses import composite_loss

TRAIN_HP = dict(lr=1e-4, weight_decay=1e-4, batch_size=32, epochs=250, patience=40, grad_clip=1.0)


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def _macro_rmse_db(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    se, n = 0.0, 0
    for wf, meta, nv, tgt, r, track, _ in loader:
        wf, meta, nv, tgt, r, track = (t.to(device) for t in (wf, meta, nv, tgt, r, track))
        pred = model(wf, meta, nv, r, track)
        se += float(((pred - tgt) ** 2).sum().item())
        n += tgt.numel()
    return float(np.sqrt(se / max(n, 1)))


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 device: torch.device, out_dir: Path, target_mean: np.ndarray, target_std: np.ndarray,
                 epochs: int = None, patience: int = None, mixed_prec: bool = True) -> List[Dict]:
    hp = TRAIN_HP
    epochs = epochs if epochs is not None else hp["epochs"]
    patience = patience if patience is not None else hp["patience"]

    model.to(device)
    t_mean = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    t_std = torch.as_tensor(target_std, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    best_val = float("inf")
    best_epoch = -1
    epochs_since_best = 0
    history: List[Dict] = []
    ckpt_dir = out_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum, n_batches = 0.0, 0
        comp_sums = {}
        for wf, meta, nv, tgt, r, track, _ in train_loader:
            wf, meta, nv, tgt, r, track = (t.to(device) for t in (wf, meta, nv, tgt, r, track))
            opt.zero_grad(set_to_none=True)
            with _autocast(device, mixed_prec):
                pred = model(wf, meta, nv, r, track)
                loss, comps = composite_loss(pred.float(), tgt.float(), t_mean, t_std)
            loss.backward()
            if hp["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            opt.step()
            train_loss_sum += comps["total"]
            n_batches += 1
            for k, v in comps.items():
                comp_sums[k] = comp_sums.get(k, 0.0) + v

        train_loss = train_loss_sum / max(n_batches, 1)
        val_rmse_db = _macro_rmse_db(model, val_loader, device)
        sched.step(val_rmse_db)
        elapsed = time.time() - t0

        row = {"epoch": epoch, "train_loss": train_loss, "val_macro_rmse_db": val_rmse_db,
               "lr": opt.param_groups[0]["lr"], "elapsed_s": round(elapsed, 2)}
        row.update({f"train_{k}": v / max(n_batches, 1) for k, v in comp_sums.items()})
        history.append(row)
        print(f"  epoch {epoch:4d}  train_loss={train_loss:.4f}  val_macro_rmse={val_rmse_db:.4f} dB  "
              f"lr={row['lr']:.2e}  ({elapsed:.1f}s)")

        improved = val_rmse_db < best_val - 1e-4
        if improved:
            best_val = val_rmse_db
            best_epoch = epoch
            epochs_since_best = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_macro_rmse_db": val_rmse_db}, ckpt_dir / "best_model.pt")
        else:
            epochs_since_best += 1
        if epochs_since_best >= patience:
            print(f"  early stopping at epoch {epoch} (best epoch {best_epoch}, val={best_val:.4f} dB)")
            break

    (ckpt_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    return history


def load_best_checkpoint(model: nn.Module, out_dir: Path, device: torch.device) -> int:
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return int(ckpt["epoch"])
