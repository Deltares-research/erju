"""mlp_utils.py — Dataset, model, training loop, and evaluation helpers for MLP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from src.ml.xgboost.xgb_utils import (
    engineer_features,
    make_event_level_test_split,
    prepare_features,
)


# ── Event-level validation split ─────────────────────────────────────────────


def make_event_level_val_split(
    df: pd.DataFrame,
    group_col: str,
    val_fraction: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split train_val DataFrame into train / val at the event level."""
    rng = np.random.default_rng(random_seed + 1)  # different seed from test split
    all_events = df[group_col].unique()
    n_val = max(1, int(len(all_events) * val_fraction))
    val_events = set(rng.choice(all_events, size=n_val, replace=False))
    mask_val = df[group_col].isin(val_events)
    return df[~mask_val].reset_index(drop=True), df[mask_val].reset_index(drop=True)


# ── Feature scaling ───────────────────────────────────────────────────────────


def fit_imputer(X_train: np.ndarray) -> SimpleImputer:
    """Fit a median imputer on training data only (handles NaN in features)."""
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    return imputer


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on training data only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# ── PyTorch Dataset ───────────────────────────────────────────────────────────


def make_tensor_dataset(
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> TensorDataset:
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    return TensorDataset(X_t, y_t)


# ── Model definition ──────────────────────────────────────────────────────────

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        act_cls = ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Training loop ─────────────────────────────────────────────────────────────


def train_one_epoch(
    model: MLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Run one training epoch, return mean train loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)  # type: ignore[arg-type]


@torch.no_grad()
def evaluate(
    model: MLP,
    loader: DataLoader,
    criterion: nn.Module,
) -> float:
    """Evaluate model on a DataLoader, return mean loss."""
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)  # type: ignore[arg-type]


@torch.no_grad()
def predict_numpy(model: MLP, loader: DataLoader) -> np.ndarray:
    """Collect raw model predictions (in log-space if target was transformed)."""
    model.eval()
    preds = []
    for X_batch, _ in loader:
        preds.append(model(X_batch).cpu().numpy())
    return np.concatenate(preds, axis=0).squeeze()


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def save_checkpoint(
    path: Path,
    epoch: int,
    model: MLP,
    optimizer: torch.optim.Optimizer,
    val_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: MLP,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ── Full training run ─────────────────────────────────────────────────────────


def train_mlp(
    model: MLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg_train: Any,
    build_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Train the MLP with early stopping and TensorBoard logging.

    Returns a summary dict with best epoch, best val_loss, history lists.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg_train.learning_rate,
        weight_decay=cfg_train.weight_decay,
    )

    ckpt_dir = build_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    tb_dir = build_dir / cfg_train.tensorboard_subdir
    writer = SummaryWriter(log_dir=str(tb_dir))

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_epoch = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    print(
        f"\nTraining for up to {cfg_train.epochs} epochs "
        f"(patience={cfg_train.patience}) …"
    )
    print(f'  TensorBoard: tensorboard --logdir "{tb_dir}"')

    for epoch in range(1, cfg_train.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # Best-model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(
                build_dir / "best_model.pt", epoch, model, optimizer, val_loss
            )
        else:
            epochs_no_improve += 1

        # Periodic checkpoint
        if epoch % cfg_train.checkpoint_every_n_epochs == 0:
            save_checkpoint(
                ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                epoch,
                model,
                optimizer,
                val_loss,
            )

        # Console progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            flag = " <-- best" if epoch == best_epoch else ""
            print(
                f"  Epoch {epoch:4d}/{cfg_train.epochs} | "
                f"train={train_loss:.5f}  val={val_loss:.5f}{flag}"
            )

        # Early stopping
        if epochs_no_improve >= cfg_train.patience:
            print(
                f"\n  Early stopping at epoch {epoch} "
                f"(no improvement for {cfg_train.patience} epochs)"
            )
            break

    writer.close()
    print(f"\n  Best val loss: {best_val_loss:.5f} at epoch {best_epoch}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


# ── RMSE in original units ────────────────────────────────────────────────────


def rmse_original_units(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    log_transform: bool,
) -> float:
    """Compute RMSE in mm/s (undoing log1p if used)."""
    if log_transform:
        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(y_pred_log)
    else:
        y_true = y_true_log
        y_pred = y_pred_log
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
