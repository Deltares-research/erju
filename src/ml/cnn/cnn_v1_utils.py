"""Model, dataset, training and evaluation utilities for the FO waveform CNN v1.

Design notes
------------
* The waveform is kept at physical amplitude (no per-event normalization).
* Only the scalar features (distance + optional metadata) are standardized,
  with the scaler fit on the training fold only.
* One waveform is stored per event; the Dataset maps each sensor-level target
  row to its event waveform by row index — waveforms are never duplicated in
  memory per sensor row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class WaveformDataset(Dataset):
    """Sensor-level samples sharing one waveform per event.

    Parameters
    ----------
    waveforms
        Shared float32 array (N_events, T).  Indexed by ``row_indices``.
    row_indices
        For each sample, the waveform row to use (int array, length N_samples).
    scalars
        Standardized scalar features (N_samples, F) float32.
    targets
        Transformed targets (N_samples,) float32 (e.g. log PGV).
    """

    def __init__(
        self,
        waveforms: np.ndarray,
        row_indices: np.ndarray,
        scalars: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self.waveforms = waveforms                       # (N_events, T) shared
        self.row_indices = row_indices.astype(np.int64)
        self.scalars = scalars.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, i: int):
        wave = self.waveforms[self.row_indices[i]]       # (T,)
        wave_t = torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0)  # (1, T)
        scal_t = torch.from_numpy(self.scalars[i])       # (F,)
        y_t = torch.tensor(self.targets[i], dtype=torch.float32)
        return wave_t, scal_t, y_t


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _act(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU()}.get(name.lower(), nn.ReLU())


class WaveformCNN(nn.Module):
    """1D CNN encoder + MLP head predicting a scalar (log PGV_z).

    When ``use_waveform`` is False the encoder is dropped entirely and the model
    becomes a plain MLP on the scalar features — the scalar-only baseline.
    """

    def __init__(
        self,
        n_scalars: int,
        conv_channels: List[int],
        conv_kernels: List[int],
        conv_strides: List[int],
        use_batchnorm: bool,
        conv_dropout: float,
        head_hidden: List[int],
        head_dropout: float,
        activation: str = "relu",
        use_waveform: bool = True,
    ) -> None:
        super().__init__()
        self.use_waveform = use_waveform

        if use_waveform:
            layers: List[nn.Module] = []
            in_ch = 1
            for out_ch, k, s in zip(conv_channels, conv_kernels, conv_strides):
                layers.append(
                    nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=k // 2)
                )
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(out_ch))
                layers.append(_act(activation))
                if conv_dropout > 0:
                    layers.append(nn.Dropout(conv_dropout))
                in_ch = out_ch
            self.encoder = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)           # global average pool
            self.embed_dim = in_ch
        else:
            self.encoder = None
            self.pool = None
            self.embed_dim = 0

        head_layers: List[nn.Module] = []
        h_in = self.embed_dim + n_scalars
        for h in head_hidden:
            head_layers.append(nn.Linear(h_in, h))
            head_layers.append(_act(activation))
            if head_dropout > 0:
                head_layers.append(nn.Dropout(head_dropout))
            h_in = h
        head_layers.append(nn.Linear(h_in, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, wave: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        if self.use_waveform:
            z = self.encoder(wave)                        # (B, C, T')
            z = self.pool(z).squeeze(-1)                  # (B, C)
            h = torch.cat([z, scalars], dim=1)            # (B, C + F)
        else:
            h = scalars                                   # (B, F)
        return self.head(h).squeeze(-1)                   # (B,)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


@dataclass
class TrainArtifacts:
    history: List[Dict[str, float]]
    best_epoch: int
    best_val_loss: float


def train_cnn(
    model: WaveformCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    best_model_path,
    use_lr_scheduler: bool = False,
    writer=None,
    verbose: bool = True,
) -> TrainArtifacts:
    """Standard regression training loop with early stopping on val MSE."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(5, patience // 3)
        )
        if use_lr_scheduler
        else None
    )

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train = 0
        for wave, scal, y in train_loader:
            wave, scal, y = wave.to(device), scal.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(wave, scal)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            bs = y.size(0)
            train_loss_sum += loss.item() * bs
            n_train += bs
        train_loss = train_loss_sum / max(n_train, 1)

        val_loss = evaluate_loss(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(val_loss)

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)

        improved = val_loss < best_val - 1e-7
        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1 or improved):
            flag = "  *" if improved else ""
            print(
                f"  epoch {epoch:4d} | train {train_loss:.4f} | val {val_loss:.4f}{flag}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    return TrainArtifacts(history=history, best_epoch=best_epoch, best_val_loss=best_val)


@torch.no_grad()
def evaluate_loss(
    model: WaveformCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    loss_sum = 0.0
    n = 0
    for wave, scal, y in loader:
        wave, scal, y = wave.to(device), scal.to(device), y.to(device)
        pred = model(wave, scal)
        bs = y.size(0)
        loss_sum += criterion(pred, y).item() * bs
        n += bs
    return loss_sum / max(n, 1)


@torch.no_grad()
def predict(
    model: WaveformCNN,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds: List[np.ndarray] = []
    for wave, scal, _y in loader:
        wave, scal = wave.to(device), scal.to(device)
        preds.append(model(wave, scal).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    pgv_true: np.ndarray,
    pgv_pred: np.ndarray,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Return both PGV-space and log-space metrics."""
    log_true = np.log(np.clip(pgv_true, eps, None))
    log_pred = np.log(np.clip(pgv_pred, eps, None))
    return {
        "rmse_mms": float(np.sqrt(mean_squared_error(pgv_true, pgv_pred))),
        "mae_mms": float(mean_absolute_error(pgv_true, pgv_pred)),
        "r2_mms": float(r2_score(pgv_true, pgv_pred)),
        "rmse_log": float(np.sqrt(mean_squared_error(log_true, log_pred))),
        "mae_log": float(mean_absolute_error(log_true, log_pred)),
        "r2_log": float(r2_score(log_true, log_pred)),
    }
