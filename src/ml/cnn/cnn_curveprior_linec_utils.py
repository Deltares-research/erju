"""Model, dataset, and utilities for track-conditioned curve-prior + residual CNN.

Physics-informed prediction:
  y_pred = c_hat - n_track * log(r/r0) + epsilon_hat

where:
  c_hat: predicted event intensity
  n_track: track-specific attenuation exponent (fitted on train split)
  epsilon_hat: learned residual correction
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class CurveDataset(Dataset):
    """Event-level dataset with distance vectors for curve-prior model.
    
    Each sample:
      - waveform: (C, T) float32 FO waveform
      - metadata: (n_meta,) float32 features
      - targets: (5,) float32 log-PGV values at sensor distances
      - distances: (5,) float32 distances to each sensor
      - track: int (1 or 2)
      - event_id: str
    """
    
    def __init__(
        self,
        waveforms: np.ndarray,           # (N_events, C, T)
        metadata: np.ndarray,             # (N_events, n_meta)
        targets: np.ndarray,              # (N_events, 5)
        distances: np.ndarray,            # (N_events, 5) distances per sensor
        tracks: np.ndarray,               # (N_events,) int
        event_ids: np.ndarray,            # (N_events,) str
    ):
        self.waveforms = waveforms.astype(np.float32)
        self.metadata = metadata.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.distances = distances.astype(np.float32)
        self.tracks = tracks.astype(np.int64)
        self.event_ids = event_ids
    
    def __len__(self) -> int:
        return len(self.waveforms)
    
    def __getitem__(self, idx: int) -> Tuple:
        wf = torch.from_numpy(np.ascontiguousarray(self.waveforms[idx]))
        wf = wf.unsqueeze(0)  # (C, T) → (1, C, T)
        
        meta = torch.from_numpy(self.metadata[idx])
        tgt = torch.from_numpy(self.targets[idx])
        dist = torch.from_numpy(self.distances[idx])
        trk = torch.tensor(self.tracks[idx], dtype=torch.long)
        evt = self.event_ids[idx]
        
        return wf, meta, tgt, dist, trk, evt


class CurvePriorCNN2D(nn.Module):
    """Track-conditioned curve-prior model with optional residual.
    
    Inputs:
      - waveform: (B, 1, C, T)
      - metadata: (B, n_meta)
      - track: (B,) with values {1, 2}
    
    Outputs:
      - c_hat: (B,) predicted event intensity
      - epsilon_hat: (B, 5) residual corrections (or None if not used)
    """
    
    def __init__(
        self,
        n_metadata: int,
        conv_channels: List[int],
        kernel_ch: List[int],
        kernel_time: List[int],
        stride_ch: List[int],
        stride_time: List[int],
        use_batchnorm: bool,
        conv_dropout: float,
        metadata_hidden: List[int],
        metadata_dropout: float,
        intensity_hidden: List[int],
        intensity_dropout: float,
        residual_hidden: List[int],
        residual_dropout: float,
        n_outputs: int = 5,
        activation: str = "relu",
    ):
        super().__init__()
        self.n_outputs = n_outputs
        self.activation = activation
        
        # =====================================================================
        # Shared 2D CNN encoder
        # =====================================================================
        
        encoder_layers: List[nn.Module] = []
        in_ch = 1
        for out_ch, kc, kt, sc, st in zip(
            conv_channels, kernel_ch, kernel_time, stride_ch, stride_time
        ):
            encoder_layers.append(
                nn.Conv2d(
                    in_ch, out_ch,
                    kernel_size=(kc, kt),
                    stride=(sc, st),
                    padding=(kc // 2, kt // 2),
                )
            )
            if use_batchnorm:
                encoder_layers.append(nn.BatchNorm2d(out_ch))
            encoder_layers.append(self._activation_module())
            if conv_dropout > 0:
                encoder_layers.append(nn.Dropout(conv_dropout))
            in_ch = out_ch
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        embed_dim = in_ch
        
        # =====================================================================
        # Metadata embedding
        # =====================================================================
        
        meta_layers: List[nn.Module] = []
        in_sz = n_metadata
        for out_sz in metadata_hidden:
            meta_layers.append(nn.Linear(in_sz, out_sz))
            meta_layers.append(self._activation_module())
            if metadata_dropout > 0:
                meta_layers.append(nn.Dropout(metadata_dropout))
            in_sz = out_sz
        meta_embed_dim = in_sz if metadata_hidden else 0
        
        self.metadata_embed = nn.Sequential(*meta_layers) if meta_layers else nn.Identity()
        
        combined_dim = embed_dim + meta_embed_dim
        
        # =====================================================================
        # Event intensity head
        # =====================================================================
        
        intensity_layers: List[nn.Module] = []
        in_sz = combined_dim
        for out_sz in intensity_hidden:
            intensity_layers.append(nn.Linear(in_sz, out_sz))
            intensity_layers.append(self._activation_module())
            if intensity_dropout > 0:
                intensity_layers.append(nn.Dropout(intensity_dropout))
            in_sz = out_sz
        intensity_layers.append(nn.Linear(in_sz, 1))  # Single output: c_hat
        
        self.intensity_head = nn.Sequential(*intensity_layers)
        
        # =====================================================================
        # Residual head (optional)
        # =====================================================================
        
        residual_layers: List[nn.Module] = []
        in_sz = combined_dim
        for out_sz in residual_hidden:
            residual_layers.append(nn.Linear(in_sz, out_sz))
            residual_layers.append(self._activation_module())
            if residual_dropout > 0:
                residual_layers.append(nn.Dropout(residual_dropout))
            in_sz = out_sz
        residual_layers.append(nn.Linear(in_sz, n_outputs))  # 5 outputs: epsilon
        
        self.residual_head = nn.Sequential(*residual_layers)
    
    def _activation_module(self) -> nn.Module:
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "elu":
            return nn.ELU()
        else:
            return nn.ReLU()
    
    def forward(
        self,
        wf: torch.Tensor,
        meta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          wf: (B, 1, C, T)
          meta: (B, n_meta)
        
        Returns:
          c_hat: (B,) event intensity
          epsilon_hat: (B, 5) residual corrections
        """
        # Encode waveform
        encoded = self.encoder(wf)  # (B, embed_dim, h, w)
        encoded = self.pool(encoded)  # (B, embed_dim, 1, 1)
        encoded = encoded.view(encoded.size(0), -1)  # (B, embed_dim)
        
        # Encode metadata
        meta_encoded = self.metadata_embed(meta)  # (B, meta_embed_dim)
        
        # Combine
        combined = torch.cat([encoded, meta_encoded], dim=1)  # (B, combined_dim)
        
        # Predict intensity and residuals
        c_hat = self.intensity_head(combined).squeeze(-1)  # (B,)
        epsilon_hat = self.residual_head(combined)  # (B, 5)
        
        return c_hat, epsilon_hat


def huber_loss_log(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 0.5,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Huber loss on log scale with optional per-output weighting.
    
    Args:
      y_pred: (B, n_out) predictions
      y_true: (B, n_out) targets
      mask: (B,) bool, True for samples to include
      delta: Huber loss threshold
      weights: (B, n_out) optional per-output weights
    
    Returns:
      scalar loss
    """
    if not mask.any():
        return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
    
    pred_masked = y_pred[mask]
    true_masked = y_true[mask]
    
    diff = pred_masked - true_masked
    
    # Huber loss
    loss_per_sample = torch.where(
        torch.abs(diff) <= delta,
        0.5 * diff ** 2,
        delta * (torch.abs(diff) - 0.5 * delta)
    )
    
    if weights is not None:
        w_masked = weights[mask]
        loss = (w_masked * loss_per_sample).mean()
    else:
        loss = loss_per_sample.mean()
    
    return loss


def residual_regularization(
    epsilon_hat: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """L2 regularization on residuals.
    
    Args:
      epsilon_hat: (B, 5) residual values
      mask: (B,) bool
    
    Returns:
      scalar loss
    """
    if not mask.any():
        return torch.tensor(0.0, device=epsilon_hat.device, dtype=epsilon_hat.dtype)
    
    epsilon_masked = epsilon_hat[mask]
    loss = (epsilon_masked ** 2).mean()
    
    return loss


def curve_intensity_loss(
    c_hat: torch.Tensor,
    c_target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss for predicted vs target event intensity.
    
    Args:
      c_hat: (B,) predicted intensity
      c_target: (B,) target intensity
      mask: (B,) bool
    
    Returns:
      scalar loss
    """
    if not mask.any():
        return torch.tensor(0.0, device=c_hat.device, dtype=c_hat.dtype)
    
    c_hat_masked = c_hat[mask]
    c_target_masked = c_target[mask]
    
    loss = ((c_hat_masked - c_target_masked) ** 2).mean()
    
    return loss


def monotonicity_penalty(
    y_pred: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Penalty for non-monotonic profiles (predicted should decrease).
    
    Args:
      y_pred: (B, 5) predictions
      mask: (B,) bool
    
    Returns:
      scalar penalty
    """
    if not mask.any():
        return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
    
    pred_masked = y_pred[mask]  # (n_masked, 5)
    
    # Differences between consecutive outputs
    diffs = pred_masked[:, 1:] - pred_masked[:, :-1]  # (n_masked, 4)
    
    # Penalty: ReLU of violations (should be negative, i.e., decreasing)
    violations = torch.relu(diffs)
    penalty = violations.mean()
    
    return penalty


def compute_output_weights(batch_size: int, mp4_weight: float, device: str) -> torch.Tensor:
    """Compute per-output weights for MP4-weighted variants.
    
    Output order: [MP4, MP8, MP10, MP1, MP2]
    MP4 is output index 0.
    """
    weights = torch.ones(batch_size, 5, device=device, dtype=torch.float32)
    weights[:, 0] = mp4_weight
    return weights


def compute_curve_target(
    targets_log: np.ndarray,
    distances: np.ndarray,
    n_track: float,
    r0: float = 10.0,
) -> np.ndarray:
    """Compute target event intensity from targets and fitted n_track.
    
    c_target = mean_j(y_ij + n_track * log(r_j / r0))
    
    Args:
      targets_log: (n_samples, 5) log-PGV targets
      distances: (n_samples, 5) distances per sensor
      n_track: fitted attenuation exponent
      r0: reference distance
    
    Returns:
      c_target: (n_samples,) target intensity values
    """
    # Curve contribution: n_track * log(r / r0)
    curve = n_track * np.log(distances / r0)  # (n_samples, 5)
    
    # Back-solve intensity: c = y + n * log(r / r0)
    intensities = targets_log + curve  # (n_samples, 5)
    
    # Average across sensors
    c_target = intensities.mean(axis=1)  # (n_samples,)
    
    return c_target
