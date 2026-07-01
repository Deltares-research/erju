"""
Query-conditioned curve-prior CNN: model, dataset, and loss functions.

Physics-informed prediction:
  y_pred(r) = c_hat  -  n_track * log(r / r0)  +  epsilon_hat(r)

Key differences from the fixed-output CurvePriorCNN2D (P3 model):
  - Processes one (event, query_distance) pair per sample, not one event + 5 distances
  - Residual head takes distance as a continuous input → arbitrary-distance generalisation
  - CurveDataset_Query flattens events × sensors into flat (event, sensor) sample pairs
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ===========================================================================
# MODEL
# ===========================================================================

class CurvePriorCNN2D_Query(nn.Module):
    """Query-conditioned curve-prior CNN for arbitrary-distance PGV prediction.

    Input per forward call (batch of B samples):
        waveform        (B, 1, C, T)  — 21 FO channels, 7 500 time samples
        metadata        (B, n_meta)   — train speed, speed-missing flag, train-type code, track norm
        query_distance  (B,)          — active-track distance in metres
        track           (B,)  long    — 1 or 2

    Outputs:
        c_hat        (B,)            event intensity scalar
        epsilon_hat  (B,) or None    residual at queried distance (None for Q1)

    Prediction:
        y_pred = c_hat  -  n_track * log(r / r0)  +  epsilon_hat
    """

    def __init__(
        self,
        n_metadata:           int,
        conv_channels:        List[int],
        kernel_ch:            List[int],
        kernel_time:          List[int],
        stride_time:          List[int],
        use_batchnorm:        bool,
        metadata_hidden:      List[int],
        intensity_hidden:     List[int],
        query_hidden:         List[int],
        residual_hidden:      List[int],
        enable_residual_head: bool,
        n_query_features:     int = 3,
    ):
        super().__init__()
        self.enable_residual_head = enable_residual_head

        # ── 2D CNN encoder (identical structure to CurvePriorCNN2D) ─────────
        enc: List[nn.Module] = []
        in_ch = 1
        for out_ch, kc, kt, st in zip(
            conv_channels, kernel_ch, kernel_time, stride_time
        ):
            enc.append(
                nn.Conv2d(
                    in_ch, out_ch,
                    kernel_size=(kc, kt),
                    stride=(1, st),
                    padding=(kc // 2, kt // 2),   # explicit — no padding="same" with stride>1
                )
            )
            if use_batchnorm:
                enc.append(nn.BatchNorm2d(out_ch))
            enc.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc)
        self.pool    = nn.AdaptiveAvgPool2d(1)    # (B, embed_dim, 1, 1)
        embed_dim    = conv_channels[-1]            # 64

        # ── Metadata MLP ─────────────────────────────────────────────────────
        meta: List[nn.Module] = []
        in_sz = n_metadata
        for out_sz in metadata_hidden:
            meta.extend([nn.Linear(in_sz, out_sz), nn.ReLU(inplace=True)])
            in_sz = out_sz
        self.metadata_embed = nn.Sequential(*meta)
        meta_dim = in_sz   # 16

        event_dim = embed_dim + meta_dim   # 80

        # ── Intensity head: event_embed → c_hat ──────────────────────────────
        ih: List[nn.Module] = []
        in_sz = event_dim
        for out_sz in intensity_hidden:
            ih.extend([nn.Linear(in_sz, out_sz), nn.ReLU(inplace=True)])
            in_sz = out_sz
        ih.append(nn.Linear(in_sz, 1))
        self.intensity_head = nn.Sequential(*ih)

        # ── Query embedding: [log(r/r0), r/r0, track/2] → z ─────────────────
        qh: List[nn.Module] = []
        in_sz = n_query_features
        for out_sz in query_hidden:
            qh.extend([nn.Linear(in_sz, out_sz), nn.ReLU(inplace=True)])
            in_sz = out_sz
        self.query_embed = nn.Sequential(*qh)
        query_dim = in_sz   # 16

        # ── Residual head: [event_embed, query_embed] → epsilon_hat ──────────
        if enable_residual_head:
            rh: List[nn.Module] = []
            in_sz = event_dim + query_dim   # 96
            for out_sz in residual_hidden:
                rh.extend([nn.Linear(in_sz, out_sz), nn.ReLU(inplace=True)])
                in_sz = out_sz
            rh.append(nn.Linear(in_sz, 1))
            self.residual_head: Optional[nn.Sequential] = nn.Sequential(*rh)
        else:
            self.residual_head = None

    def forward(
        self,
        waveform:       torch.Tensor,    # (B, 1, C, T)
        metadata:       torch.Tensor,    # (B, n_meta)
        query_distance: torch.Tensor,    # (B,)  metres
        track:          torch.Tensor,    # (B,)  long, values 1 or 2
        r0: float = 10.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # CNN encoder
        h = self.encoder(waveform)                     # (B, 64, H', T')
        h = self.pool(h).squeeze(-1).squeeze(-1)       # (B, 64)

        # Metadata
        m = self.metadata_embed(metadata)              # (B, 16)

        # Combined event embedding
        ev = torch.cat([h, m], dim=-1)                 # (B, 80)

        # Intensity
        c_hat = self.intensity_head(ev).squeeze(-1)    # (B,)

        # Residual (optional)
        eps: Optional[torch.Tensor] = None
        if self.residual_head is not None:
            log_r   = torch.log(query_distance / r0)   # (B,)
            r_norm  = query_distance / r0               # (B,)
            tr_norm = track.float() / 2.0              # (B,)  0.5 or 1.0
            q_feat  = torch.stack([log_r, r_norm, tr_norm], dim=-1)  # (B, 3)
            z       = self.query_embed(q_feat)          # (B, 16)
            res     = torch.cat([ev, z], dim=-1)        # (B, 96)
            eps     = self.residual_head(res).squeeze(-1)  # (B,)

        return c_hat, eps


# ===========================================================================
# DATASET
# ===========================================================================

class CurveDataset_Query(Dataset):
    """(event, sensor) pair dataset for the query-conditioned curve-prior model.

    The dataset is flat: each sample is one (event, sensor) pair.
    The same waveform appears once per sensor query per epoch.

    Args:
        waveforms:            (N, C, T)   already subset to the desired split events
        metadata:             (N, n_meta)
        targets_log:          (N, S)      log(PGV_z) per sensor
        distances:            (N, S)      active-track distance (metres)
        tracks:               (N,)        int  1 or 2
        event_ids:            (N,)
        sensor_names:         list[str]   length S
        holdout_sensor:       None → all sensors included
                              str  → exclude this sensor from samples (train/val mode)
        include_holdout_only: True → include ONLY holdout-sensor samples (held-out test)
    """

    def __init__(
        self,
        waveforms:            np.ndarray,
        metadata:             np.ndarray,
        targets_log:          np.ndarray,
        distances:            np.ndarray,
        tracks:               np.ndarray,
        event_ids:            np.ndarray,
        sensor_names:         List[str],
        holdout_sensor:       Optional[str] = None,
        include_holdout_only: bool          = False,
    ):
        self.waveforms    = waveforms.astype(np.float32)
        self.metadata     = metadata.astype(np.float32)
        self.targets_log  = targets_log.astype(np.float32)
        self.distances    = distances.astype(np.float32)
        self.tracks       = tracks.astype(np.int64)
        self.event_ids    = event_ids
        self.sensor_names = sensor_names

        self.samples: List[Tuple[int, int]] = []
        for ev_i in range(len(waveforms)):
            for s_j, sname in enumerate(sensor_names):
                is_ho = holdout_sensor is not None and sname == holdout_sensor
                if include_holdout_only:
                    if is_ho:
                        self.samples.append((ev_i, s_j))
                else:
                    if not is_ho:
                        self.samples.append((ev_i, s_j))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        ev_i, s_j = self.samples[idx]
        wf = torch.from_numpy(np.ascontiguousarray(self.waveforms[ev_i]))
        wf = wf.unsqueeze(0)                                               # (1, C, T)
        return {
            "waveform":        wf,
            "metadata":        torch.from_numpy(self.metadata[ev_i]),
            "target_log":      torch.tensor(self.targets_log[ev_i, s_j], dtype=torch.float32),
            "distance":        torch.tensor(self.distances[ev_i, s_j],   dtype=torch.float32),
            "track":           torch.tensor(self.tracks[ev_i],            dtype=torch.long),
            "event_id":        self.event_ids[ev_i],
            "event_array_idx": ev_i,   # position in the subsetted waveform array
            "sensor_name":     self.sensor_names[s_j],
            "sensor_idx":      s_j,
        }


# ===========================================================================
# LOSS FUNCTIONS
# ===========================================================================

def huber_loss_log(
    pred_log:   torch.Tensor,
    target_log: torch.Tensor,
    delta:   float = 0.5,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Huber loss in log space with optional per-sample weighting."""
    diff = pred_log - target_log
    loss = torch.where(
        diff.abs() <= delta,
        0.5 * diff ** 2,
        delta * (diff.abs() - 0.5 * delta),
    )
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def mse_scalar(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE for scalar outputs (intensity auxiliary loss L_c)."""
    return ((pred - target) ** 2).mean()


def compute_sample_weights(
    sensor_names: List[str],
    mp4_weight:   float,
    device:       Optional[torch.device] = None,
) -> torch.Tensor:
    """Per-sample weight tensor.  MP4 → mp4_weight, all others → 1.0."""
    w = torch.tensor(
        [mp4_weight if s == "MP4" else 1.0 for s in sensor_names],
        dtype=torch.float32,
    )
    return w if device is None else w.to(device)

