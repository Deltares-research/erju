"""Model and dataset for the multi-channel FO waveform CNN (cnn v2).

The training loop, evaluation, prediction and metrics are reused from
``cnn_v1_utils`` (they are model-agnostic).  This module only adds:
  * ``WaveformDataset2D`` — yields (1, C, T) blocks + scalar features + target
  * ``WaveformCNN2D``     — compact 2D CNN over channel x time + MLP head
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.ml.cnn.cnn_v1_utils import _act  # shared activation factory


class WaveformDataset2D(Dataset):
    """Sensor-level samples sharing one (C, T) block per event.

    ``waveforms`` is the shared float32 array (N_events, C, T).  Each sample
    references a block by row index and carries its own scalar features and
    target; blocks are never duplicated per sensor row.
    """

    def __init__(
        self,
        waveforms: np.ndarray,         # (N_events, C, T)
        row_indices: np.ndarray,
        scalars: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self.waveforms = waveforms
        self.row_indices = row_indices.astype(np.int64)
        self.scalars = scalars.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, i: int):
        block = self.waveforms[self.row_indices[i]]          # (C, T)
        block_t = torch.from_numpy(np.ascontiguousarray(block)).unsqueeze(0)  # (1, C, T)
        scal_t = torch.from_numpy(self.scalars[i])
        y_t = torch.tensor(self.targets[i], dtype=torch.float32)
        return block_t, scal_t, y_t


class WaveformCNN2D(nn.Module):
    """Compact 2D CNN over (channel x time) + MLP head -> scalar (log PGV_z).

    With ``use_waveform=False`` the encoder is dropped and the model is a plain
    MLP on the scalar features (scalar-only baseline).
    """

    def __init__(
        self,
        n_scalars: int,
        conv_channels: List[int],
        kernel_ch: List[int],
        kernel_time: List[int],
        stride_ch: List[int],
        stride_time: List[int],
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
            for out_ch, kc, kt, sc, st in zip(
                conv_channels, kernel_ch, kernel_time, stride_ch, stride_time
            ):
                layers.append(
                    nn.Conv2d(
                        in_ch, out_ch,
                        kernel_size=(kc, kt),
                        stride=(sc, st),
                        padding=(kc // 2, kt // 2),
                    )
                )
                if use_batchnorm:
                    layers.append(nn.BatchNorm2d(out_ch))
                layers.append(_act(activation))
                if conv_dropout > 0:
                    layers.append(nn.Dropout(conv_dropout))
                in_ch = out_ch
            self.encoder = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool2d(1)              # -> (B, C, 1, 1)
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

    def forward(self, block: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        if self.use_waveform:
            z = self.encoder(block)                          # (B, C, H', W')
            z = self.pool(z).flatten(1)                      # (B, C)
            h = torch.cat([z, scalars], dim=1)
        else:
            h = scalars
        return self.head(h).squeeze(-1)
