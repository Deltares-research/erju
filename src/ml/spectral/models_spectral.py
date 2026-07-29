"""models_spectral.py
=====================
Model architectures for MP8 spectral (and PGV) prediction from the 51-channel
FO waveform.

Architectures
-------------
CNN2DEncoder      — four 2D-conv layers, temporal-dominant strides   (S1, S5, P1)
ResNet2DEncoder   — same plan with residual projections               (S2, S3, S5)
SeparableCNNEncoder — global spatial compression then 1D temporal CNN (S4)

All encoders share the same interface:
    forward(wf, n_valid) → embedding (B, embed_dim)
    where wf: (B, 1, 51, 7500) float32 microstrain

Top-level models
----------------
SpectralNet  → 19-band dB output   (S1–S5)
PGVNet       → 1 scalar output     (P1)
MultiTaskNet → 19-band + 1 PGV     (S5)

build_model(cfg, n_meta) → appropriate top-level model
count_parameters(model)  → int
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ml.spectral.config_spectral_v001 import (
    ArchConfig, N_BANDS, N_CHANNELS, N_SAMPLES,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _act(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}.get(
        name.lower(), nn.ReLU()
    )


def _conv2d_out_size(h: int, w: int,
                     kh: int, kw: int, sh: int, sw: int,
                     ph: int, pw: int) -> Tuple[int, int]:
    return (math.floor((h + 2 * ph - kh) / sh) + 1,
            math.floor((w + 2 * pw - kw) / sw) + 1)


# ── Masked temporal pooling ───────────────────────────────────────────────────

class MaskedTemporalPool(nn.Module):
    """Average pool the time dimension over valid (non-padded) time steps.

    The FO waveform is zero-padded at the START; signal runs to the END.
    After a 2D or 1D encoder, the valid region in the downsampled feature map
    is approximated by:
        valid_start = max(0, T_prime - round(n_valid * T_prime / N_SAMPLES))

    Spatial (channel) dimension is averaged normally.
    """

    def __init__(self, n_samples: int = N_SAMPLES) -> None:
        super().__init__()
        self.n_samples = n_samples

    def forward(self, x: torch.Tensor, n_valid: torch.Tensor) -> torch.Tensor:
        """Pool over time (last dim), then over spatial (second-to-last).

        Parameters
        ----------
        x       : (B, C, H, T) for 2D encoder  or  (B, C, T) for 1D encoder
        n_valid : (B,) int — original valid samples at 7500-sample resolution
        """
        is_2d = (x.dim() == 4)
        if not is_2d:
            # Add dummy spatial dim
            x = x.unsqueeze(2)   # (B, C, 1, T)

        B, C, H, T = x.shape
        # Valid time steps at downsampled resolution
        n_valid_t = (n_valid.float() * T / self.n_samples).round().long().clamp(1, T)

        # Vectorised time mask: (B, 1, 1, T)
        t_idx  = torch.arange(T, device=x.device).unsqueeze(0)      # (1, T)
        start  = (T - n_valid_t).unsqueeze(1)                        # (B, 1)
        t_mask = (t_idx >= start).float().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

        masked = x * t_mask                                           # (B, C, H, T)
        denom  = n_valid_t.float().clamp(min=1.0).view(B, 1, 1)      # (B, 1, 1)
        pooled = masked.sum(dim=-1) / denom                           # (B, C, H)
        return pooled.mean(dim=-1)                                    # (B, C)


# ── Metadata branch ───────────────────────────────────────────────────────────

class MetadataBranch(nn.Module):
    """Small MLP over pre-processed metadata features."""

    def __init__(self, n_meta: int, hidden: List[int],
                 dropout: float, activation: str = "relu") -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_d = n_meta
        for h in hidden:
            layers.extend([nn.Linear(in_d, h), _act(activation)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = in_d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Fusion head ───────────────────────────────────────────────────────────────

class FusionHead(nn.Module):
    """Concat(encoder_embed, meta_embed) → MLP → n_out."""

    def __init__(self, in_dim: int, hidden: List[int],
                 n_out: int, dropout: float, activation: str = "relu") -> None:
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.extend([nn.Linear(d, h), _act(activation)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── CNN2D Encoder ─────────────────────────────────────────────────────────────

class CNN2DEncoder(nn.Module):
    """Four-layer 2D CNN on (1, 51, 7500) with temporal-dominant strides.

    Output: (B, embed_dim) after masked temporal + spatial average pool.
    """

    def __init__(self, arch: ArchConfig) -> None:
        super().__init__()
        self.pool = MaskedTemporalPool()

        layers: List[nn.Module] = []
        in_ch = 1
        for out_ch, kc, kt, sc, st in zip(
            arch.conv_channels, arch.kernel_ch, arch.kernel_time,
            arch.stride_ch, arch.stride_time
        ):
            layers.append(
                nn.Conv2d(in_ch, out_ch,
                          kernel_size=(kc, kt),
                          stride=(sc, st),
                          padding=(kc // 2, kt // 2))
            )
            if arch.use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(_act(arch.activation))
            if arch.conv_dropout > 0:
                layers.append(nn.Dropout2d(arch.conv_dropout))
            in_ch = out_ch

        self.encoder = nn.Sequential(*layers)
        self.embed_dim = in_ch

    def forward(self, wf: torch.Tensor, n_valid: torch.Tensor) -> torch.Tensor:
        z = self.encoder(wf)               # (B, C, H, T')
        return self.pool(z, n_valid)       # (B, C)


# ── ResNet2D Encoder ──────────────────────────────────────────────────────────

class _ResBlock2D(nn.Module):
    """Residual block: two Conv2D layers with a skip projection if needed."""

    def __init__(self, in_ch: int, out_ch: int,
                 kc: int, kt: int, sc: int, st: int,
                 use_bn: bool, dropout: float, activation: str) -> None:
        super().__init__()
        pad_c, pad_t = kc // 2, kt // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, (kc, kt), stride=(sc, st),
                               padding=(pad_c, pad_t))
        self.bn1   = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, (kc, kt), stride=(1, 1),
                               padding=(pad_c, pad_t))
        self.bn2   = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act   = _act(activation)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Skip projection if shapes differ
        if in_ch != out_ch or sc != 1 or st != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, (1, 1), stride=(sc, st)),
                nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return self.act(h + self.skip(x))


class ResNet2DEncoder(nn.Module):
    """Residual 2D CNN with same channel plan as CNN2DEncoder."""

    def __init__(self, arch: ArchConfig) -> None:
        super().__init__()
        self.pool = MaskedTemporalPool()

        blocks: List[nn.Module] = []
        in_ch = 1
        for out_ch, kc, kt, sc, st in zip(
            arch.conv_channels, arch.kernel_ch, arch.kernel_time,
            arch.stride_ch, arch.stride_time
        ):
            blocks.append(_ResBlock2D(
                in_ch, out_ch, kc, kt, sc, st,
                arch.use_batchnorm, arch.conv_dropout, arch.activation
            ))
            in_ch = out_ch

        self.encoder  = nn.Sequential(*blocks)
        self.embed_dim = in_ch

    def forward(self, wf: torch.Tensor, n_valid: torch.Tensor) -> torch.Tensor:
        z = self.encoder(wf)
        return self.pool(z, n_valid)


# ── Separable CNN Encoder ─────────────────────────────────────────────────────

class SeparableCNNEncoder(nn.Module):
    """Global spatial compression (51 → 1) followed by 1D temporal CNN.

    Step 0: Conv2d(1, C0, kernel=(51, 1)) → squeeze spatial → (B, C0, T)
    Steps 1-3: 1D Conv on (B, C, T)
    """

    def __init__(self, arch: ArchConfig) -> None:
        super().__init__()
        self.pool = MaskedTemporalPool()

        # Spatial compression layer: full-span across 51 channels, 1 time step
        sp_out = arch.conv_channels[0]
        self.spatial_layer = nn.Sequential(
            nn.Conv2d(1, sp_out, kernel_size=(N_CHANNELS, 1),
                      stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(sp_out) if arch.use_batchnorm else nn.Identity(),
            _act(arch.activation),
        )

        # 1D temporal CNN: layers 1-3
        layers: List[nn.Module] = []
        in_ch = sp_out
        for out_ch, kt, st in zip(
            arch.conv_channels[1:], arch.kernel_time[1:], arch.stride_time[1:]
        ):
            layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=kt, stride=st,
                          padding=kt // 2)
            )
            if arch.use_batchnorm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(_act(arch.activation))
            if arch.conv_dropout > 0:
                layers.append(nn.Dropout(arch.conv_dropout))
            in_ch = out_ch

        self.temporal_layers = nn.Sequential(*layers)
        self.embed_dim = in_ch

    def forward(self, wf: torch.Tensor, n_valid: torch.Tensor) -> torch.Tensor:
        # wf: (B, 1, 51, T)
        z = self.spatial_layer(wf)         # (B, C0, 1, T)
        z = z.squeeze(2)                   # (B, C0, T)
        z = self.temporal_layers(z)        # (B, C_last, T')

        # Masked pool (add dummy spatial dim for MaskedTemporalPool)
        return self.pool(z, n_valid)       # (B, C_last)


# ── Top-level models ──────────────────────────────────────────────────────────

def _build_encoder(arch: ArchConfig) -> nn.Module:
    et = arch.encoder_type.lower()
    if et == "cnn2d":
        return CNN2DEncoder(arch)
    elif et == "resnet2d":
        return ResNet2DEncoder(arch)
    elif et == "separable":
        return SeparableCNNEncoder(arch)
    else:
        raise ValueError(f"Unknown encoder_type: {arch.encoder_type}")


class SpectralNet(nn.Module):
    """FO encoder + metadata branch → 19-band spectral output.
    Used for S1, S2, S3, S4.
    """

    def __init__(self, arch: ArchConfig, n_meta: int, n_out: int = N_BANDS) -> None:
        super().__init__()
        self.encoder  = _build_encoder(arch)
        self.meta_net = MetadataBranch(n_meta, arch.meta_hidden,
                                       arch.meta_dropout, arch.activation)
        fuse_dim = self.encoder.embed_dim + self.meta_net.out_dim
        self.head = FusionHead(fuse_dim, arch.head_hidden, n_out,
                               arch.head_dropout, arch.activation)

    def forward(self,
                wf:      torch.Tensor,   # (B, 1, 51, T)
                meta:    torch.Tensor,   # (B, n_meta)
                n_valid: torch.Tensor,   # (B,) int
                ) -> torch.Tensor:       # (B, n_out)
        enc  = self.encoder(wf, n_valid)
        meta_emb = self.meta_net(meta)
        fused = torch.cat([enc, meta_emb], dim=1)
        return self.head(fused)


class PGVNet(nn.Module):
    """FO encoder + metadata branch → scalar PGV output (log-space).
    Used for P1.
    """

    def __init__(self, arch: ArchConfig, n_meta: int) -> None:
        super().__init__()
        self.encoder  = _build_encoder(arch)
        self.meta_net = MetadataBranch(n_meta, arch.meta_hidden,
                                       arch.meta_dropout, arch.activation)
        fuse_dim = self.encoder.embed_dim + self.meta_net.out_dim
        self.head = FusionHead(fuse_dim, arch.head_hidden, 1,
                               arch.head_dropout, arch.activation)

    def forward(self,
                wf:      torch.Tensor,
                meta:    torch.Tensor,
                n_valid: torch.Tensor,
                ) -> torch.Tensor:       # (B,)
        enc  = self.encoder(wf, n_valid)
        meta_emb = self.meta_net(meta)
        fused = torch.cat([enc, meta_emb], dim=1)
        return self.head(fused).squeeze(-1)


class MultiTaskNet(nn.Module):
    """Shared encoder → spectral head + auxiliary PGV head.
    Used for S5.
    """

    def __init__(self, arch: ArchConfig, n_meta: int, n_spec: int = N_BANDS) -> None:
        super().__init__()
        self.encoder  = _build_encoder(arch)
        self.meta_net = MetadataBranch(n_meta, arch.meta_hidden,
                                       arch.meta_dropout, arch.activation)
        fuse_dim = self.encoder.embed_dim + self.meta_net.out_dim
        self.spec_head = FusionHead(fuse_dim, arch.head_hidden, n_spec,
                                    arch.head_dropout, arch.activation)
        self.pgv_head  = FusionHead(fuse_dim, arch.head_hidden[:1], 1,
                                    arch.head_dropout, arch.activation)

    def forward(self,
                wf:      torch.Tensor,
                meta:    torch.Tensor,
                n_valid: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc  = self.encoder(wf, n_valid)
        meta_emb = self.meta_net(meta)
        fused = torch.cat([enc, meta_emb], dim=1)
        spec = self.spec_head(fused)         # (B, 19)
        pgv  = self.pgv_head(fused).squeeze(-1)  # (B,)
        return spec, pgv


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(cfg, n_meta: int) -> nn.Module:
    """Instantiate the appropriate model for a given experiment config."""
    if cfg.use_pgv_aux:          # S5
        return MultiTaskNet(cfg.arch, n_meta)
    elif cfg.target_type == "pgv":  # P1
        return PGVNet(cfg.arch, n_meta)
    else:                        # S1, S2, S3, S4
        return SpectralNet(cfg.arch, n_meta)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
