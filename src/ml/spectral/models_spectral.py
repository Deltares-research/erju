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
    ArchConfig,
    N_BANDS,
    N_CHANNELS,
    N_SAMPLES,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _act(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}.get(
        name.lower(), nn.ReLU()
    )


def _conv2d_out_size(
    h: int, w: int, kh: int, kw: int, sh: int, sw: int, ph: int, pw: int
) -> Tuple[int, int]:
    return (
        math.floor((h + 2 * ph - kh) / sh) + 1,
        math.floor((w + 2 * pw - kw) / sw) + 1,
    )


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
        is_2d = x.dim() == 4
        if not is_2d:
            # Add dummy spatial dim
            x = x.unsqueeze(2)  # (B, C, 1, T)

        B, C, H, T = x.shape
        # Valid time steps at downsampled resolution
        n_valid_t = (n_valid.float() * T / self.n_samples).round().long().clamp(1, T)

        # Vectorised time mask: (B, 1, 1, T)
        t_idx = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        start = (T - n_valid_t).unsqueeze(1)  # (B, 1)
        t_mask = (t_idx >= start).float().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

        masked = x * t_mask  # (B, C, H, T)
        denom = n_valid_t.float().clamp(min=1.0).view(B, 1, 1)  # (B, 1, 1)
        pooled = masked.sum(dim=-1) / denom  # (B, C, H)
        return pooled.mean(dim=-1)  # (B, C)


# ── Amplitude-preserving masking and pooling (S6) ─────────────────────────────


def _infer_raw_valid_mask(
    wf: torch.Tensor,
    n_valid: torch.Tensor,
    zero_tol: float = 0.0,
) -> torch.Tensor:
    """Infer valid time samples from the actual waveform.

    Padding in the waveform build is exactly zero.  Deriving the mask from the
    data avoids assuming whether padding is at the start or the end.  If an
    event is entirely zero (unexpected), fall back to the first ``n_valid``
    samples, which matches the audited trailing-padding convention.

    Parameters
    ----------
    wf : (B, 1, 51, T)
    n_valid : (B,)
    zero_tol : absolute tolerance used to identify padding

    Returns
    -------
    bool tensor (B, T)
    """
    with torch.no_grad():
        sample_peak = wf.detach().abs().amax(dim=(1, 2))  # (B, T)
        mask = sample_peak > float(zero_tol)
        empty = ~mask.any(dim=1)
        if empty.any():
            T = wf.shape[-1]
            idx = torch.arange(T, device=wf.device).unsqueeze(0)
            fallback = idx < n_valid.long().clamp(1, T).unsqueeze(1)
            mask = torch.where(empty.unsqueeze(1), fallback, mask)
    return mask


def _resize_time_mask(mask: torch.Tensor, target_t: int) -> torch.Tensor:
    """Downsample a raw boolean time mask while preserving any valid samples."""
    if mask.shape[-1] == target_t:
        return mask
    pooled = F.adaptive_max_pool1d(mask.float().unsqueeze(1), target_t)
    return pooled.squeeze(1) > 0.5


class MaskedStatisticsPool(nn.Module):
    """Concatenate masked mean, standard deviation and maximum activations.

    Unlike mean-only pooling, this retains event amplitude and activation
    spread.  Invalid padded time samples never contribute.
    """

    n_statistics = 3

    def forward(self, x: torch.Tensor, raw_valid_mask: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(2)  # (B, C, 1, T)
        B, C, H, T = x.shape
        time_mask = _resize_time_mask(raw_valid_mask, T)
        mask = time_mask[:, None, None, :].to(dtype=x.dtype)
        count = (time_mask.sum(dim=1).to(dtype=x.dtype) * H).clamp_min(1.0)
        count = count[:, None]  # (B, 1)

        summed = (x * mask).sum(dim=(2, 3))
        mean = summed / count
        second = (x.square() * mask).sum(dim=(2, 3)) / count
        std = (second - mean.square()).clamp_min(0.0).sqrt()

        neg_inf = torch.finfo(x.dtype).min
        maxv = x.masked_fill(~time_mask[:, None, None, :], neg_inf).amax(dim=(2, 3))
        maxv = torch.where(torch.isfinite(maxv), maxv, torch.zeros_like(maxv))
        return torch.cat([mean, std, maxv], dim=1)


class RawAmplitudeFeatures(nn.Module):
    """Cheap amplitude summaries from the valid raw FO waveform.

    For centre, local 11-channel and full 51-channel views, calculate log-RMS,
    log-absolute-peak and optionally log crest factor.  The calculations are
    forced to float32 so they remain stable inside a mixed-precision model.
    """

    def __init__(self, local_half_width: int = 5, include_crest: bool = True) -> None:
        super().__init__()
        self.local_half_width = int(local_half_width)
        self.include_crest = bool(include_crest)
        self.centre_index = N_CHANNELS // 2
        self.n_groups = 3
        self.out_dim = self.n_groups * (3 if self.include_crest else 2)

    @staticmethod
    def _group_stats(
        x: torch.Tensor, time_mask: torch.Tensor, include_crest: bool
    ) -> List[torch.Tensor]:
        # x: (B, Cg, T), time_mask: (B, T)
        mask = time_mask[:, None, :]
        xf = x.float()
        mf = mask.float()
        count = (mf.sum(dim=(1, 2)) * x.shape[1]).clamp_min(1.0)
        rms = ((xf.square() * mf).sum(dim=(1, 2)) / count).clamp_min(0.0).sqrt()
        peak = xf.abs().masked_fill(~mask, 0.0).amax(dim=(1, 2))
        feats = [torch.log1p(rms), torch.log1p(peak)]
        if include_crest:
            crest = peak / rms.clamp_min(1e-8)
            feats.append(torch.log1p(crest))
        return feats

    def forward(self, wf: torch.Tensor, raw_valid_mask: torch.Tensor) -> torch.Tensor:
        # wf: (B, 1, 51, T) -> (B, 51, T)
        x = wf[:, 0]
        c = self.centre_index
        h = self.local_half_width
        groups = [x[:, c : c + 1], x[:, max(0, c - h) : min(N_CHANNELS, c + h + 1)], x]
        features: List[torch.Tensor] = []
        for group in groups:
            features.extend(
                self._group_stats(group, raw_valid_mask, self.include_crest)
            )
        return torch.stack(features, dim=1)


# ── Metadata branch ───────────────────────────────────────────────────────────


class MetadataBranch(nn.Module):
    """Small MLP over pre-processed metadata features."""

    def __init__(
        self, n_meta: int, hidden: List[int], dropout: float, activation: str = "relu"
    ) -> None:
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

    def __init__(
        self,
        in_dim: int,
        hidden: List[int],
        n_out: int,
        dropout: float,
        activation: str = "relu",
    ) -> None:
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
            arch.conv_channels,
            arch.kernel_ch,
            arch.kernel_time,
            arch.stride_ch,
            arch.stride_time,
        ):
            layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=(kc, kt),
                    stride=(sc, st),
                    padding=(kc // 2, kt // 2),
                )
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
        z = self.encoder(wf)  # (B, C, H, T')
        return self.pool(z, n_valid)  # (B, C)


# ── ResNet2D Encoder ──────────────────────────────────────────────────────────


class _ResBlock2D(nn.Module):
    """Residual block: two Conv2D layers with a skip projection if needed."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kc: int,
        kt: int,
        sc: int,
        st: int,
        use_bn: bool,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        pad_c, pad_t = kc // 2, kt // 2
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, (kc, kt), stride=(sc, st), padding=(pad_c, pad_t)
        )
        self.bn1 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, (kc, kt), stride=(1, 1), padding=(pad_c, pad_t)
        )
        self.bn2 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act = _act(activation)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

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
    """Residual 2D CNN with optional amplitude-preserving statistics pooling."""

    def __init__(self, arch: ArchConfig) -> None:
        super().__init__()
        self.auto_valid_mask = bool(getattr(arch, "auto_valid_mask", False))
        self.padding_zero_tol = float(getattr(arch, "padding_zero_tol", 0.0))
        self.pooling = str(getattr(arch, "pooling", "mean")).lower()
        self.pool = (
            MaskedStatisticsPool() if self.pooling == "stats" else MaskedTemporalPool()
        )

        blocks: List[nn.Module] = []
        in_ch = 1
        for out_ch, kc, kt, sc, st in zip(
            arch.conv_channels,
            arch.kernel_ch,
            arch.kernel_time,
            arch.stride_ch,
            arch.stride_time,
        ):
            blocks.append(
                _ResBlock2D(
                    in_ch,
                    out_ch,
                    kc,
                    kt,
                    sc,
                    st,
                    arch.use_batchnorm,
                    arch.conv_dropout,
                    arch.activation,
                )
            )
            in_ch = out_ch

        self.encoder = nn.Sequential(*blocks)
        self.embed_dim = in_ch * (
            MaskedStatisticsPool.n_statistics if self.pooling == "stats" else 1
        )

    def forward(self, wf: torch.Tensor, n_valid: torch.Tensor) -> torch.Tensor:
        raw_mask = None
        if self.auto_valid_mask or self.pooling == "stats":
            raw_mask = _infer_raw_valid_mask(wf, n_valid, self.padding_zero_tol)
        z = self.encoder(wf)
        if self.pooling == "stats":
            return self.pool(z, raw_mask)
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
            nn.Conv2d(
                1, sp_out, kernel_size=(N_CHANNELS, 1), stride=(1, 1), padding=(0, 0)
            ),
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
                nn.Conv1d(in_ch, out_ch, kernel_size=kt, stride=st, padding=kt // 2)
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
        z = self.spatial_layer(wf)  # (B, C0, 1, T)
        z = z.squeeze(2)  # (B, C0, T)
        z = self.temporal_layers(z)  # (B, C_last, T')

        # Masked pool (add dummy spatial dim for MaskedTemporalPool)
        return self.pool(z, n_valid)  # (B, C_last)


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
        self.encoder = _build_encoder(arch)
        self.meta_net = MetadataBranch(
            n_meta, arch.meta_hidden, arch.meta_dropout, arch.activation
        )
        self.use_raw_amp_features = bool(getattr(arch, "use_raw_amp_features", False))
        self.padding_zero_tol = float(getattr(arch, "padding_zero_tol", 0.0))
        self.raw_amp = None
        raw_amp_dim = 0
        if self.use_raw_amp_features:
            self.raw_amp = RawAmplitudeFeatures(
                local_half_width=getattr(arch, "raw_amp_local_half_width", 5),
                include_crest=getattr(arch, "raw_amp_include_crest", True),
            )
            raw_amp_dim = self.raw_amp.out_dim
        fuse_dim = self.encoder.embed_dim + self.meta_net.out_dim + raw_amp_dim
        self.head = FusionHead(
            fuse_dim, arch.head_hidden, n_out, arch.head_dropout, arch.activation
        )

        # Explicit initialization is useful for the no-BatchNorm S6 configuration.
        if not arch.use_batchnorm:
            self.apply(self._init_no_norm)

    @staticmethod
    def _init_no_norm(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        wf: torch.Tensor,  # (B, 1, 51, T)
        meta: torch.Tensor,  # (B, n_meta)
        n_valid: torch.Tensor,  # (B,) int
    ) -> torch.Tensor:  # (B, n_out)
        enc = self.encoder(wf, n_valid)
        meta_emb = self.meta_net(meta)
        parts = [enc, meta_emb]
        if self.raw_amp is not None:
            raw_mask = _infer_raw_valid_mask(wf, n_valid, self.padding_zero_tol)
            parts.append(self.raw_amp(wf, raw_mask).to(dtype=enc.dtype))
        fused = torch.cat(parts, dim=1)
        return self.head(fused)


class PGVNet(nn.Module):
    """FO encoder + metadata branch → scalar PGV output (log-space).
    Used for P1.
    """

    def __init__(self, arch: ArchConfig, n_meta: int) -> None:
        super().__init__()
        self.encoder = _build_encoder(arch)
        self.meta_net = MetadataBranch(
            n_meta, arch.meta_hidden, arch.meta_dropout, arch.activation
        )
        fuse_dim = self.encoder.embed_dim + self.meta_net.out_dim
        self.head = FusionHead(
            fuse_dim, arch.head_hidden, 1, arch.head_dropout, arch.activation
        )

    def forward(
        self,
        wf: torch.Tensor,
        meta: torch.Tensor,
        n_valid: torch.Tensor,
    ) -> torch.Tensor:  # (B,)
        enc = self.encoder(wf, n_valid)
        meta_emb = self.meta_net(meta)
        fused = torch.cat([enc, meta_emb], dim=1)
        return self.head(fused).squeeze(-1)


class MultiTaskNet(nn.Module):
    """Shared encoder → spectral head + auxiliary PGV head.
    Used for S5.
    """

    def __init__(self, arch: ArchConfig, n_meta: int, n_spec: int = N_BANDS) -> None:
        super().__init__()
        self.encoder = _build_encoder(arch)
        self.meta_net = MetadataBranch(
            n_meta, arch.meta_hidden, arch.meta_dropout, arch.activation
        )
        fuse_dim = self.encoder.embed_dim + self.meta_net.out_dim
        self.spec_head = FusionHead(
            fuse_dim, arch.head_hidden, n_spec, arch.head_dropout, arch.activation
        )
        self.pgv_head = FusionHead(
            fuse_dim, arch.head_hidden[:1], 1, arch.head_dropout, arch.activation
        )

    def forward(
        self,
        wf: torch.Tensor,
        meta: torch.Tensor,
        n_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder(wf, n_valid)
        meta_emb = self.meta_net(meta)
        fused = torch.cat([enc, meta_emb], dim=1)
        spec = self.spec_head(fused)  # (B, 19)
        pgv = self.pgv_head(fused).squeeze(-1)  # (B,)
        return spec, pgv


# ── Factory ───────────────────────────────────────────────────────────────────


def build_model(cfg, n_meta: int) -> nn.Module:
    """Instantiate the appropriate model for a given experiment config."""
    if cfg.use_pgv_aux:  # S5
        return MultiTaskNet(cfg.arch, n_meta)
    elif cfg.target_type == "pgv":  # P1
        return PGVNet(cfg.arch, n_meta)
    else:  # S1, S2, S3, S4
        return SpectralNet(cfg.arch, n_meta)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
