"""models.py
============
M0 (metadata baseline), M1 (direct waveform model) and M2 (O0 source-
propagation model), all producing a uniform (B, 5, 19) dB output so the same
loss/evaluation code applies to all three.

Reuses (read-only import, not modified):
    src.ml.spectral.models_spectral.{ResNet2DEncoder, MetadataBranch,
        FusionHead, RawAmplitudeFeatures, _infer_raw_valid_mask, count_parameters}
    src.ml.spectral.config_spectral_v001.get_config("S6_amp").arch
        (the exact S6 no-BatchNorm architecture hyperparameters)

Clean rerun (2026-08-03): FP32 only, no autocast/bfloat16 anywhere. M2's
shape normalisation uses a float32 F.log_softmax (numerically stable for any
logit magnitude).
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ml.spectral.config_spectral_v001 import get_config
from src.ml.spectral.models_spectral import (
    FusionHead,
    MetadataBranch,
    RawAmplitudeFeatures,
    ResNet2DEncoder,
    _infer_raw_valid_mask,
    count_parameters,
)

N_SENSORS = 5
N_BANDS = 19
R0 = 10.0

# Exact S6_amp architecture, reused read-only (no modification to config_spectral_v001.py)
S6_ARCH = get_config("S6_amp").arch


class WaveformMetaEncoder(nn.Module):
    """Shared M1/M2 trunk: S6 no-BatchNorm ResNet2D encoder + metadata MLP + raw-amp features."""

    def __init__(self, n_meta: int) -> None:
        super().__init__()
        arch = S6_ARCH
        self.encoder = ResNet2DEncoder(arch)
        self.meta_net = MetadataBranch(n_meta, arch.meta_hidden, arch.meta_dropout, arch.activation)
        self.raw_amp = RawAmplitudeFeatures(arch.raw_amp_local_half_width, arch.raw_amp_include_crest)
        self.padding_zero_tol = float(arch.padding_zero_tol)
        self.out_dim = self.encoder.embed_dim + self.meta_net.out_dim + self.raw_amp.out_dim

    def forward(self, wf: torch.Tensor, meta: torch.Tensor, n_valid: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(wf, n_valid)
        meta_emb = self.meta_net(meta)
        raw_mask = _infer_raw_valid_mask(wf, n_valid, self.padding_zero_tol)
        amp = self.raw_amp(wf, raw_mask).to(dtype=enc.dtype)
        return torch.cat([enc, meta_emb, amp], dim=1)


class M0Model(nn.Module):
    """Metadata-only baseline: MLP(meta) -> (B, 5, 19). No waveform input."""

    def __init__(self, n_meta: int, hidden: List[int] = (64, 64)) -> None:
        super().__init__()
        self.meta_net = MetadataBranch(n_meta, list(hidden), dropout=0.1, activation="silu")
        self.head = nn.Linear(self.meta_net.out_dim, N_SENSORS * N_BANDS)

    def forward(self, wf: torch.Tensor, meta: torch.Tensor, n_valid: torch.Tensor,
                r: torch.Tensor, track: torch.Tensor) -> torch.Tensor:
        del wf, n_valid, r, track  # unused: metadata-only model, kept for uniform interface
        z = self.meta_net(meta)
        return self.head(z).view(-1, N_SENSORS, N_BANDS)


class M1Model(nn.Module):
    """Direct waveform model: S6 encoder + metadata -> (B, 5, 19)."""

    def __init__(self, n_meta: int) -> None:
        super().__init__()
        self.trunk = WaveformMetaEncoder(n_meta)
        self.head = FusionHead(self.trunk.out_dim, S6_ARCH.head_hidden, N_SENSORS * N_BANDS,
                                S6_ARCH.head_dropout, S6_ARCH.activation)

    def forward(self, wf: torch.Tensor, meta: torch.Tensor, n_valid: torch.Tensor,
                r: torch.Tensor, track: torch.Tensor) -> torch.Tensor:
        del r, track  # unused: no physics decode in M1
        z = self.trunk(wf, meta, n_valid)
        return self.head(z).view(-1, N_SENSORS, N_BANDS)


class M2Model(nn.Module):
    """O0 source-propagation model: same trunk as M1, but predicts T_hat + shape
    logits, energy-normalises the shape, and decodes all 5 sensors with a FROZEN
    O0 geometric-spreading exponent n_track[f] loaded from the completed oracle.

        C_hat[e,f] = T_hat[e] + S_hat[e,f]         (S_hat energy-normalised)
        L_hat[e,j,f] = C_hat[e,f] - 20*n_O0[track,f]*log10(r[e,j]/r0)

    No alpha term, no sensor-specific correction, no event-dependent
    attenuation, no near-field residual.
    """

    def __init__(self, n_meta: int, n_o0: torch.Tensor, r0: float = R0) -> None:
        super().__init__()
        self.trunk = WaveformMetaEncoder(n_meta)
        self.head = FusionHead(self.trunk.out_dim, S6_ARCH.head_hidden, 1 + N_BANDS,
                                S6_ARCH.head_dropout, S6_ARCH.activation)
        self.register_buffer("n_o0", n_o0)  # (2, 19) frozen; row0=track1, row1=track2
        self.r0 = float(r0)

    def forward(self, wf: torch.Tensor, meta: torch.Tensor, n_valid: torch.Tensor,
                r: torch.Tensor, track: torch.Tensor) -> torch.Tensor:
        z = self.trunk(wf, meta, n_valid)
        out = self.head(z).float()   # clean run: fp32 only
        t_hat = out[:, 0]            # (B,)
        shape_logits = out[:, 1:]    # (B, 19)

        # Energy-normalise via float32 log_softmax: sum_f 10**(S_hat[f]/10) = 1
        ln10_10 = math.log(10.0) / 10.0
        log_probs = F.log_softmax(shape_logits * ln10_10, dim=1)  # natural-log domain
        s_hat = log_probs / ln10_10                   # (B, 19) energy-normalised dB shape
        c_hat = t_hat.unsqueeze(1) + s_hat             # (B, 19) source spectrum at r0

        n_idx = (track - 1).clamp(0, 1)               # track {1,2} -> row {0,1}
        n_sel = self.n_o0[n_idx]                       # (B, 19) frozen
        corr = -20.0 * n_sel.unsqueeze(1) * torch.log10(r.unsqueeze(-1) / self.r0)  # (B,5,19)
        return c_hat.unsqueeze(1) + corr               # (B, 5, 19)


def build_model(name: str, n_meta: int, n_o0: torch.Tensor = None) -> nn.Module:
    if name == "M0":
        return M0Model(n_meta)
    if name == "M1":
        return M1Model(n_meta)
    if name == "M2":
        if n_o0 is None:
            raise ValueError("M2 requires n_o0 (frozen O0 propagation exponents)")
        return M2Model(n_meta, torch.as_tensor(n_o0, dtype=torch.float32))
    raise ValueError(f"Unknown model name: {name}")


__all__ = ["M0Model", "M1Model", "M2Model", "build_model", "count_parameters", "S6_ARCH"]
