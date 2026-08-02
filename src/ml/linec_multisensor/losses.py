"""losses.py
============
Composite loss shared by M0/M1/M2 (fixed weights, no sweep):

    1. Huber spectral loss   over 5 sensors x 19 bands (standardized dB)
    2. Huber total-level loss on event total spectral level (dB)
    3. CCC loss (1 - concordance correlation) on event total spectral level
    4. Shape-only Huber loss (per-sensor dB residual after removing that
       sensor's own event-level total)

All four terms are computed identically from the model's final (B,5,19) dB
output, so M0/M1/M2 are directly comparable.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

SPECTRAL_HUBER_DELTA = 1.0     # standardized (z) units
TOTAL_LEVEL_HUBER_DELTA = 1.5  # dB
SHAPE_HUBER_DELTA = 1.5        # dB

LOSS_WEIGHTS: Dict[str, float] = {
    "spectral": 1.0,
    "total_level": 0.3,
    "ccc": 0.1,
    "shape": 0.3,
}


def total_level_db(x_db: torch.Tensor) -> torch.Tensor:
    """(B,5,19) dB -> (B,) event total level: mean across sensors of each
    sensor's own 10*log10(sum_f 10**(x/10))."""
    power = torch.pow(10.0, x_db / 10.0)
    sensor_total = 10.0 * torch.log10(power.sum(dim=-1).clamp_min(1e-30))  # (B,5)
    return sensor_total.mean(dim=1)  # (B,)


def _per_sensor_total_db(x_db: torch.Tensor) -> torch.Tensor:
    """(B,5,19) dB -> (B,5) per-sensor total level."""
    power = torch.pow(10.0, x_db / 10.0)
    return 10.0 * torch.log10(power.sum(dim=-1).clamp_min(1e-30))


def _ccc_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """1 - Lin's concordance correlation coefficient, batch-level."""
    pm, tm = pred.mean(), true.mean()
    pv, tv = pred.var(unbiased=False), true.var(unbiased=False)
    cov = ((pred - pm) * (true - tm)).mean()
    ccc = (2.0 * cov) / (pv + tv + (pm - tm) ** 2 + 1e-12)
    return 1.0 - ccc


def composite_loss(pred_db: torch.Tensor, true_db: torch.Tensor,
                    target_mean: torch.Tensor, target_std: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """pred_db/true_db: (B,5,19) raw dB. target_mean/target_std: (5,19)."""
    pred_std = (pred_db - target_mean) / target_std
    true_std = (true_db - target_mean) / target_std
    l_spec = F.huber_loss(pred_std, true_std, delta=SPECTRAL_HUBER_DELTA)

    t_pred = total_level_db(pred_db)
    t_true = total_level_db(true_db)
    l_total = F.huber_loss(t_pred, t_true, delta=TOTAL_LEVEL_HUBER_DELTA)
    l_ccc = _ccc_loss(t_pred, t_true)

    pred_shape = pred_db - _per_sensor_total_db(pred_db).unsqueeze(-1)
    true_shape = true_db - _per_sensor_total_db(true_db).unsqueeze(-1)
    l_shape = F.huber_loss(pred_shape, true_shape, delta=SHAPE_HUBER_DELTA)

    w = LOSS_WEIGHTS
    total = w["spectral"] * l_spec + w["total_level"] * l_total + w["ccc"] * l_ccc + w["shape"] * l_shape
    components = {
        "spectral": float(l_spec.item()), "total_level": float(l_total.item()),
        "ccc": float(l_ccc.item()), "shape": float(l_shape.item()), "total": float(total.item()),
    }
    return total, components
