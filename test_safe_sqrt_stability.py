"""Focused tests for the SAFE_SQRT_EPS numerical-stability fix.

Verifies that MaskedStatisticsPool's variance->std and RawAmplitudeFeatures'
mean-square->RMS operations produce finite forward outputs AND finite
backward gradients for degenerate (zero-variance / all-zero) inputs, while
still producing sensible results for normal, non-constant inputs.
"""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.ml.spectral.models_spectral import (
    SAFE_SQRT_EPS,
    MaskedStatisticsPool,
    RawAmplitudeFeatures,
)

print("\n" + "=" * 80)
print("TESTS: SAFE_SQRT_EPS numerical stability")
print("=" * 80)
print(f"SAFE_SQRT_EPS = {SAFE_SQRT_EPS}")

failures = 0


def check(name: str, cond: bool) -> None:
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        failures += 1


def finite_forward_backward(out: torch.Tensor, x: torch.Tensor, label: str) -> None:
    check(f"{label}: forward finite", bool(torch.isfinite(out).all()))
    loss = out.sum()
    loss.backward()
    check(f"{label}: backward finite", x.grad is not None and bool(torch.isfinite(x.grad).all()))


# ── 1. MaskedStatisticsPool: constant input (exact zero variance) ────────────
print("\n[1] MaskedStatisticsPool -- constant input (zero variance)")
pool = MaskedStatisticsPool()
B, C, H, T = 2, 3, 1, 8
x_const = torch.full((B, C, H, T), 5.0, requires_grad=True)
mask = torch.ones(B, T, dtype=torch.bool)
out = pool(x_const, mask)
finite_forward_backward(out, x_const, "constant input")

# ── 2. MaskedStatisticsPool: all-zero input ───────────────────────────────────
print("\n[2] MaskedStatisticsPool -- all-zero input")
x_zero = torch.zeros((B, C, H, T), requires_grad=True)
out = pool(x_zero, mask)
finite_forward_backward(out, x_zero, "zero input")

# ── 3. MaskedStatisticsPool: normal non-constant input still sensible ────────
print("\n[3] MaskedStatisticsPool -- normal non-constant input")
torch.manual_seed(0)
x_rand = (torch.randn((B, C, H, T)) * 10.0 + 3.0).requires_grad_()
out = pool(x_rand, mask)
finite_forward_backward(out, x_rand, "random input")
std_out = out[:, C:2 * C]
with torch.no_grad():
    expected_std = x_rand.detach().std(dim=-1, unbiased=False).squeeze(-1)
close = torch.allclose(std_out, expected_std, atol=1e-2, rtol=1e-2)
check("random input: std matches unbiased-population std within tol", close)

# ── 4. RawAmplitudeFeatures._group_stats: constant and zero groups ──────────
print("\n[4] RawAmplitudeFeatures._group_stats -- constant and zero groups")
Bg, Cg, Tg = 2, 5, 8
x_g_const = torch.full((Bg, Cg, Tg), 5.0, requires_grad=True)
tmask = torch.ones(Bg, Tg, dtype=torch.bool)
feats = RawAmplitudeFeatures._group_stats(x_g_const, tmask, include_crest=True)
out = torch.stack(feats, dim=1)
finite_forward_backward(out, x_g_const, "constant group")

x_g_zero = torch.zeros((Bg, Cg, Tg), requires_grad=True)
feats = RawAmplitudeFeatures._group_stats(x_g_zero, tmask, include_crest=True)
out = torch.stack(feats, dim=1)
finite_forward_backward(out, x_g_zero, "zero group")

# ── 5. RawAmplitudeFeatures._group_stats: normal non-constant group ──────────
print("\n[5] RawAmplitudeFeatures._group_stats -- normal non-constant group")
x_g_rand = (torch.randn((Bg, Cg, Tg)) * 2.0).requires_grad_()
feats = RawAmplitudeFeatures._group_stats(x_g_rand, tmask, include_crest=True)
out = torch.stack(feats, dim=1)
finite_forward_backward(out, x_g_rand, "random group")

# ── 6. Full RawAmplitudeFeatures module, end to end ──────────────────────────
print("\n[6] RawAmplitudeFeatures -- full forward/backward, constant waveform")
raw_amp = RawAmplitudeFeatures()
wf_const = torch.full((2, 1, 51, 32), 1.0, requires_grad=True)
raw_mask = torch.ones(2, 32, dtype=torch.bool)
out = raw_amp(wf_const, raw_mask)
finite_forward_backward(out, wf_const, "full module, constant waveform")

print("\n" + "=" * 80)
if failures:
    print(f"RESULT: {failures} check(s) FAILED")
    sys.exit(1)
else:
    print("RESULT: all checks PASSED")
print("=" * 80)
