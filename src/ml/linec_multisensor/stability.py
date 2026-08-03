"""stability.py
================
Numerical-stability guards shared by the clean (FP32-only) M0/M1/M2 pipeline.

Policy: the moment a non-finite value (NaN/Inf) is detected anywhere
(inputs, predictions, loss components, gradients, validation metrics), a
diagnostic artifact is written and the run aborts immediately. Batches are
NEVER skipped silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch


class NonFiniteError(RuntimeError):
    """Raised the instant a non-finite value is detected; the run must abort."""


def assert_finite(x, name: str) -> None:
    """Raise NonFiniteError if x (tensor or array-like) contains NaN/Inf."""
    if isinstance(x, torch.Tensor):
        ok = bool(torch.isfinite(x).all())
    else:
        ok = bool(np.isfinite(np.asarray(x)).all())
    if not ok:
        raise NonFiniteError(f"Non-finite values detected in '{name}'")


def _tensor_stats(t: torch.Tensor) -> Dict:
    tf = t.detach().float()
    finite = torch.isfinite(tf)
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "n_nan": int(torch.isnan(tf).sum()),
        "n_inf": int(torch.isinf(tf).sum()),
        "min_finite": float(tf[finite].min()) if bool(finite.any()) else None,
        "max_finite": float(tf[finite].max()) if bool(finite.any()) else None,
    }


def save_diagnostic_and_abort(out_dir: Path, stage: str, epoch: int, batch_idx: int,
                               tensors: Dict[str, torch.Tensor], message: str) -> None:
    """Dump tensor stats/values to out_dir, mark the run invalid, then raise.

    This function never returns -- it always raises NonFiniteError.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "stage": stage, "epoch": epoch, "batch_idx": batch_idx, "message": message,
        "tensors": {k: _tensor_stats(v) for k, v in tensors.items() if isinstance(v, torch.Tensor)},
    }
    (out_dir / "diagnostic_nonfinite.json").write_text(json.dumps(summary, indent=2))
    tensor_dump = {k: v.detach().cpu() for k, v in tensors.items() if isinstance(v, torch.Tensor)}
    if tensor_dump:
        torch.save(tensor_dump, out_dir / "diagnostic_nonfinite.pt")
    (out_dir / "RUN_INVALID").write_text(
        f"Run aborted: non-finite value at stage={stage} epoch={epoch} batch={batch_idx}: {message}\n"
    )
    raise NonFiniteError(
        f"[{stage}] epoch={epoch} batch={batch_idx}: {message} (diagnostic saved to {out_dir})"
    )
