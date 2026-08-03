"""models_m4.py
===============
M4_FO_RESIDUAL_FACTORIZED: FO-signal-only residual correction on top of a
frozen M0 (metadata) baseline, factorized into a single amplitude
(total-level) correction and a low-rank PCA shape-residual correction.

Does NOT modify M0, M1, M2 or M3 -- this module only *loads* M0's and M3's
completed seed-42 checkpoints (read-only) to build M4.

    B[e,j,f]       = frozen M0 prediction (never updated)
    A_target[e]    = total_level_db(L_true[e]) - total_level_db(B[e])
    R_shape[e,j,f] = L_true[e,j,f] - B[e,j,f] - A_target[e]

R_shape is flattened to 95 values (5 sensors x 19 bands) and factorized via
PCA fit on TRAIN residuals only (<=20 components, >=90% cumulative train
variance). M4's trainable FO trunk (WaveformOnlyEncoder, initialized from
M3's completed checkpoint, no metadata/track/distance) predicts a
standardized amplitude scalar and standardized PCA shape coefficients,
reconstructing:

    L_hat = B + A_hat + R_hat
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from src.ml.linec_multisensor.losses import total_level_db
from src.ml.linec_multisensor.models import M0Model, S6_ARCH, WaveformOnlyEncoder
from src.ml.spectral.models_spectral import FusionHead

N_SENSORS = 5
N_BANDS = 19
AUX_HUBER_DELTA = 1.0  # standardized z-units, matches losses.SPECTRAL_HUBER_DELTA convention


@dataclass
class PCAResidualStats:
    """PCA + standardization stats for M4's residual factorization, fit on
    TRAIN-split residuals only. `mean_`/`components_` follow sklearn.PCA
    convention (mean-centered, `components_` shape (k, 95))."""

    n_components: int
    mean_: np.ndarray                       # (95,) residual mean (train)
    components_: np.ndarray                 # (k, 95) PCA loading vectors
    score_std: np.ndarray                   # (k,) train std of each PCA score
    explained_variance_ratio_full: np.ndarray  # (k_fit,) full curve up to max_components
    a_mean: float
    a_std: float

    def save(self, path) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "PCAResidualStats":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


@torch.no_grad()
def fit_residual_pca(m0_model: nn.Module, loader: DataLoader, device: torch.device,
                      max_components: int = 20, variance_threshold: float = 0.90) -> PCAResidualStats:
    """Run frozen M0 over `loader` (must be the TRAIN split only), then fit
    the residual PCA + amplitude/PCA-score standardization stats. Does not
    look at validation or test data."""
    m0_model.eval()
    b_list, tgt_list = [], []
    for wf, meta, nv, tgt, r, track, _ in loader:
        wf_d, meta_d, nv_d, r_d, track_d = (t.to(device) for t in (wf, meta, nv, r, track))
        b = m0_model(wf_d, meta_d, nv_d, r_d, track_d)
        b_list.append(b.cpu())
        tgt_list.append(tgt)
    B = torch.cat(b_list, dim=0)    # (n,5,19)
    L = torch.cat(tgt_list, dim=0)  # (n,5,19)

    a_target_t = total_level_db(L) - total_level_db(B)                        # (n,)
    r_shape_t = (L - B - a_target_t.view(-1, 1, 1)).reshape(L.shape[0], -1)    # (n,95)

    a_target = a_target_t.numpy()
    r_shape = r_shape_t.numpy()
    if not np.isfinite(a_target).all() or not np.isfinite(r_shape).all():
        raise ValueError("Non-finite values in M4 residual-PCA precompute (A_target/R_shape)")

    n_samples, n_features = r_shape.shape
    k_fit = int(min(max_components, n_features, n_samples - 1))
    pca = PCA(n_components=k_fit)
    scores_full = pca.fit_transform(r_shape)  # (n, k_fit)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum_var, variance_threshold) + 1)
    k = min(k, max_components, k_fit)

    score_std = np.clip(scores_full[:, :k].std(axis=0), 1e-6, None)
    a_mean = float(a_target.mean())
    a_std = float(max(a_target.std(), 1e-6))

    return PCAResidualStats(
        n_components=k,
        mean_=pca.mean_.astype(np.float32),
        components_=pca.components_[:k].astype(np.float32),
        score_std=score_std.astype(np.float32),
        explained_variance_ratio_full=pca.explained_variance_ratio_.astype(np.float32),
        a_mean=a_mean, a_std=a_std,
    )


class M4Model(nn.Module):
    """L_hat = B + A_hat + R_hat, B frozen (M0), A_hat/R_hat from an
    FO-signal-only trunk (no metadata/track/distance) initialized from M3."""

    def __init__(self, n_meta: int, m0_state_dict: dict, m3_trunk_state_dict: dict,
                 pca_stats: PCAResidualStats) -> None:
        super().__init__()
        self.frozen_m0 = M0Model(n_meta)
        self.frozen_m0.load_state_dict(m0_state_dict)
        for p in self.frozen_m0.parameters():
            p.requires_grad_(False)
        self.frozen_m0.eval()

        self.trunk = WaveformOnlyEncoder()
        self.trunk.load_state_dict(m3_trunk_state_dict)

        k = pca_stats.n_components
        self.amp_head = FusionHead(self.trunk.out_dim, S6_ARCH.head_hidden, 1,
                                    S6_ARCH.head_dropout, S6_ARCH.activation)
        self.shape_head = FusionHead(self.trunk.out_dim, S6_ARCH.head_hidden, k,
                                      S6_ARCH.head_dropout, S6_ARCH.activation)
        # Zero-init only the final linear layer of each head: guarantees raw
        # output 0 regardless of input, so at initialization A_hat_z=z_hat=0
        # and L_hat reconstructs to B plus the fixed train-mean residual bias
        # (a_mean/pca_mean) -- not FO-dependent, effectively the baseline.
        nn.init.zeros_(self.amp_head.net[-1].weight)
        nn.init.zeros_(self.amp_head.net[-1].bias)
        nn.init.zeros_(self.shape_head.net[-1].weight)
        nn.init.zeros_(self.shape_head.net[-1].bias)

        self.register_buffer("pca_mean", torch.as_tensor(pca_stats.mean_, dtype=torch.float32))
        self.register_buffer("pca_components", torch.as_tensor(pca_stats.components_, dtype=torch.float32))
        self.register_buffer("pca_score_std", torch.as_tensor(pca_stats.score_std, dtype=torch.float32))
        self.register_buffer("a_mean", torch.tensor(float(pca_stats.a_mean), dtype=torch.float32))
        self.register_buffer("a_std", torch.tensor(float(pca_stats.a_std), dtype=torch.float32))

    def train(self, mode: bool = True) -> "M4Model":
        super().train(mode)
        self.frozen_m0.eval()  # frozen baseline never uses dropout, regardless of outer mode
        return self

    def freeze_trunk(self) -> None:
        for p in self.trunk.parameters():
            p.requires_grad_(False)

    def unfreeze_trunk(self) -> None:
        for p in self.trunk.parameters():
            p.requires_grad_(True)

    def forward_full(self, wf: torch.Tensor, meta: torch.Tensor, n_valid: torch.Tensor,
                      r: torch.Tensor, track: torch.Tensor):
        with torch.no_grad():
            b = self.frozen_m0(wf, meta, n_valid, r, track)
        z = self.trunk(wf, n_valid)  # FO-signal-only: no metadata/track/distance
        raw_a = self.amp_head(z).squeeze(-1)
        raw_z = self.shape_head(z)
        a_hat_z = 4.0 * torch.tanh(raw_a / 4.0)
        z_hat = 4.0 * torch.tanh(raw_z / 4.0)

        a_hat = a_hat_z * self.a_std + self.a_mean
        r_hat_flat = (z_hat * self.pca_score_std) @ self.pca_components + self.pca_mean
        r_hat = r_hat_flat.view(-1, N_SENSORS, N_BANDS)

        l_hat = b + a_hat.view(-1, 1, 1) + r_hat
        return l_hat, a_hat_z, z_hat, b

    def forward(self, wf: torch.Tensor, meta: torch.Tensor, n_valid: torch.Tensor,
                r: torch.Tensor, track: torch.Tensor) -> torch.Tensor:
        l_hat, _, _, _ = self.forward_full(wf, meta, n_valid, r, track)
        return l_hat

    def target_transform(self, l_true: torch.Tensor, b: torch.Tensor):
        """Standardized amplitude/PCA-shape training targets from a batch's
        true spectrum and the frozen M0 baseline B (same definition used to
        fit the training-only PCA/standardization stats)."""
        a_target = total_level_db(l_true) - total_level_db(b)
        r_shape = (l_true - b - a_target.view(-1, 1, 1)).reshape(l_true.shape[0], -1)
        z_raw = (r_shape - self.pca_mean) @ self.pca_components.t()
        z_target = z_raw / self.pca_score_std
        a_target_z = (a_target - self.a_mean) / self.a_std
        return a_target_z, z_target, a_target
