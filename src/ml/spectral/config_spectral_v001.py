"""config_spectral_v001.py
========================
Experiment configurations for MP8 spectral prediction from the 51-channel FO
waveform.

Six named configurations
------------------------
S1  2D CNN baseline, equal-weight MSE, standard fusion
S2  2D CNN + residual blocks, equal-weight MSE
S3  ResNet encoder, predicts residual relative to frozen metadata baseline
S4  Separable CNN (spatial compression then 1D temporal), MSE
S5  ResNet encoder, multi-task: 19 spectral bands + auxiliary raw-PGV head
P1  Matched direct-PGV control (same CNN2D encoder as S1, single PGV output)

Fixed invariants across all configs
-------------------------------------
* WF build  : holten_waveform_v003_ch51_20260626_094551
* Spec DB   : holten_spectral_targets_v002_corrected
* Alignment : spectral_fo_alignment_v001
* Events    : 1,697 aligned events (train=1103, val=254, test=340)
* Sensor    : MP8 only
* Waveform  : (51, 7500) × WF_SCALE=1e6  (microstrain)
* Meta feats: train_family (OHE-7), track_number (OHE-1), speed (z-scored),
              log_mp8_distance (1), valid_fraction (1)  → n_meta=11
* Target    : velocity_band_level_db, 19 primary bands, z-score standardized
              (S3: target is residual after subtracting metadata-baseline pred)
* Loss      : MSE on standardized targets (S5 adds aux PGV term)
* Optimizer : AdamW, lr=3e-4, weight_decay=1e-4
* Scheduler : CosineAnnealingLR, T_max=cfg.epochs
* Mixed prec: bfloat16 (H100)
* Seed      : 42 for initial sweep; 41,42,43 for finalist robustness
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


# ── Platform-aware path helpers ───────────────────────────────────────────────

def _data_root() -> Path:
    return Path(r"P:\11210978-erju-ai") if os.name == "nt" else Path("/p/11210978-erju-ai")


WF_BUILD_NAME  = "holten_waveform_v003_ch51_20260626_094551"
SPEC_DIR_NAME  = "holten_spectral_targets_v002_corrected"
ALIGN_DIR_NAME = "holten_models/outputs/spectral_fo_alignment_v001"
OUT_BASE_NAME  = "holten_models/outputs/mp8_spectral_cnn_v001"

# Fixed geometry: MP8 distances per track (from sites/holten.json)
MP8_DIST_TRACK1 = 4.0   # metres
MP8_DIST_TRACK2 = 8.0   # metres

# FO waveform constants (consistent with all other scripts)
WF_SCALE   = 1.0e6    # strain → microstrain
N_CHANNELS = 51
N_SAMPLES  = 7500
FS_WF      = 250.0

# Target
N_BANDS    = 19
PRIMARY_NOMINALS = [1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0,
                    12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0]
V_REF      = 1e-6     # mm/s reference velocity for dB

# Metadata baseline frozen results (from train_meta_spectral_v001.py)
META_BASELINE_MACRO_RMSE = 6.26   # dB
META_BASELINE_MACRO_R2   = 0.114
META_BASELINE_TOTAL_RMS_RMSE = 0.1256  # mm/s
META_BASELINE_TOTAL_RMS_R2   = -0.024

# Per-band metadata baseline RMSE (order: PRIMARY_NOMINALS)
META_BASELINE_RMSE_PER_BAND = [
    5.7233, 6.8654, 8.2041, 8.5330, 8.2652, 9.4974, 7.5197, 5.7282, 4.8060,
    4.5631, 4.3345, 5.0785, 5.3444, 4.2955, 5.8823, 7.2713, 4.0539, 3.8466,
    9.0602,
]


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class ArchConfig:
    """Architecture hyperparameters."""
    # Encoder type: "cnn2d" | "resnet2d" | "separable"
    encoder_type: str = "cnn2d"
    # Conv2D channel depths
    conv_channels: List[int] = field(default_factory=lambda: [16, 32, 64, 64])
    # Spatial (channel) kernel sizes per layer
    kernel_ch:    List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    # Temporal kernel sizes per layer
    kernel_time:  List[int] = field(default_factory=lambda: [15, 9, 7, 5])
    # Spatial strides
    stride_ch:    List[int] = field(default_factory=lambda: [1, 1, 1, 2])
    # Temporal strides
    stride_time:  List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    use_batchnorm: bool = True
    conv_dropout:  float = 0.1
    activation:    str = "relu"
    # Metadata MLP
    meta_hidden:   List[int] = field(default_factory=lambda: [32, 32])
    meta_dropout:  float = 0.1
    # Fusion head
    head_hidden:   List[int] = field(default_factory=lambda: [64, 32])
    head_dropout:  float = 0.2


@dataclass
class TrainHyperparams:
    """Training hyperparameters."""
    lr:           float = 3e-4
    weight_decay: float = 1e-4
    batch_size:   int   = 32
    epochs:       int   = 200
    patience:     int   = 30
    grad_clip:    float = 1.0        # max grad norm (0 = disabled)
    scheduler:    str   = "cosine"   # "cosine" | "plateau" | "none"
    loss:         str   = "mse"      # "mse" | "huber"
    huber_delta:  float = 1.5
    mixed_prec:   bool  = True       # bfloat16 on H100


@dataclass
class SpectralExperimentConfig:
    """Full config for one experiment run."""

    # ── Identity ────────────────────────────────────────────────────────────
    name:        str = "S1"
    description: str = "2D CNN baseline"
    seed:        int = 42

    # ── Target formulation ──────────────────────────────────────────────────
    # "spectral"  → 19-band velocity_band_level_db (z-score standardized)
    # "pgv"       → raw_pgv_z_mms (log-transformed, z-score standardized)
    target_type: str = "spectral"

    # ── Special formulations ────────────────────────────────────────────────
    # S3: subtract frozen metadata-baseline prediction before training
    use_meta_residual: bool = False
    # S5: add auxiliary PGV head
    use_pgv_aux:       bool = False
    pgv_aux_weight:    float = 0.2   # λ in: L = L_spec + λ * L_pgv

    # ── Sub-configs ─────────────────────────────────────────────────────────
    arch:  ArchConfig      = field(default_factory=ArchConfig)
    train: TrainHyperparams = field(default_factory=TrainHyperparams)

    # ── Data workers ────────────────────────────────────────────────────────
    num_workers: int = 4

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ── Named factory ─────────────────────────────────────────────────────────────

def get_config(name: str, seed: int = 42) -> SpectralExperimentConfig:
    """Return a named experiment configuration."""

    # ── S1: 2D CNN baseline ──────────────────────────────────────────────────
    if name == "S1":
        return SpectralExperimentConfig(
            name="S1",
            description="2D CNN baseline — standard fusion, MSE loss",
            seed=seed,
            target_type="spectral",
            arch=ArchConfig(
                encoder_type="cnn2d",
                conv_channels=[16, 32, 64, 64],
                kernel_ch=[3, 3, 3, 3],
                kernel_time=[15, 9, 7, 5],
                stride_ch=[1, 1, 1, 2],
                stride_time=[2, 2, 2, 2],
                use_batchnorm=True,
                conv_dropout=0.1,
                meta_hidden=[32, 32],
                head_hidden=[64, 32],
                head_dropout=0.2,
            ),
            train=TrainHyperparams(
                lr=3e-4, weight_decay=1e-4, batch_size=32,
                epochs=200, patience=30,
                loss="mse", scheduler="cosine", mixed_prec=True,
            ),
        )

    # ── S2: 2D CNN with residual blocks ──────────────────────────────────────
    elif name == "S2":
        return SpectralExperimentConfig(
            name="S2",
            description="2D ResNet encoder — residual blocks, MSE loss",
            seed=seed,
            target_type="spectral",
            arch=ArchConfig(
                encoder_type="resnet2d",
                conv_channels=[16, 32, 64, 64],
                kernel_ch=[3, 3, 3, 3],
                kernel_time=[15, 9, 7, 5],
                stride_ch=[1, 1, 1, 2],
                stride_time=[2, 2, 2, 2],
                use_batchnorm=True,
                conv_dropout=0.1,
                meta_hidden=[32, 32],
                head_hidden=[64, 32],
                head_dropout=0.2,
            ),
            train=TrainHyperparams(
                lr=3e-4, weight_decay=1e-4, batch_size=32,
                epochs=200, patience=30,
                loss="mse", scheduler="cosine", mixed_prec=True,
            ),
        )

    # ── S3: ResNet + metadata residual prediction ─────────────────────────────
    elif name == "S3":
        return SpectralExperimentConfig(
            name="S3",
            description="ResNet encoder predicting residual relative to frozen metadata baseline",
            seed=seed,
            target_type="spectral",
            use_meta_residual=True,
            arch=ArchConfig(
                encoder_type="resnet2d",
                conv_channels=[16, 32, 64, 64],
                kernel_ch=[3, 3, 3, 3],
                kernel_time=[15, 9, 7, 5],
                stride_ch=[1, 1, 1, 2],
                stride_time=[2, 2, 2, 2],
                use_batchnorm=True,
                conv_dropout=0.1,
                meta_hidden=[32, 32],
                head_hidden=[64, 32],
                head_dropout=0.2,
            ),
            train=TrainHyperparams(
                lr=3e-4, weight_decay=1e-4, batch_size=32,
                epochs=200, patience=30,
                loss="mse", scheduler="cosine", mixed_prec=True,
            ),
        )

    # ── S4: Separable CNN (spatial compression + 1D temporal) ────────────────
    elif name == "S4":
        return SpectralExperimentConfig(
            name="S4",
            description="Separable CNN — global spatial compression then 1D temporal",
            seed=seed,
            target_type="spectral",
            arch=ArchConfig(
                encoder_type="separable",
                conv_channels=[32, 64, 64, 64],   # after spatial layer + 3 temporal
                kernel_ch=[N_CHANNELS, 1, 1, 1],   # only first is spatial (full-span)
                kernel_time=[1, 9, 7, 5],           # temporal kernels
                stride_ch=[1, 1, 1, 1],
                stride_time=[1, 2, 2, 2],
                use_batchnorm=True,
                conv_dropout=0.1,
                meta_hidden=[32, 32],
                head_hidden=[64, 32],
                head_dropout=0.2,
            ),
            train=TrainHyperparams(
                lr=3e-4, weight_decay=1e-4, batch_size=32,
                epochs=200, patience=30,
                loss="mse", scheduler="cosine", mixed_prec=True,
            ),
        )

    # ── S5: ResNet + multi-task (spectral + auxiliary PGV) ───────────────────
    elif name == "S5":
        return SpectralExperimentConfig(
            name="S5",
            description="ResNet encoder, multi-task: 19 spectral bands + auxiliary raw-PGV",
            seed=seed,
            target_type="spectral",
            use_pgv_aux=True,
            pgv_aux_weight=0.2,
            arch=ArchConfig(
                encoder_type="resnet2d",
                conv_channels=[16, 32, 64, 64],
                kernel_ch=[3, 3, 3, 3],
                kernel_time=[15, 9, 7, 5],
                stride_ch=[1, 1, 1, 2],
                stride_time=[2, 2, 2, 2],
                use_batchnorm=True,
                conv_dropout=0.1,
                meta_hidden=[32, 32],
                head_hidden=[64, 32],
                head_dropout=0.2,
            ),
            train=TrainHyperparams(
                lr=3e-4, weight_decay=1e-4, batch_size=32,
                epochs=200, patience=30,
                loss="mse", scheduler="cosine", mixed_prec=True,
            ),
        )

    # ── P1: Matched direct-PGV control ───────────────────────────────────────
    elif name == "P1":
        return SpectralExperimentConfig(
            name="P1",
            description="Matched direct-PGV control — same CNN2D encoder as S1, single output",
            seed=seed,
            target_type="pgv",
            arch=ArchConfig(
                encoder_type="cnn2d",
                conv_channels=[16, 32, 64, 64],
                kernel_ch=[3, 3, 3, 3],
                kernel_time=[15, 9, 7, 5],
                stride_ch=[1, 1, 1, 2],
                stride_time=[2, 2, 2, 2],
                use_batchnorm=True,
                conv_dropout=0.1,
                meta_hidden=[32, 32],
                head_hidden=[64, 32],
                head_dropout=0.2,
            ),
            train=TrainHyperparams(
                lr=3e-4, weight_decay=1e-4, batch_size=32,
                epochs=200, patience=30,
                loss="mse", scheduler="cosine", mixed_prec=True,
            ),
        )

    else:
        raise ValueError(f"Unknown config name: '{name}'. "
                         f"Valid: S1, S2, S3, S4, S5, P1")


ALL_CONFIG_NAMES = ["S1", "S2", "S3", "S4", "S5", "P1"]
SPECTRAL_CONFIG_NAMES = ["S1", "S2", "S3", "S4", "S5"]
PGV_CONFIG_NAMES = ["P1"]

# Default output root (platform-aware)
DEFAULT_OUT_ROOT: str = str(_data_root() / "holten_models" / "outputs")
