"""dataset_spectral.py
======================
PyTorch Dataset and preprocessing utilities for the MP8 spectral prediction
experiment.

Each sample is ONE EVENT (event-level, not sensor-level).
  waveform : float32 (51, 7500) — FO block, scaled to microstrain (×1e6)
  meta     : float32 (11,)      — preprocessed metadata features
  n_valid  : int32              — number of valid (non-padded) time samples
  target   : float32 (19,) or (1,) — standardized spectral dB or log-PGV
  ev_idx   : int                — index into the event table (for lookup)

Metadata features (n_meta = 11)
  train_family (OHE-7, reference=DDZ)
  track_number (OHE-1, reference=track 1)
  train_speed_kmh (z-scored)
  log_mp8_distance (log(d) where d=4 m track 1, d=8 m track 2)
  valid_fraction = n_valid_250hz / 7500

Preprocessing
  All scalers and OHE categories fitted on training events only.
  DataStats object is saved alongside the model for reproducibility.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.db.parquet.parquet_v3_utils import assign_train_type_family
from src.ml.spectral.config_spectral_v001 import (
    MP8_DIST_TRACK1, MP8_DIST_TRACK2, N_BANDS, N_CHANNELS, N_SAMPLES,
    PRIMARY_NOMINALS, V_REF, WF_SCALE, _data_root,
    ALIGN_DIR_NAME, SPEC_DIR_NAME, WF_BUILD_NAME,
)


# ── Paths ─────────────────────────────────────────────────────────────────────

def _paths():
    r = _data_root()
    return {
        "align": r / ALIGN_DIR_NAME,
        "spec":  r / SPEC_DIR_NAME,
        "wf_dir": r / "holten_waveform" / WF_BUILD_NAME,
        "bands": Path(__file__).resolve().parents[3] / "spectral_definitions" / "bands.parquet",
    }


# ── DataStats ─────────────────────────────────────────────────────────────────

@dataclass
class DataStats:
    """Preprocessing statistics fitted on training data only."""
    # Metadata preprocessing objects
    ohe: OneHotEncoder = field(default=None)
    speed_scaler: StandardScaler = field(default=None)
    speed_imputer: SimpleImputer = field(default=None)
    # Target normalization per band
    target_mean: np.ndarray = field(default=None)   # (19,) or (1,) for pgv
    target_std:  np.ndarray = field(default=None)
    # Metadata residual baseline (optional, for S3)
    meta_baseline_pred: Optional[np.ndarray] = field(default=None)  # (N_events, 19)
    # PGV stats for auxiliary task (optional)
    pgv_mean: Optional[float] = field(default=None)
    pgv_std:  Optional[float] = field(default=None)
    # List of event_ids in the combined event table (preserves order)
    event_ids: Optional[np.ndarray] = field(default=None)
    # Feature names
    meta_feature_names: Optional[list] = field(default=None)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "DataStats":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Raw data loading ─────────────────────────────────────────────────────────

def load_raw_data(use_meta_residual: bool = False,
                  meta_baseline_dir: Optional[Path] = None,
                  target_type: str = "spectral",
                  ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load aligned events, spectral targets, and waveform array.

    Returns
    -------
    events_df : DataFrame with one row per event
                Columns: event_id, train_family, train_speed_kmh, track_number,
                         split, n_valid_250hz, valid_fraction,
                         log_mp8_distance,
                         + target columns (band_*/log_pgv_z)
    waveforms  : float32 (N_events, 51, 7500) memory-mapped array
    wf_row_map : int array (N_events,) waveform_row_idx → position in waveforms
    """
    p = _paths()

    # ── Authoritative event metadata ──────────────────────────────────────────
    aligned = pd.read_parquet(
        p["align"] / "aligned_events.parquet",
        columns=["event_id", "train_type", "train_speed_kmh",
                 "track_number", "split"],
    )
    aligned["event_id"] = aligned["event_id"].astype(str)
    aligned["track_number"] = aligned["track_number"].fillna(0).astype(int)
    aligned["train_family"] = aligned["train_type"].apply(assign_train_type_family)

    # ── Waveform event index (valid duration) ─────────────────────────────────
    ei = pd.read_parquet(
        p["wf_dir"] / "event_index.parquet",
        columns=["event_id", "waveform_row_idx", "n_samples_original"],
    )
    ei["event_id"] = ei["event_id"].astype(str)
    ei["n_valid_250hz"] = (
        ei["n_samples_original"] * 250.0 / 1000.0
    ).round().astype(int).clip(1, N_SAMPLES)
    ei["valid_fraction"] = ei["n_valid_250hz"].astype(float) / N_SAMPLES

    # ── Merge events ──────────────────────────────────────────────────────────
    df = aligned.merge(ei[["event_id", "waveform_row_idx",
                             "n_valid_250hz", "valid_fraction"]],
                        on="event_id", how="inner")

    # ── MP8 log-distance from active track ────────────────────────────────────
    df["mp8_dist_m"] = df["track_number"].map({1: MP8_DIST_TRACK1,
                                                 2: MP8_DIST_TRACK2}).fillna(MP8_DIST_TRACK1)
    df["log_mp8_distance"] = np.log(df["mp8_dist_m"].clip(lower=0.1))

    # ── Spectral targets ──────────────────────────────────────────────────────
    spec = pd.read_parquet(
        p["spec"] / "spectral_targets.parquet",
        columns=["event_id", "sensor_id", "band_nominal_hz",
                 "velocity_band_level_db", "fully_inside_valid_range",
                 "raw_pgv_z_mms"],
    )
    spec["event_id"] = spec["event_id"].astype(str)
    mp8 = spec[
        (spec["sensor_id"] == "MP8") & spec["fully_inside_valid_range"]
    ].copy()
    mp8 = mp8[mp8["band_nominal_hz"].isin(PRIMARY_NOMINALS)]

    # Pivot spectral targets to wide (one row per event)
    band_col = {hz: f"b{hz:.4g}hz" for hz in PRIMARY_NOMINALS}
    mp8["col"] = mp8["band_nominal_hz"].map(band_col)
    wide = mp8.pivot(index="event_id", columns="col",
                     values="velocity_band_level_db").reset_index()
    band_cols = [band_col[hz] for hz in PRIMARY_NOMINALS]

    # Extract PGV (same for all bands per event)
    pgv_df = mp8.groupby("event_id")["raw_pgv_z_mms"].first().reset_index()
    pgv_df.columns = ["event_id", "raw_pgv_z_mms"]

    # Merge everything
    df = df.merge(wide, on="event_id", how="inner")
    df = df.merge(pgv_df, on="event_id", how="inner")
    df = df.reset_index(drop=True)

    # ── Optionally compute metadata-residual target ───────────────────────────
    if use_meta_residual:
        _add_meta_residual(df, band_cols, meta_baseline_dir, p)

    # ── Waveform array (memory-mapped) ────────────────────────────────────────
    wf = np.load(str(p["wf_dir"] / "waveforms.npy"), mmap_mode="r")
    # wf shape: (N_total_wf_events, 51, 7500) float32

    return df, band_cols, wf


def _add_meta_residual(df: pd.DataFrame,
                       band_cols: list,
                       meta_baseline_dir: Optional[Path],
                       p: dict) -> None:
    """Compute and add metadata-baseline-residual columns to df (in-place).

    Adds columns: resid_{col} for each band col.
    """
    # Auto-discover latest meta_spectral_baseline_v001_* if not specified
    if meta_baseline_dir is None:
        cands = sorted(p["align"].parent.glob(
            "meta_spectral_baseline_v001_*"))
        if not cands:
            raise FileNotFoundError(
                "No meta_spectral_baseline_v001_* found. "
                "Run train_meta_spectral_v001.py first."
            )
        meta_baseline_dir = cands[-1]

    model_path = meta_baseline_dir / "model.pkl"
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    meta_model  = bundle["model"]
    meta_prep   = bundle["preprocessor"]
    meta_cats   = bundle["cat_cols"]
    meta_nums   = bundle["num_cols"]
    meta_tcols  = bundle["target_cols"]   # ordered band columns from baseline

    # Build feature matrix for all events
    feat = df[meta_cats + meta_nums].copy()
    for c in meta_cats:
        feat[c] = feat[c].astype(str).fillna("Unknown")
    X_all = meta_prep.transform(feat).astype(np.float64)
    meta_pred = meta_model.predict(X_all)   # (N_events, 19) dB predictions

    # meta_tcols may use different column name format; align to band_cols
    # meta_tcols format: "b{hz:.4g}hz" matches band_cols (same format here)
    # But the baseline used "band_{hz:.4g}hz" format — need to re-map
    # Check which format is used
    for i, hz in enumerate(PRIMARY_NOMINALS):
        resid_col = f"resid_b{hz:.4g}hz"
        # find matching baseline column
        match_idx = None
        for j, tc in enumerate(meta_tcols):
            tc_hz_str = tc.replace("band_", "").replace("hz", "").lstrip("b")
            if abs(float(tc_hz_str) - hz) < 0.01:
                match_idx = j
                break
        if match_idx is not None:
            df[resid_col] = df[band_cols[i]].values - meta_pred[:, match_idx]
        else:
            # fallback: assume same order
            df[resid_col] = df[band_cols[i]].values - meta_pred[:, i]

    df["_meta_baseline_dir"] = str(meta_baseline_dir)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def fit_preprocessing(
    df: pd.DataFrame,
    train_mask: pd.Series,
    band_cols: list,
    target_type: str = "spectral",
    use_meta_residual: bool = False,
) -> DataStats:
    """Fit all preprocessing statistics on training events only.

    Parameters
    ----------
    df         : full events DataFrame (all splits)
    train_mask : boolean mask selecting training events
    band_cols  : list of 19 band column names
    """
    stats = DataStats()
    tr = df[train_mask].copy()

    # ── OHE for train_family + track_number ──────────────────────────────────
    cat_cols = ["train_family", "track_number_str"]
    df["track_number_str"] = df["track_number"].astype(str)
    tr["track_number_str"] = tr["track_number"].astype(str)

    stats.ohe = OneHotEncoder(
        drop="first", sparse_output=False, handle_unknown="ignore"
    )
    stats.ohe.fit(tr[["train_family", "track_number_str"]])

    ohe_names = list(stats.ohe.get_feature_names_out(
        ["train_family", "track_number_str"]))

    # ── Speed scaler ─────────────────────────────────────────────────────────
    stats.speed_imputer = SimpleImputer(strategy="median")
    stats.speed_scaler  = StandardScaler()
    speed_tr = stats.speed_imputer.fit_transform(
        tr[["train_speed_kmh"]].values)
    stats.speed_scaler.fit(speed_tr)

    stats.meta_feature_names = ohe_names + [
        "train_speed_kmh_scaled",
        "log_mp8_distance",
        "valid_fraction",
    ]

    # ── Target stats ─────────────────────────────────────────────────────────
    if target_type == "spectral":
        if use_meta_residual:
            resid_cols = [f"resid_{c}" for c in band_cols]
            y_tr = tr[resid_cols].values.astype(np.float64)
        else:
            y_tr = tr[band_cols].values.astype(np.float64)
        stats.target_mean = y_tr.mean(axis=0).astype(np.float32)
        stats.target_std  = y_tr.std(axis=0).clip(min=1e-6).astype(np.float32)
    else:  # pgv
        pgv_tr = tr["raw_pgv_z_mms"].clip(lower=1e-6).values
        log_pgv_tr = np.log(pgv_tr).astype(np.float64)
        stats.target_mean = np.array([log_pgv_tr.mean()], dtype=np.float32)
        stats.target_std  = np.array([max(log_pgv_tr.std(), 1e-6)], dtype=np.float32)

    # ── PGV stats for auxiliary task ──────────────────────────────────────────
    pgv_tr_vals = tr["raw_pgv_z_mms"].clip(lower=1e-6).values
    log_pgv_tr = np.log(pgv_tr_vals)
    stats.pgv_mean = float(log_pgv_tr.mean())
    stats.pgv_std  = float(max(log_pgv_tr.std(), 1e-6))

    stats.event_ids = df["event_id"].values

    return stats


def apply_preprocessing(
    df: pd.DataFrame,
    stats: DataStats,
    band_cols: list,
    target_type: str = "spectral",
    use_meta_residual: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform features and targets for a subset of events.

    Returns
    -------
    meta_arr : float32 (N, 11) metadata features
    y_arr    : float32 (N, 19) standardized targets or (N, 1) for PGV
    n_valid  : int32 (N,) valid sample counts
    """
    n = len(df)
    df2 = df.copy()
    df2["track_number_str"] = df2["track_number"].astype(str)

    # ── Metadata features ─────────────────────────────────────────────────────
    cat_mat = stats.ohe.transform(
        df2[["train_family", "track_number_str"]].fillna("Unknown")
    ).astype(np.float32)

    speed_imp = stats.speed_imputer.transform(df2[["train_speed_kmh"]].values)
    speed_sc  = stats.speed_scaler.transform(speed_imp).astype(np.float32)

    log_dist   = df2["log_mp8_distance"].fillna(np.log(MP8_DIST_TRACK1)).values.reshape(-1, 1).astype(np.float32)
    valid_frac = df2["valid_fraction"].fillna(1.0).values.reshape(-1, 1).astype(np.float32)

    meta_arr = np.hstack([cat_mat, speed_sc, log_dist, valid_frac])  # (N, 11)

    # ── Targets ───────────────────────────────────────────────────────────────
    if target_type == "spectral":
        if use_meta_residual:
            src_cols = [f"resid_{c}" for c in band_cols]
        else:
            src_cols = band_cols
        y_raw = df2[src_cols].values.astype(np.float32)  # (N, 19)
        y_arr = (y_raw - stats.target_mean) / stats.target_std
    else:  # pgv
        pgv_vals = df2["raw_pgv_z_mms"].clip(lower=1e-6).values
        log_pgv  = np.log(pgv_vals).astype(np.float32).reshape(-1, 1)
        y_arr = (log_pgv - stats.target_mean) / stats.target_std

    # ── PGV auxiliary target (always available) ───────────────────────────────
    pgv_vals = df2["raw_pgv_z_mms"].clip(lower=1e-6).values
    log_pgv_aux = ((np.log(pgv_vals) - stats.pgv_mean) / stats.pgv_std).astype(np.float32)

    n_valid = df2["n_valid_250hz"].values.astype(np.int32)

    return meta_arr, y_arr.astype(np.float32), log_pgv_aux, n_valid


# ── Dataset ───────────────────────────────────────────────────────────────────

class SpectralEventDataset(Dataset):
    """One sample = one event. Waveform is shared (mmap'd) and never copied."""

    def __init__(
        self,
        waveforms: np.ndarray,        # (N_total, 51, 7500) mmap float32
        row_indices: np.ndarray,      # (N_subset,) waveform_row_idx
        meta: np.ndarray,             # (N_subset, 11) float32
        targets: np.ndarray,          # (N_subset, 19) or (N_subset, 1) float32
        pgv_aux: np.ndarray,          # (N_subset,) float32 standardized log-PGV
        n_valid: np.ndarray,          # (N_subset,) int32
        ev_indices: np.ndarray,       # (N_subset,) int — index into events_df
    ) -> None:
        self.waveforms   = waveforms
        self.row_indices = row_indices.astype(np.int64)
        self.meta        = meta.astype(np.float32)
        self.targets     = targets.astype(np.float32)
        self.pgv_aux     = pgv_aux.astype(np.float32)
        self.n_valid     = n_valid.astype(np.int32)
        self.ev_indices  = ev_indices.astype(np.int64)

    @property
    def n_meta_features(self) -> int:
        return self.meta.shape[1]

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, i: int):
        # (1, 51, 7500) — add channel dim for Conv2D
        block = self.waveforms[self.row_indices[i]]    # (51, 7500) float32
        wf_t  = torch.from_numpy(np.ascontiguousarray(block)).unsqueeze(0)
        meta_t   = torch.from_numpy(self.meta[i])
        tgt_t    = torch.from_numpy(self.targets[i])
        pgv_t    = torch.tensor(self.pgv_aux[i], dtype=torch.float32)
        nv_t     = torch.tensor(int(self.n_valid[i]), dtype=torch.int32)
        ev_idx_t = torch.tensor(int(self.ev_indices[i]), dtype=torch.int64)
        return wf_t, meta_t, nv_t, tgt_t, pgv_t, ev_idx_t


# ── Build function ────────────────────────────────────────────────────────────

def build_datasets(
    cfg,  # SpectralExperimentConfig
    smoke_n: Optional[int] = None,
) -> Tuple["SpectralEventDataset", "SpectralEventDataset",
           "SpectralEventDataset", DataStats, pd.DataFrame, list]:
    """Load data, fit preprocessing on train, return all three splits.

    Parameters
    ----------
    cfg     : SpectralExperimentConfig
    smoke_n : if given, only use this many training events (overfit smoke test)

    Returns
    -------
    train_ds, val_ds, test_ds, stats, events_df, band_cols
    """
    events_df, band_cols, wf = load_raw_data(
        use_meta_residual=cfg.use_meta_residual,
        target_type=cfg.target_type,
    )

    # ── Masks ─────────────────────────────────────────────────────────────────
    tr_mask  = events_df["split"] == "train"
    va_mask  = events_df["split"] == "val"
    te_mask  = events_df["split"] == "test"

    # Smoke-test override: use only first smoke_n training events
    if smoke_n is not None:
        tr_idx = events_df.index[tr_mask].tolist()[:smoke_n]
        tr_mask = events_df.index.isin(tr_idx)
        va_mask = tr_mask.copy()   # same events for val in smoke mode
        te_mask = tr_mask.copy()

    # ── Fit preprocessing on training events ──────────────────────────────────
    stats = fit_preprocessing(
        events_df, tr_mask, band_cols,
        target_type=cfg.target_type,
        use_meta_residual=cfg.use_meta_residual,
    )

    def _make_ds(mask: pd.Series) -> "SpectralEventDataset":
        sub = events_df[mask].reset_index(drop=False)   # keep original index
        orig_idx = sub["index"].values                   # original row in events_df
        meta, tgt, pgv_aux, nv = apply_preprocessing(
            sub, stats, band_cols,
            target_type=cfg.target_type,
            use_meta_residual=cfg.use_meta_residual,
        )
        row_idx = sub["waveform_row_idx"].values.astype(np.int64)
        return SpectralEventDataset(
            waveforms=wf,
            row_indices=row_idx,
            meta=meta,
            targets=tgt,
            pgv_aux=pgv_aux,
            n_valid=nv,
            ev_indices=orig_idx,
        )

    train_ds = _make_ds(tr_mask)
    val_ds   = _make_ds(va_mask)
    test_ds  = _make_ds(te_mask)

    return train_ds, val_ds, test_ds, stats, events_df, band_cols


def make_loader(ds: SpectralEventDataset, batch_size: int,
                shuffle: bool, num_workers: int = 4) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0),
        drop_last=False,
    )
