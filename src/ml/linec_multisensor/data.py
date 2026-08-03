"""data.py
==========
Data loading, preprocessing and torch Dataset for the M0/M1/M2 multi-sensor
Line-C spectral experiment.

Read-only reuse:
    src.db.parquet.parquet_v3_utils.assign_train_type_family
    src.utils.geometry_utils.apply_corrected_distances

Everything else here is new (dataset_spectral.py is single-sensor MP8-only
and is not modified or imported).

Sources (authoritative, read-only)
-----------------------------------
    holten_spectral_targets_v002_corrected/spectral_targets.parquet
        5-sensor spectral targets + train_type/train_speed_kmh/track_number/split
    holten_waveform_v003_ch51_20260626_094551/{waveforms.npy,event_index.parquet}
        validated 51-channel waveform product + train_speed_kmh_is_missing flag
    linec_spectral_oracle_v1/<oracle_run_id>/propagation_curves.csv
        frozen n_track[f] (O0) used by M2's physics decode

Metadata features (n_meta, fit on TRAIN split only)
----------------------------------------------------
    train_family      OHE(drop_first)   -- assign_train_type_family(train_type)
    track_number      OHE(drop_first)   -- {1,2}
    train_speed_kmh   z-scored          -- median-imputed on train
    speed_is_missing  {0,1}             -- from waveform event_index
    log(distance)     z-scored (x5)     -- one per sensor, active-track distance
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset

from src.db.parquet.parquet_v3_utils import assign_train_type_family
from src.utils.geometry_utils import apply_corrected_distances

SENSORS = ["MP4", "MP8", "MP10", "MP1", "MP2"]
N_BANDS = 19
N_CHANNELS = 51
N_SAMPLES = 7500
WF_SCALE = 1.0e6  # strain -> microstrain
R0 = 10.0
ORACLE_RUN_ID = "run_20260802_222103"  # authoritative full O0/O1 oracle run (this session)


def _root() -> Path:
    return Path(r"P:\11210978-erju-ai") if os.name == "nt" else Path("/p/11210978-erju-ai")


def _spec_path() -> Path:
    return _root() / "holten_spectral_targets_v002_corrected" / "spectral_targets.parquet"


def _wf_dir() -> Path:
    return _root() / "holten_waveform" / "holten_waveform_v003_ch51_20260626_094551"


def _oracle_curve_path() -> Path:
    return _root() / "holten_models" / "outputs" / "linec_spectral_oracle_v1" / ORACLE_RUN_ID / "propagation_curves.csv"


# ── DataStats ─────────────────────────────────────────────────────────────────

@dataclass
class DataStats:
    """All preprocessing objects/arrays, fit on TRAIN split only."""

    ohe: OneHotEncoder = field(default=None)
    ohe_feature_names: list = field(default=None)
    speed_imputer: SimpleImputer = field(default=None)
    speed_scaler: StandardScaler = field(default=None)
    dist_scaler: StandardScaler = field(default=None)  # over log(distance), 5 columns
    target_mean: np.ndarray = field(default=None)  # (5, 19)
    target_std: np.ndarray = field(default=None)   # (5, 19)
    n_meta: int = 0
    band_nominal_hz: np.ndarray = field(default=None)  # (19,)
    n_o0: np.ndarray = field(default=None)  # (2, 19) track1=row0, track2=row1 -- FROZEN

    def save(self, path: Path) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "DataStats":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Raw loading ───────────────────────────────────────────────────────────────

def _load_active_track_distances(events_track: np.ndarray) -> np.ndarray:
    """(n_events,) track number -> (n_events, 5) effective distance per SENSOR [m]."""
    combo = pd.DataFrame([(s, t) for s in SENSORS for t in (1, 2)], columns=["sensor_id", "track_number"])
    combo["acc_distance_to_track_m"] = np.nan  # required input column; overwritten by apply_corrected_distances
    combo = apply_corrected_distances(combo, sensor_col="sensor_id", track_col="track_number")
    lookup = {
        (row.sensor_id, row.track_number): row.effective_distance_to_active_track_m
        for row in combo.itertuples()
    }
    R = np.zeros((len(events_track), len(SENSORS)), dtype=np.float64)
    for j, s in enumerate(SENSORS):
        R[:, j] = np.where(events_track == 1, lookup[(s, 1)], lookup[(s, 2)])
    return R


def load_raw_tables():
    """Load and merge event-level tables; returns dict of numpy/pandas objects."""
    spec_cols = [
        "event_id", "sensor_id", "band_index", "band_nominal_hz",
        "fully_inside_valid_range", "velocity_band_level_db",
        "track_number", "train_type", "train_speed_kmh", "split",
    ]
    df = pd.read_parquet(_spec_path(), columns=spec_cols)
    df = df[df["sensor_id"].isin(SENSORS) & df["fully_inside_valid_range"]].copy()

    band_ids = sorted(df["band_index"].unique())
    n_bands = len(band_ids)
    if n_bands != N_BANDS:
        print(f"[WARN] expected {N_BANDS} fully-inside bands, found {n_bands}")
    band_nominal = (
        df[["band_index", "band_nominal_hz"]].drop_duplicates()
        .set_index("band_index").loc[band_ids, "band_nominal_hz"].to_numpy(dtype=np.float64)
    )

    ev_meta = (
        df[["event_id", "track_number", "train_type", "train_speed_kmh", "split"]]
        .drop_duplicates(subset="event_id").set_index("event_id")
    )

    pivot = df.pivot_table(index=["event_id", "sensor_id"], columns="band_index", values="velocity_band_level_db")
    pivot = pivot[band_ids]

    sensor_frames = {}
    common_events = None
    for s in SENSORS:
        sub = pivot.xs(s, level="sensor_id")
        sensor_frames[s] = sub
        ev_set = set(sub.dropna(how="any").index)
        common_events = ev_set if common_events is None else (common_events & ev_set)

    valid_track_events = set(ev_meta.index[ev_meta["track_number"].isin([1, 2])])
    common_events = common_events & valid_track_events

    # ── Waveform event index (join for row index + n_valid + speed-missing flag) ──
    ei = pd.read_parquet(
        _wf_dir() / "event_index.parquet",
        columns=["event_id", "waveform_row_idx", "n_samples_original", "train_speed_kmh_is_missing"],
    )
    ei["event_id"] = ei["event_id"].astype(str)
    ei = ei.set_index("event_id")
    common_events = common_events & set(ei.index)

    events = np.array(sorted(common_events))
    n_events = len(events)
    print(f"Loaded {len(df):,} spectral rows + {len(ei):,} waveform-index rows -> "
          f"{n_events:,} events with complete 5-sensor/19-band/waveform/known-track data "
          f"(dropped {1697 - n_events} of 1697 canonical events)")

    L = np.stack([sensor_frames[s].reindex(events).to_numpy(dtype=np.float64) for s in SENSORS], axis=1)  # (n,5,19)
    TRACK = ev_meta.loc[events, "track_number"].to_numpy(dtype=int)
    SPLIT = ev_meta.loc[events, "split"].to_numpy(dtype=str)
    TRAIN_TYPE = ev_meta.loc[events, "train_type"].to_numpy(dtype=str)
    SPEED = ev_meta.loc[events, "train_speed_kmh"].to_numpy(dtype=np.float64)
    SPEED_MISSING = ei.loc[events, "train_speed_kmh_is_missing"].to_numpy(dtype=np.float64)
    N_SAMPLES_ORIG = ei.loc[events, "n_samples_original"].to_numpy(dtype=np.float64)
    ROW_IDX = ei.loc[events, "waveform_row_idx"].to_numpy(dtype=np.int64)

    N_VALID_250HZ = np.clip(np.round(N_SAMPLES_ORIG * 250.0 / 1000.0).astype(np.int64), 1, N_SAMPLES)
    R = _load_active_track_distances(TRACK)  # (n,5) meters

    if not np.isfinite(L).all():
        raise ValueError("Non-finite values in spectral target matrix L after filtering to fully_inside_valid_range")
    if not np.isfinite(R).all():
        raise ValueError("Non-finite values in active-track distance matrix R")

    for split_name in ("train", "val", "test"):
        n = int(np.sum(SPLIT == split_name))
        print(f"  split={split_name:5s} n_events={n}")

    wf = np.load(str(_wf_dir() / "waveforms.npy"), mmap_mode="r")

    return {
        "events": events, "L": L, "TRACK": TRACK, "SPLIT": SPLIT,
        "TRAIN_TYPE": TRAIN_TYPE, "SPEED": SPEED, "SPEED_MISSING": SPEED_MISSING,
        "N_VALID": N_VALID_250HZ, "ROW_IDX": ROW_IDX, "R": R,
        "band_nominal_hz": band_nominal, "waveforms": wf,
    }


def _load_n_o0() -> np.ndarray:
    """Frozen O0 n_track[f] from the completed train-only oracle, shape (2, 19)."""
    path = _oracle_curve_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Oracle propagation_curves.csv not found at {path}; "
            f"run analyse_linec_spectral_oracle_v1.py (full, non-smoke) first."
        )
    curves = pd.read_csv(path)
    n_o0 = np.stack([curves["n_track1_O0"].to_numpy(dtype=np.float64),
                      curves["n_track2_O0"].to_numpy(dtype=np.float64)], axis=0)
    if not np.isfinite(n_o0).all():
        raise ValueError(f"Non-finite values in frozen O0 n_track curve loaded from {path}")
    return n_o0


# ── Preprocessing (fit on TRAIN only) ─────────────────────────────────────────

def fit_preprocessing(raw: dict) -> DataStats:
    SPLIT = raw["SPLIT"]
    train_mask = SPLIT == "train"

    family = np.array([assign_train_type_family(t) for t in raw["TRAIN_TYPE"]])
    track_str = raw["TRACK"].astype(str)
    cat_df = pd.DataFrame({"train_family": family, "track_number_str": track_str})

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    ohe.fit(cat_df.loc[train_mask])
    ohe_names = list(ohe.get_feature_names_out(["train_family", "track_number_str"]))

    speed_imputer = SimpleImputer(strategy="median")
    speed_imputer.fit(raw["SPEED"][train_mask].reshape(-1, 1))
    speed_imp_all = speed_imputer.transform(raw["SPEED"].reshape(-1, 1))
    speed_scaler = StandardScaler()
    speed_scaler.fit(speed_imp_all[train_mask])

    log_dist = np.log(np.clip(raw["R"], 0.1, None))  # (n, 5)
    dist_scaler = StandardScaler()
    dist_scaler.fit(log_dist[train_mask])

    target_mean = raw["L"][train_mask].mean(axis=0)  # (5,19)
    target_std = raw["L"][train_mask].std(axis=0).clip(min=1e-6)

    n_o0 = _load_n_o0()

    stats = DataStats(
        ohe=ohe, ohe_feature_names=ohe_names,
        speed_imputer=speed_imputer, speed_scaler=speed_scaler, dist_scaler=dist_scaler,
        target_mean=target_mean, target_std=target_std,
        band_nominal_hz=raw["band_nominal_hz"], n_o0=n_o0,
    )
    stats.n_meta = len(ohe_names) + 1 + 1 + len(SENSORS)
    return stats


def build_meta_matrix(raw: dict, stats: DataStats) -> np.ndarray:
    family = np.array([assign_train_type_family(t) for t in raw["TRAIN_TYPE"]])
    track_str = raw["TRACK"].astype(str)
    cat_df = pd.DataFrame({"train_family": family, "track_number_str": track_str})
    ohe_mat = stats.ohe.transform(cat_df)

    speed_imp = stats.speed_imputer.transform(raw["SPEED"].reshape(-1, 1))
    speed_z = stats.speed_scaler.transform(speed_imp)  # (n,1)
    speed_missing = raw["SPEED_MISSING"].reshape(-1, 1)

    log_dist = np.log(np.clip(raw["R"], 0.1, None))
    dist_z = stats.dist_scaler.transform(log_dist)  # (n,5)

    meta = np.hstack([ohe_mat, speed_z, speed_missing, dist_z]).astype(np.float32)
    assert meta.shape[1] == stats.n_meta
    if not np.isfinite(meta).all():
        raise ValueError("Non-finite values in metadata matrix after preprocessing")
    return meta


# ── Dataset ───────────────────────────────────────────────────────────────────

class LinecMultiSensorDataset(Dataset):
    """One sample = one event: waveform block + metadata + raw target dB + geometry."""

    def __init__(self, waveforms: np.ndarray, row_indices: np.ndarray, meta: np.ndarray,
                 n_valid: np.ndarray, target_db: np.ndarray, r_m: np.ndarray,
                 track: np.ndarray, events: np.ndarray) -> None:
        self.waveforms = waveforms
        self.row_indices = row_indices
        self.meta = meta.astype(np.float32)
        self.n_valid = n_valid.astype(np.int32)
        self.target_db = target_db.astype(np.float32)  # (N,5,19)
        self.r_m = r_m.astype(np.float32)  # (N,5)
        self.track = track.astype(np.int64)
        self.events = events
        self.n_meta_features = self.meta.shape[1]

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, i: int):
        block = self.waveforms[self.row_indices[i]]  # (51, 7500) float32 (strain), read-only mmap
        block = np.array(block, dtype=np.float32, copy=True, order="C") * WF_SCALE
        wf_t = torch.from_numpy(block).unsqueeze(0)  # (1, 51, 7500)
        meta_t = torch.from_numpy(self.meta[i])
        nv_t = torch.tensor(int(self.n_valid[i]), dtype=torch.int32)
        tgt_t = torch.from_numpy(self.target_db[i])  # (5,19)
        r_t = torch.from_numpy(self.r_m[i])  # (5,)
        track_t = torch.tensor(int(self.track[i]), dtype=torch.int64)
        ev_idx_t = torch.tensor(i, dtype=torch.int64)
        return wf_t, meta_t, nv_t, tgt_t, r_t, track_t, ev_idx_t


def build_datasets(smoke_n: Optional[int] = None):
    """Returns (train_ds, val_ds, test_ds, stats, events_df)."""
    raw = load_raw_tables()
    stats = fit_preprocessing(raw)
    meta = build_meta_matrix(raw, stats)

    events_df = pd.DataFrame({
        "event_id": raw["events"], "split": raw["SPLIT"], "track_number": raw["TRACK"],
        "train_type": raw["TRAIN_TYPE"], "train_speed_kmh": raw["SPEED"],
    })

    datasets = {}
    for split_name in ("train", "val", "test"):
        sel = np.where(raw["SPLIT"] == split_name)[0]
        if smoke_n is not None:
            sel = sel[: min(smoke_n, len(sel))]
        datasets[split_name] = LinecMultiSensorDataset(
            waveforms=raw["waveforms"], row_indices=raw["ROW_IDX"][sel], meta=meta[sel],
            n_valid=raw["N_VALID"][sel], target_db=raw["L"][sel], r_m=raw["R"][sel],
            track=raw["TRACK"][sel], events=raw["events"][sel],
        )
    return datasets["train"], datasets["val"], datasets["test"], stats, events_df


def make_loader(ds: LinecMultiSensorDataset, batch_size: int, shuffle: bool, num_workers: int = 0):
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                       pin_memory=torch.cuda.is_available(), persistent_workers=(num_workers > 0), drop_last=False)
