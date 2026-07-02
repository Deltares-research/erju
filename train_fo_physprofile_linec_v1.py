"""
train_fo_physprofile_linec_v1.py  —  FO-PhysProfile v1

Source-path-receiver attenuation-profile model for Holten Line-C side -1.
Explicit physics decomposition:

    pred_log_ij  =  c_hat_i
                    - (n_track + delta_n_hat_i) * log(r_ij / r0)
                    + pca_shape_i[sensor_j]
                    + local_residual_hat_ij

Stage 1  Event-level models (XGBoost):
    C     predict c_target_event
    N     predict delta_n_target
    PC1   predict pc1_target
    PC2   predict pc2_target

Stage 2  Row-level local residual (XGBoost):
    target = target_log - pred_log_profile

Current baselines (test):
    P3_corrected_n:  RMSE(PGV)=2.4006  RMSE(log)=0.5993
    PXGBR-R2:        RMSE(PGV)=2.2718  RMSE(log)=0.5946
    PXGBR_ens_top5:  RMSE(PGV)=2.2473  RMSE(log)=0.6223

Target: RMSE(PGV) < 2.20, RMSE(log) <= 0.61, MP4 RMSE < 4.50
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.signal import welch
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

from src.utils.geometry_utils import apply_corrected_distances


# ─── Constants ────────────────────────────────────────────────────────────────

def _root() -> Path:
    return Path(r"P:\11210978-erju-ai") if os.name == "nt" else Path("/p/11210978-erju-ai")

MODELS_ROOT  = _root() / "holten_models"
PARQUET_ROOT = _root() / "holten_parquet"
WAVE_ROOT    = _root() / "holten_waveform"

SENSOR_ORDER = ["MP4", "MP8", "MP10", "MP1", "MP2"]
SENSOR_CODE  = {s: i for i, s in enumerate(SENSOR_ORDER)}

N_TRACK      = {1: 1.0655, 2: 1.3246}   # corrected-intercept method
R0           = 10.0
N_SHRINK_W   = 0.3                       # weight on raw n_event
N_CLIP       = (0.5, 2.0)
PCA_N_COMP   = 3                         # fit up to 3, use 2 by default

# Waveform geometry (ch51 slice → 21ch Line-C window)
# WF_FS is a default fallback; actual value is read from build_config.json at load time
WF_FS_DEFAULT = 250.0                    # Hz fallback if metadata absent
WF_FS         = WF_FS_DEFAULT            # overwritten by load_waveforms()
WF_N_CH      = 21                        # channels 1184-1204
WF_CH_LO     = 1184
WF_CH_HI     = 1204
WF_CTR_IDX   = 10                        # channel 1194, index in 0..20
WF_SLICE     = slice(19, 40)             # ch51 → ch21
WF_BANDPASS  = (1.0, 100.0)             # expected bandpass; applied if not already done
WF_ALREADY_BANDPASSED = True            # v003 build applies 1-100 Hz bandpass

# No-leakage guard
LEAKAGE = frozenset({
    "target_pgv_z_mms", "target_log", "target_pgv",
    "c_target", "c_target_event",
    "residual_log", "scaled_residual",
    "max_pgv", "mp4_pgv",
    "event_id", "split",
    "sensor", "sensor_id", "site_id",
    "train_type", "acc_side_of_track",
})


# ─── Path discovery ───────────────────────────────────────────────────────────

def find_parquet_v2() -> Path:
    hits = sorted(PARQUET_ROOT.glob("parquet_v002_*"))
    if not hits:
        raise FileNotFoundError("No parquet_v002_* build")
    return hits[-1] / "dataset.parquet"


def find_waveform_v3() -> Path:
    hits = sorted(WAVE_ROOT.glob("holten_waveform_v003_ch51_*"))
    if not hits:
        hits = sorted(WAVE_ROOT.glob("holten_waveform_v003_*"))
    if not hits:
        raise FileNotFoundError("No holten_waveform_v003* build in " + str(WAVE_ROOT))
    return hits[-1]


def find_p3_dir() -> Optional[Path]:
    hits = sorted(MODELS_ROOT.glob("cnn_curveprior_p3_savepred_linec_v001_vP3_*"))
    return hits[-1] if hits else None


def find_pxgbr_dir() -> Optional[Path]:
    hits = sorted(MODELS_ROOT.glob("pxgbr_linec_v001_*"))
    return hits[-1] if hits else None


def find_pxgbr_ens_dir() -> Optional[Path]:
    hits = sorted(MODELS_ROOT.glob("pxgbr_ensemble_linec_v001_*"))
    return hits[-1] if hits else None


def make_output_dir(base: Path, version: str = "fo_physprofile_linec_v001") -> Path:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out  = base / "outputs" / "fo_physprofile_linec_v1" / f"{version}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ─── TASK 1 — Load data ───────────────────────────────────────────────────────

def load_linec_parquet() -> pd.DataFrame:
    pq = find_parquet_v2()
    print(f"[data] Parquet v2: {pq}")
    df = pd.read_parquet(pq)
    if "effective_distance_to_active_track_m" not in df.columns:
        df = apply_corrected_distances(df)
    df = df[df["sensor_id"].isin(SENSOR_ORDER)].copy()
    df = df.dropna(subset=["target_pgv_z_mms", "track_number"])
    df = df[(df["target_pgv_z_mms"] > 0) & df["track_number"].isin([1, 2])]
    df["event_id"] = df["event_id"].astype(str)
    df["sensor"]   = df["sensor_id"]
    df["target_log"]  = np.log(df["target_pgv_z_mms"].clip(lower=1e-6))
    df["target_pgv"]  = df["target_pgv_z_mms"]
    df["log_distance"] = np.log(df["effective_distance_to_active_track_m"] / R0)
    df["distance"]     = df["effective_distance_to_active_track_m"]
    df["sensor_code"]  = df["sensor"].map(SENSOR_CODE).astype(float)
    print(f"[data] Line-C rows: {len(df):,}  events: {df['event_id'].nunique():,}")
    return df


def load_waveforms() -> Tuple[np.ndarray, Dict[str, int], float]:
    """
    Load waveform array and event index.  Also reads actual sampling frequency
    from build_config.json and verifies shape / implied duration.
    Returns (waveforms, event_map, actual_fs).
    """
    global WF_FS

    wave_dir = find_waveform_v3()
    print(f"[data] Waveform dir: {wave_dir}")

    # Read actual fs from build config
    cfg_path = wave_dir / "build_config.json"
    actual_fs = WF_FS_DEFAULT
    bandpassed = WF_ALREADY_BANDPASSED
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            # navigate nested: signal.target_fs_hz or signal/target_fs_hz
            sig = cfg.get("signal", cfg)
            actual_fs = float(sig.get("target_fs_hz",
                              sig.get("target_fs",
                              cfg.get("target_fs_hz", WF_FS_DEFAULT))))
            bp_min = float(sig.get("bandpass_freqmin", -1))
            bp_max = float(sig.get("bandpass_freqmax", -1))
            if bp_min > 0 and bp_max > 0:
                bandpassed = True
                print(f"[data] Waveform bandpass confirmed: {bp_min}-{bp_max} Hz")
        except Exception as exc:
            print(f"[WARN] Could not read build_config.json: {exc}")
    else:
        print(f"[WARN] build_config.json not found, using default fs={WF_FS_DEFAULT} Hz")

    wf = np.load(wave_dir / "waveforms.npy").astype(np.float32)

    # slice ch51 → ch21 if needed
    if wf.shape[1] == 51:
        wf = wf[:, WF_SLICE, :]
        print(f"[data] Sliced wf ch51 → ch21  shape={wf.shape}")
    elif wf.shape[1] != WF_N_CH:
        raise ValueError(f"Unexpected waveform shape: {wf.shape}")

    T = wf.shape[2]
    implied_dur = T / actual_fs
    print(f"[data] Waveform shape={wf.shape}  fs={actual_fs} Hz  "
          f"T={T} samples  implied_dur={implied_dur:.1f} s")
    if implied_dur < 1.0 or implied_dur > 120.0:
        print(f"[WARN] implied duration {implied_dur:.1f}s is outside expected range 1-120 s. "
              f"Verify fs: if T={T} and expected 7.5 s use fs=1000, if 30 s use fs=250.")

    # Apply bandpass if not already done
    if not bandpassed:
        from scipy.signal import iirfilter, zpk2sos, sosfiltfilt
        print(f"[data] Applying bandpass {WF_BANDPASS[0]}-{WF_BANDPASS[1]} Hz to waveforms...")
        fe = 0.5 * actual_fs
        z, p, k = iirfilter(5, [WF_BANDPASS[0]/fe, WF_BANDPASS[1]/fe],
                             btype="band", ftype="butter", output="zpk")
        sos = zpk2sos(z, p, k)
        # Process batch in float64 for numerical stability
        wf_f = wf.astype(np.float64)
        for i in range(wf_f.shape[0]):
            wf_f[i] = sosfiltfilt(sos, wf_f[i], axis=1)
        wf = wf_f.astype(np.float32)
        print("[data] Bandpass applied.")

    # Update global WF_FS
    WF_FS = actual_fs

    idx_df = pd.read_parquet(wave_dir / "event_index.parquet")
    ok = idx_df[idx_df["build_status"] == "ok"].copy()
    ok["event_id"] = ok["event_id"].astype(str)
    event_map = dict(zip(ok["event_id"], ok["waveform_row_idx"].astype(int)))
    print(f"[data] Waveform events ok: {len(event_map):,}")
    return wf, event_map, actual_fs


def get_split_labels(df: pd.DataFrame, p3_dir: Optional[Path]) -> pd.DataFrame:
    """
    Return event-level split assignments from P3 predictions (train/val/test).
    If P3 not available, reproduce the same 65/15/20 stratified split with seed 42.
    """
    if p3_dir is not None:
        p = p3_dir / "all_predictions.parquet"
        if p.exists():
            p3 = pd.read_parquet(p)[["event_id", "split"]].drop_duplicates("event_id")
            p3["event_id"] = p3["event_id"].astype(str)
            print(f"[split] Loaded from P3: {p3['split'].value_counts().to_dict()}")
            return p3

    # Fallback: re-derive from event list
    from sklearn.model_selection import train_test_split
    events = sorted(df["event_id"].unique())
    n_total = len(events)
    tr_events, tmp = train_test_split(events, test_size=0.35, random_state=42)
    va_events, te_events = train_test_split(tmp, test_size=4/7, random_state=42)
    rows = (
        [(e, "train") for e in tr_events]
        + [(e, "val")   for e in va_events]
        + [(e, "test")  for e in te_events]
    )
    split_df = pd.DataFrame(rows, columns=["event_id", "split"])
    print(f"[split] Derived: {split_df['split'].value_counts().to_dict()}")
    return split_df


# ─── TASK 2 — Waveform feature extraction ─────────────────────────────────────

def _band_energy(pxx: np.ndarray, freqs: np.ndarray, f_lo: float, f_hi: float) -> float:
    df = freqs[1] - freqs[0]
    mask = (freqs >= f_lo) & (freqs < f_hi)
    return float(np.sum(pxx[mask]) * df) if np.any(mask) else 0.0


def extract_waveform_features_event(wf_block: np.ndarray,
                                      fs: float = WF_FS_DEFAULT) -> Dict[str, float]:
    """
    Extract physically meaningful features from one event's waveform block.
    Input: wf_block shape (21, T), bandpass filtered, at given fs.
    """
    T   = wf_block.shape[1]
    ctr = wf_block[WF_CTR_IDX].astype(np.float64)    # centre channel (1194)
    all_flat = wf_block.flatten().astype(np.float64)

    feats: Dict[str, float] = {}

    # ── Time-domain global ────────────────────────────────────────────────────
    feats["wf_global_rms"]          = float(np.sqrt(np.mean(all_flat ** 2)))
    feats["wf_global_peak_abs"]     = float(np.max(np.abs(all_flat)))
    feats["wf_global_energy"]       = float(np.sum(all_flat ** 2))
    feats["wf_global_p95_abs"]      = float(np.percentile(np.abs(all_flat), 95))
    rms_g = feats["wf_global_rms"]
    feats["wf_crest_factor"]        = feats["wf_global_peak_abs"] / (rms_g + 1e-12)

    peak_ctr = np.max(np.abs(ctr))
    feats["wf_duration_above_25pct"] = float(np.sum(np.abs(ctr) > 0.25 * peak_ctr) / fs)
    feats["wf_duration_above_50pct"] = float(np.sum(np.abs(ctr) > 0.50 * peak_ctr) / fs)

    # ── Frequency-domain (centre channel Welch) ───────────────────────────────
    nperseg = min(256, T // 4)
    nfft    = max(nperseg, 512)
    freqs, pxx = welch(ctr, fs=fs, nperseg=nperseg, nfft=nfft,
                       window="hamming", scaling="density", detrend="linear")

    feats["wf_band_1_5"]    = _band_energy(pxx, freqs, 1,   5)
    feats["wf_band_5_10"]   = _band_energy(pxx, freqs, 5,  10)
    feats["wf_band_10_20"]  = _band_energy(pxx, freqs, 10, 20)
    feats["wf_band_20_40"]  = _band_energy(pxx, freqs, 20, 40)
    feats["wf_band_40_80"]  = _band_energy(pxx, freqs, 40, 80)
    feats["wf_band_80_125"] = _band_energy(pxx, freqs, 80, 125)

    total_pxx = np.sum(pxx) + 1e-30
    feats["wf_spectral_centroid"]   = float(np.sum(freqs * pxx) / total_pxx)
    feats["wf_spectral_bandwidth"]  = float(
        np.sqrt(np.sum((freqs - feats["wf_spectral_centroid"]) ** 2 * pxx) / total_pxx)
    )
    feats["wf_dominant_frequency"]  = float(freqs[np.argmax(pxx)])

    low_e  = feats["wf_band_1_5"] + feats["wf_band_5_10"]
    high_e = feats["wf_band_20_40"] + feats["wf_band_40_80"]
    feats["wf_high_low_energy_ratio"] = high_e / (low_e + 1e-30)

    # ── Spatial / channel features ────────────────────────────────────────────
    ch_energy = np.sum(wf_block.astype(np.float64) ** 2, axis=1)   # (21,)
    ch_rms    = np.sqrt(ch_energy / T)

    feats["wf_ch_energy_mean"]  = float(np.mean(ch_energy))
    feats["wf_ch_energy_std"]   = float(np.std(ch_energy))
    feats["wf_ch_energy_max"]   = float(np.max(ch_energy))
    feats["wf_ch_energy_min"]   = float(np.min(ch_energy))
    feats["wf_ch_energy_p95"]   = float(np.percentile(ch_energy, 95))
    feats["wf_ch_of_max_energy"] = float(WF_CH_LO + int(np.argmax(ch_energy)))

    positions = np.arange(WF_CH_LO, WF_CH_HI + 1, dtype=float)  # (21,)
    total_e  = ch_energy.sum() + 1e-30
    centroid = float(np.sum(positions * ch_energy) / total_e)
    feats["wf_spatial_centroid"] = centroid
    feats["wf_spatial_spread"]   = float(
        np.sqrt(np.sum((positions - centroid) ** 2 * ch_energy) / total_e)
    )

    left_e  = ch_energy[:WF_CTR_IDX].sum()
    right_e = ch_energy[WF_CTR_IDX + 1:].sum()
    feats["wf_left_right_asymmetry"] = (left_e - right_e) / (left_e + right_e + 1e-30)

    # Adjacent channel correlations
    adj_corr = []
    for ch in range(WF_N_CH - 1):
        a, b = wf_block[ch].astype(np.float64), wf_block[ch + 1].astype(np.float64)
        denom = np.std(a) * np.std(b)
        if denom < 1e-30:
            adj_corr.append(0.0)
        else:
            adj_corr.append(float(np.mean((a - a.mean()) * (b - b.mean())) / denom))

    feats["wf_adj_corr_mean"] = float(np.mean(adj_corr))
    feats["wf_adj_corr_min"]  = float(np.min(adj_corr))
    feats["wf_spatial_corr_length"] = float(np.sum(np.array(adj_corr) > 0.5))

    # Energy gradient near centre (channels within ±4 of centre)
    near_lo = max(0, WF_CTR_IDX - 4)
    near_hi = min(WF_N_CH, WF_CTR_IDX + 5)
    near_pos = np.arange(near_lo, near_hi, dtype=float)
    near_e   = ch_energy[near_lo:near_hi]
    if len(near_pos) > 1:
        feats["wf_energy_grad_near_centre"] = float(np.polyfit(near_pos, near_e, 1)[0])
    else:
        feats["wf_energy_grad_near_centre"] = 0.0

    # ── Pre / post / during features ──────────────────────────────────────────
    n_edge = max(1, T // 10)           # 10% at each end ≈ 3 s

    pre_ctr   = ctr[:n_edge]
    post_ctr  = ctr[-n_edge:]
    dur_ctr   = ctr[n_edge:-n_edge]

    feats["wf_pre_event_mean"]   = float(np.mean(pre_ctr))
    feats["wf_post_event_mean"]  = float(np.mean(post_ctr))
    feats["wf_post_minus_pre"]   = feats["wf_post_event_mean"] - feats["wf_pre_event_mean"]
    feats["wf_pre_event_rms"]    = float(np.sqrt(np.mean(pre_ctr ** 2)))
    feats["wf_post_event_rms"]   = float(np.sqrt(np.mean(post_ctr ** 2)))

    dur_e_per_s = np.mean(dur_ctr ** 2)
    pre_e_per_s = np.mean(pre_ctr ** 2) if len(pre_ctr) > 0 else 0.0
    feats["wf_during_minus_pre_energy"] = float(dur_e_per_s - pre_e_per_s)

    feats["wf_signed_cumsum"]   = float(np.sum(ctr))
    feats["wf_abs_cumsum"]      = float(np.sum(np.abs(ctr)))

    pre_all  = wf_block[:, :n_edge].astype(np.float64)
    post_all = wf_block[:, -n_edge:].astype(np.float64)
    ch_post_minus_pre = np.mean(post_all, axis=1) - np.mean(pre_all, axis=1)
    feats["wf_post_minus_pre_ch_mean"]    = float(np.mean(ch_post_minus_pre))
    feats["wf_post_minus_pre_ch_max_abs"] = float(np.max(np.abs(ch_post_minus_pre)))

    return feats


def build_waveform_feature_df(
    wf_array: np.ndarray,
    event_map: Dict[str, int],
    event_ids: List[str],
) -> pd.DataFrame:
    """
    Compute waveform features for all requested events.
    Returns DataFrame with event_id index.
    """
    rows = []
    n_fail = 0
    for eid in event_ids:
        if eid not in event_map:
            n_fail += 1
            continue
        idx   = event_map[eid]
        block = wf_array[idx]           # (21, T)
        feats = extract_waveform_features_event(block, fs=WF_FS)
        feats["event_id"] = eid
        rows.append(feats)
    print(f"[wf_feat] Extracted {len(rows):,} event waveform feature rows  (failed={n_fail})")
    df = pd.DataFrame(rows)
    df["event_id"] = df["event_id"].astype(str)
    return df


# ─── TASK 2 — Parquet FO event-level features ────────────────────────────────

def build_parquet_event_features(df_linec: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate parquet v2 per-sensor FO features to event level.
    Uses mean, max, std across 5 Line-C sensors per event.
    """
    # Identify FO feature columns (octave bands, time-domain bands)
    skip = LEAKAGE | {
        "event_id", "sensor", "sensor_id", "sensor_code",
        "track_number", "train_type_code", "train_speed_kmh",
        "distance", "log_distance", "target_log", "target_pgv",
        "train_speed_missing",
        "acc_distance_to_track_m", "acc_distance_to_track_2_m",
        "effective_distance_to_active_track_m",
    }
    fo_cols = [
        c for c in df_linec.columns
        if pd.api.types.is_numeric_dtype(df_linec[c])
        and c not in skip
        and not c.startswith("target_")
        and "pgv" not in c.lower()
    ]

    # Aggregate: mean, max, std across sensors per event
    agg = {}
    grp = df_linec.groupby("event_id")
    for col in fo_cols:
        agg[f"{col}_mean"] = grp[col].mean()
        agg[f"{col}_max"]  = grp[col].max()
        agg[f"{col}_std"]  = grp[col].std()

    # Metadata: first value per event
    meta_cols = ["train_speed_kmh", "train_type_code", "track_number"]
    for c in meta_cols:
        if c in df_linec.columns:
            agg[c] = grp[c].first()

    event_fo = pd.DataFrame(agg).reset_index()
    event_fo["event_id"] = event_fo["event_id"].astype(str)
    # Speed missing flag
    if "train_speed_kmh" in event_fo.columns:
        event_fo["train_speed_missing"] = event_fo["train_speed_kmh"].isna().astype(float)
        event_fo["train_speed_kmh"] = event_fo["train_speed_kmh"].fillna(0.0)
    print(f"[pq_feat] Event-level parquet features: {len(event_fo):,} events, "
          f"{event_fo.shape[1]} columns")
    return event_fo


# ─── TASK 3 — Physics / profile targets ──────────────────────────────────────

def compute_physics_targets(df_linec: pd.DataFrame) -> pd.DataFrame:
    """
    For each event compute:
      c_target_event, n_event_raw, n_event_shrunk, delta_n_target,
      residual_ij for 5 sensors.
    Returns event-level DataFrame and row-level residual DataFrame.
    """
    rows = []
    residuals = []          # list of (event_id, sensor, residual)
    n_bad = 0

    for eid, grp in df_linec.groupby("event_id"):
        # Need all 5 sensors
        if len(grp) != 5 or set(grp["sensor"]) != set(SENSOR_ORDER):
            n_bad += 1
            continue
        grp = grp.set_index("sensor").reindex(SENSOR_ORDER)
        if grp["track_number"].nunique() != 1:
            n_bad += 1
            continue

        track  = int(grp["track_number"].iloc[0])
        n_tr   = N_TRACK.get(track, 1.0)
        y      = grp["target_log"].values             # (5,)
        r      = grp["distance"].values               # (5,)
        log_r  = np.log(np.maximum(r, 0.1) / R0)     # (5,)

        # 1. c_target_event: mean(y + n_tr * log_r)
        c_target = float(np.mean(y + n_tr * log_r))

        # 2. n_event_raw: OLS fit  y = c - n * log_r
        #    Demean to avoid intercept issue: X = log_r - mean, y_dm = y - mean
        X_raw = np.column_stack([np.ones(5), -log_r])   # [1, -log_r]
        coef, _, _, _ = np.linalg.lstsq(X_raw, y, rcond=None)
        n_raw  = float(coef[1])

        # 3. Shrinkage toward global n_track
        n_shrunk = float(np.clip(
            (1.0 - N_SHRINK_W) * n_tr + N_SHRINK_W * n_raw,
            N_CLIP[0], N_CLIP[1]
        ))
        delta_n = n_shrunk - n_tr

        # 4. Residual profile
        y_profile = c_target - n_shrunk * log_r
        res_vec   = y - y_profile                      # (5,)

        rows.append({
            "event_id":       eid,
            "track":          track,
            "c_target_event": c_target,
            "n_event_raw":    n_raw,
            "n_event_shrunk": n_shrunk,
            "delta_n_target": delta_n,
            "n_track":        n_tr,
        })
        for s, res in zip(SENSOR_ORDER, res_vec):
            residuals.append({"event_id": eid, "sensor": s, "residual_profile": res})

    if n_bad:
        print(f"[targets] Skipped {n_bad} events (incomplete sensor set / mixed track)")

    target_df = pd.DataFrame(rows)
    target_df["event_id"] = target_df["event_id"].astype(str)
    resid_df  = pd.DataFrame(residuals)
    resid_df["event_id"] = resid_df["event_id"].astype(str)
    print(f"[targets] Event targets: {len(target_df):,}  "
          f"c_target mean={target_df['c_target_event'].mean():.3f}  "
          f"delta_n mean={target_df['delta_n_target'].mean():.4f}")
    return target_df, resid_df


def fit_pca_residuals(
    target_df: pd.DataFrame,
    resid_df: pd.DataFrame,
    split_df: pd.DataFrame,
    n_comp: int = PCA_N_COMP,
) -> Tuple[PCA, pd.DataFrame, np.ndarray]:
    """
    Fit PCA on training residual profiles (events × 5 sensors).
    Returns fitted PCA, per-event PC score DataFrame, and explained variance array.
    """
    # Build matrix: rows = events, cols = MP4, MP8, MP10, MP1, MP2 residuals
    pivot = resid_df.pivot(index="event_id", columns="sensor", values="residual_profile")
    pivot = pivot.reindex(columns=SENSOR_ORDER).dropna()
    pivot.index = pivot.index.astype(str)

    # Identify training events
    tr_events = split_df[split_df["split"] == "train"]["event_id"].astype(str)
    tr_mask   = pivot.index.isin(tr_events)
    X_tr      = pivot.loc[tr_mask].values          # (n_train, 5)

    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X_tr)
    ev = pca.explained_variance_ratio_
    print(f"[pca] Fitted PCA n_comp={n_comp}  "
          f"VE: {ev.round(3).tolist()}  cumsum={ev.cumsum()[-1]:.3f}")

    # Project all events
    X_all  = pivot.values
    scores = pca.transform(X_all)                  # (n_events, n_comp)
    score_df = pd.DataFrame(
        scores,
        index=pivot.index,
        columns=[f"pc{i+1}_target" for i in range(n_comp)],
    ).reset_index()
    score_df.columns = ["event_id"] + [f"pc{i+1}_target" for i in range(n_comp)]
    return pca, score_df, ev


# ─── TASK 4 — Event-level model training ─────────────────────────────────────

def _xgbr_event(params: dict) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        **params,
        n_estimators=3000,
        early_stopping_rounds=50,
        eval_metric="rmse",
        tree_method="hist",
        verbosity=0,
        random_state=42,
    )


def _event_grid() -> List[dict]:
    """Hyperparameter grid for event-level models."""
    grid = []
    for depth, lr, mcw, rl in product(
        [2, 3, 4],
        [0.01, 0.02, 0.05],
        [1, 5, 10],
        [1, 10, 20],
    ):
        grid.append({
            "max_depth": depth,
            "learning_rate": lr,
            "min_child_weight": mcw,
            "reg_lambda": rl,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        })
    return grid


def train_event_model(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    weights_tr: Optional[np.ndarray] = None,
    label: str = "?",
) -> Tuple[xgb.XGBRegressor, float, np.ndarray]:
    """
    Grid-search over event-level hyperparameters.
    Returns (best model, val RMSE, val predictions).
    """
    grid = _event_grid()
    best_rmse = 1e9
    best_model = None
    best_va_pred = None
    for i, params in enumerate(grid):
        m = _xgbr_event(params)
        m.fit(X_tr, y_tr,
              sample_weight=weights_tr,
              eval_set=[(X_va, y_va)],
              verbose=False)
        vp = m.predict(X_va)
        rmse = float(np.sqrt(np.mean((vp - y_va) ** 2)))
        if rmse < best_rmse:
            best_rmse  = rmse
            best_model = m
            best_va_pred = vp
    print(f"[event_model:{label}] Best val RMSE={best_rmse:.5f}  "
          f"({len(grid)} fits)")
    return best_model, best_rmse, best_va_pred


def _high_pgv_weights(event_df: pd.DataFrame) -> np.ndarray:
    """Training weights: up-weight events with high max PGV."""
    max_pgv = event_df.get("max_pgv_event", pd.Series(np.zeros(len(event_df)))).values
    w = np.ones(len(event_df), dtype=np.float32)
    w += 2.0 * (max_pgv > 4).astype(float)
    w += 3.0 * (max_pgv > 8).astype(float)
    return w


def _profile_rmse_pgv(
    c_hat_va:   np.ndarray,
    n_hat_va:   np.ndarray,
    pc_scores_va: np.ndarray,  # (n_events, n_comp)
    pca:        PCA,
    va_event_df: pd.DataFrame,
    df_linec_va: pd.DataFrame,
) -> float:
    """
    Reconstruct profiles for validation events and compute RMSE(PGV).
    """
    n_comp   = pca.n_components_
    n_events = len(va_event_df)
    true_pgv = []
    pred_pgv = []

    # Build per-event lookup
    c_by_eid    = dict(zip(va_event_df["event_id"].values, c_hat_va))
    n_by_eid    = dict(zip(va_event_df["event_id"].values, n_hat_va))
    pc_by_eid   = {eid: pc_scores_va[i] for i, eid in enumerate(va_event_df["event_id"].values)}

    for eid, grp in df_linec_va.groupby("event_id"):
        if eid not in c_by_eid:
            continue
        c   = c_by_eid[eid]
        dn  = n_by_eid[eid]
        pcs = pc_by_eid[eid]        # (n_comp,)
        track = int(grp["track_number"].iloc[0]) if "track_number" in grp.columns else 1
        n_tr  = N_TRACK.get(track, 1.0)
        n_hat_val = float(np.clip(n_tr + dn, N_CLIP[0], N_CLIP[1]))
        pca_shape = pca.inverse_transform(pcs.reshape(1, -1))[0]   # (5,) or (n_sensors,)
        # Map to sensors in order
        for sensor in SENSOR_ORDER:
            row = grp[grp["sensor"] == sensor]
            if len(row) == 0:
                continue
            r      = float(row["distance"].iloc[0])
            log_r  = np.log(np.maximum(r, 0.1) / R0)
            s_idx  = SENSOR_ORDER.index(sensor)
            shape_i = float(pca_shape[s_idx]) if s_idx < len(pca_shape) else 0.0
            pred_log = c - n_hat_val * log_r + shape_i
            true_pgv.append(float(row["target_pgv"].iloc[0]))
            pred_pgv.append(float(np.exp(pred_log)))

    if len(true_pgv) == 0:
        return 999.0
    return float(np.sqrt(np.mean((np.array(pred_pgv) - np.array(true_pgv)) ** 2)))


# ─── Profile reconstruction ───────────────────────────────────────────────────

def reconstruct_profile_predictions(
    df_linec: pd.DataFrame,
    event_ids: np.ndarray,
    c_hat_by_eid:  dict,
    dn_hat_by_eid: dict,
    pc_hat_by_eid: dict,   # eid → (n_comp,)
    pca: PCA,
    n_comp_use: int = 2,
) -> np.ndarray:
    """
    Compute pred_log_profile for every row in df_linec that belongs to event_ids.
    Returns array aligned with df_linec's rows.
    """
    pred_log = np.full(len(df_linec), np.nan, dtype=np.float64)

    for i, row in enumerate(df_linec.itertuples(index=False)):
        eid    = str(row.event_id)
        sensor = str(row.sensor)
        if eid not in c_hat_by_eid:
            continue
        c   = c_hat_by_eid[eid]
        dn  = dn_hat_by_eid[eid]
        pcs = pc_hat_by_eid[eid]                 # (n_comp,)
        track = int(getattr(row, "track_number", 1))
        n_tr  = N_TRACK.get(track, 1.0)
        n_hat = float(np.clip(n_tr + dn, N_CLIP[0], N_CLIP[1]))
        r     = getattr(row, "distance", R0)
        log_r = np.log(max(float(r), 0.1) / R0)
        s_idx = SENSOR_ORDER.index(sensor) if sensor in SENSOR_ORDER else -1
        if s_idx < 0:
            continue
        # Partial inverse: only first n_comp_use components
        pcs_use = np.zeros(pca.n_components_)
        pcs_use[:n_comp_use] = pcs[:n_comp_use]
        shape_vec = pca.inverse_transform(pcs_use.reshape(1, -1))[0]
        shape_i   = float(shape_vec[s_idx])
        pred_log[i] = c - n_hat * log_r + shape_i

    return pred_log


def apply_monotonic(df: pd.DataFrame, pred_col: str) -> pd.Series:
    """Apply cumulative minimum monotonicity (distance increasing → pred decreasing)."""
    out = df[pred_col].copy()
    for eid, grp in df.groupby("event_id"):
        idxs = grp.sort_values("distance").index
        vals = out[idxs].values
        # cumulative minimum ensures monotone non-increasing
        for k in range(1, len(vals)):
            vals[k] = min(vals[k], vals[k - 1])
        out[idxs] = vals
    return out


# ─── TASK 5 — Row-level features and local residual model ────────────────────

def build_row_features(
    df_linec: pd.DataFrame,
    event_feature_df: pd.DataFrame,
    pred_profile_col: str = "pred_log_profile",
    c_hat_col: str = "c_hat_profile",
    n_hat_col: str = "n_hat_profile",
    pc1_col: str = "pc1_hat",
    pc2_col: str = "pc2_hat",
) -> pd.DataFrame:
    """
    Build row-level feature matrix:
    event features + distance features + pred profile features + interactions.
    """
    # Event-level features joined to row level
    row_df = df_linec.merge(
        event_feature_df, on="event_id", how="left", suffixes=("", "_ev")
    )

    # Distance-frequency interactions
    log_d = row_df["log_distance"]
    if "wf_spectral_centroid" in row_df.columns:
        row_df["feat_logd_x_cent"]     = log_d * row_df["wf_spectral_centroid"]
    if "wf_high_low_energy_ratio" in row_df.columns:
        row_df["feat_logd_x_hlrat"]    = log_d * row_df["wf_high_low_energy_ratio"]
    if "wf_band_1_5" in row_df.columns:
        row_df["feat_logd_x_band1_5"]  = log_d * row_df["wf_band_1_5"]
        row_df["feat_logd_x_band5_10"] = log_d * row_df.get("wf_band_5_10", 0.0)
        row_df["feat_logd_x_band10_20"] = log_d * row_df.get("wf_band_10_20", 0.0)
        row_df["feat_logd_x_band20_40"] = log_d * row_df.get("wf_band_20_40", 0.0)
        row_df["feat_logd_x_band40_80"] = log_d * row_df.get("wf_band_40_80", 0.0)
    if "train_speed_kmh" in row_df.columns:
        row_df["feat_logd_x_speed"] = log_d * row_df["train_speed_kmh"]
    if "train_type_code" in row_df.columns:
        row_df["feat_logd_x_type"]  = log_d * row_df["train_type_code"]

    # Sensor-spectral interactions
    sc = row_df["sensor_code"]
    if "wf_spectral_centroid" in row_df.columns:
        row_df["feat_sc_x_cent"]  = sc * row_df["wf_spectral_centroid"]
    if "wf_high_low_energy_ratio" in row_df.columns:
        row_df["feat_sc_x_hlrat"] = sc * row_df["wf_high_low_energy_ratio"]

    return row_df


def _row_feature_cols(row_df: pd.DataFrame, pq_fo_cols: List[str]) -> List[str]:
    explicit = [
        "pred_log_profile", "c_hat_profile", "n_hat_profile",
        "pc1_hat", "pc2_hat",
        "distance", "log_distance", "sensor_code", "track_number",
        "train_speed_kmh", "train_speed_missing", "train_type_code",
    ]
    interaction_cols = [c for c in row_df.columns if c.startswith("feat_")]
    wf_cols = [c for c in row_df.columns if c.startswith("wf_")]
    all_cols = explicit + interaction_cols + wf_cols + [
        c for c in pq_fo_cols if c in row_df.columns and c not in explicit
    ]
    # deduplicate, keep only valid non-leakage numeric cols
    seen = set()
    out = []
    for c in all_cols:
        if c in seen or c not in row_df.columns:
            continue
        if c in LEAKAGE or not pd.api.types.is_numeric_dtype(row_df[c]):
            continue
        if c.startswith("target_") or "pgv" in c.lower():
            continue
        seen.add(c)
        out.append(c)
    return out


def _local_grid() -> List[dict]:
    grid = []
    for depth, lr, mcw, rl in product(
        [3, 4, 5],
        [0.01, 0.02, 0.05],
        [1, 5],
        [1, 10],
    ):
        grid.append({
            "max_depth": depth,
            "learning_rate": lr,
            "min_child_weight": mcw,
            "reg_lambda": rl,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        })
    return grid


def _local_weights(df: pd.DataFrame, scheme: str) -> np.ndarray:
    s  = df["sensor"].values
    tp = df["target_pgv"].values
    w  = np.ones(len(df), dtype=np.float32)
    if scheme == "mp4_hi":
        w += 1.5 * (s == "MP4") + 1.0 * (tp > 4) + 1.0 * (tp > 8)
    elif scheme == "agg_hi":
        w += 1.0 * (s == "MP4") + 2.0 * (tp > 4) + 3.0 * (tp > 8)
    return w


def train_local_residual_model(
    row_tr: pd.DataFrame, row_va: pd.DataFrame, row_te: pd.DataFrame,
    feat_cols: List[str],
    pred_profile_col: str = "pred_log_profile",
) -> Tuple[xgb.XGBRegressor, float, str]:
    """
    Train local residual XGBoost: target = target_log - pred_log_profile.
    Searches over grid × 3 weight schemes.  Selects by val RMSE(PGV).
    Returns (best model, best val RMSE PGV, scheme label).
    """
    valid_cols = [c for c in feat_cols if c in row_tr.columns and c in row_va.columns]
    y_tr = row_tr["target_log"].values - row_tr[pred_profile_col].values
    y_va = row_va["target_log"].values - row_va[pred_profile_col].values
    X_tr = row_tr[valid_cols].values.astype(np.float32)
    X_va = row_va[valid_cols].values.astype(np.float32)
    tgv  = row_va["target_pgv"].values
    p3_va = row_va[pred_profile_col].values

    grid    = _local_grid()
    schemes = ["uniform", "mp4_hi", "agg_hi"]

    best_rmse  = 1e9
    best_model = None
    best_scheme = "uniform"

    for scheme in schemes:
        w_tr = _local_weights(row_tr, scheme)
        for params in grid:
            m = xgb.XGBRegressor(
                **params,
                n_estimators=3000,
                early_stopping_rounds=50,
                eval_metric="rmse",
                tree_method="hist",
                verbosity=0,
                random_state=42,
            )
            m.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_va, y_va)], verbose=False)
            res_pred = m.predict(X_va)
            pred_pgv = np.exp(p3_va + res_pred)
            rmse = float(np.sqrt(np.mean((pred_pgv - tgv) ** 2)))
            if rmse < best_rmse:
                best_rmse  = rmse
                best_model = m
                best_scheme = scheme

    print(f"[local_res] Best val RMSE(PGV)={best_rmse:.4f}  "
          f"scheme={best_scheme}  ({len(grid) * len(schemes)} fits)")
    return best_model, best_rmse, best_scheme


# ─── TASK 6 / 5 — Load saved baseline predictions ────────────────────────────

def load_pxgbr_predictions(row_te: pd.DataFrame) -> Dict[str, Optional[np.ndarray]]:
    """
    Load PXGBR-R2 and PXGBR ensemble top-5 test predictions from their saved
    output directories.  Aligns predictions to row_te order by event_id+sensor.
    Returns dict {label: array | None}.
    """
    result: Dict[str, Optional[np.ndarray]] = {}

    # ── PXGBR-R2 ──────────────────────────────────────────────────────────────
    pxgbr_dir = find_pxgbr_dir()
    if pxgbr_dir is not None:
        pred_path = pxgbr_dir / "predictions_test.parquet"
        if pred_path.exists():
            try:
                pxgbr_df = pd.read_parquet(pred_path)
                pxgbr_df["event_id"] = pxgbr_df["event_id"].astype(str)
                pxgbr_df["sensor"]   = pxgbr_df["sensor"].astype(str)
                # PXGBR-R2 uses column pred_log_R2
                pred_col = next((c for c in ["pred_log_R2", "pred_log_r2",
                                              "pred_log_final"]
                                  if c in pxgbr_df.columns), None)
                if pred_col:
                    aligned = row_te[["event_id", "sensor"]].merge(
                        pxgbr_df[["event_id", "sensor", pred_col]],
                        on=["event_id", "sensor"], how="left"
                    )[pred_col].values
                    result["PXGBR-R2"] = aligned
                    n_ok = int(np.isfinite(aligned).sum())
                    print(f"[baseline_pxgbr] Loaded PXGBR-R2 from {pxgbr_dir.name}  "
                          f"n_ok={n_ok}")
                else:
                    print(f"[WARN] PXGBR-R2: pred_log_R2 column not found in "
                          f"{pred_path.name}")
                    result["PXGBR-R2"] = None
            except Exception as exc:
                print(f"[WARN] Could not load PXGBR-R2 predictions: {exc}")
                result["PXGBR-R2"] = None
        else:
            print(f"[WARN] PXGBR-R2 predictions not found: {pred_path}")
            result["PXGBR-R2"] = None
    else:
        print("[WARN] No pxgbr_linec_v001_* directory found")
        result["PXGBR-R2"] = None

    # ── PXGBR ensemble top-5 ──────────────────────────────────────────────────
    ens_dir = find_pxgbr_ens_dir()
    if ens_dir is not None:
        ens_path = ens_dir / "ensemble_predictions_test.parquet"
        if not ens_path.exists():
            ens_path = ens_dir / "predictions_test.parquet"
        if ens_path.exists():
            try:
                ens_df = pd.read_parquet(ens_path)
                ens_df["event_id"] = ens_df["event_id"].astype(str)
                ens_df["sensor"]   = ens_df["sensor"].astype(str)
                ens_col = next((c for c in ["pred_log_ens_top5", "pred_log_top5",
                                            "pred_log_ens5", "pred_log_final"]
                                if c in ens_df.columns), None)
                if ens_col:
                    aligned = row_te[["event_id", "sensor"]].merge(
                        ens_df[["event_id", "sensor", ens_col]],
                        on=["event_id", "sensor"], how="left"
                    )[ens_col].values
                    result["PXGBR_ens_top5"] = aligned
                    n_ok = int(np.isfinite(aligned).sum())
                    print(f"[baseline_pxgbr_ens] Loaded PXGBR_ens_top5 from "
                          f"{ens_dir.name}  n_ok={n_ok}")
                else:
                    print(f"[WARN] PXGBR_ens_top5: no matching column in {ens_path.name}; "
                          f"available: {list(ens_df.columns)}")
                    result["PXGBR_ens_top5"] = None
            except Exception as exc:
                print(f"[WARN] Could not load PXGBR ensemble predictions: {exc}")
                result["PXGBR_ens_top5"] = None
        else:
            print(f"[WARN] PXGBR ensemble predictions not found in {ens_dir}")
            result["PXGBR_ens_top5"] = None
    else:
        print("[WARN] No pxgbr_ensemble_linec_v001_* directory found")
        result["PXGBR_ens_top5"] = None

    return result


# ─── TASK 6 — Metrics ─────────────────────────────────────────────────────────

def compute_metrics(
    pred_log: np.ndarray,
    true_log: np.ndarray,
    sensors:  np.ndarray,
    event_ids: np.ndarray,
    label: str,
) -> dict:
    pp = np.exp(pred_log); tp = np.exp(true_log)
    rmse_p = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    bias_p = lambda a, b: float(np.mean(a - b))
    r2_log = lambda pl, tl: float(
        1 - np.sum((tl - pl) ** 2) / max(np.sum((tl - tl.mean()) ** 2), 1e-12)
    )

    m = {
        "model":    label,
        "rmse_log": rmse_p(pred_log, true_log),
        "rmse_pgv": rmse_p(pp, tp),
        "mae_pgv":  float(np.mean(np.abs(pp - tp))),
        "r2_log":   r2_log(pred_log, true_log),
        "bias_pgv": bias_p(pp, tp),
    }

    m["per_sensor"] = {}
    for s in SENSOR_ORDER:
        msk = sensors == s
        if msk.any():
            m["per_sensor"][s] = {
                "rmse_pgv": rmse_p(pp[msk], tp[msk]),
                "rmse_log": rmse_p(pred_log[msk], true_log[msk]),
                "bias_pgv": bias_p(pp[msk], tp[msk]),
                "bias_log": bias_p(pred_log[msk], true_log[msk]),
            }

    for thr in [4.0, 8.0]:
        hi = tp > thr
        if hi.any():
            m[f"pgv_gt{int(thr)}"] = {"n": int(hi.sum()), "rmse_pgv": rmse_p(pp[hi], tp[hi])}
        mp4hi = (sensors == "MP4") & hi
        if mp4hi.any():
            m[f"mp4_pgv_gt{int(thr)}"] = {"n": int(mp4hi.sum()), "rmse_pgv": rmse_p(pp[mp4hi], tp[mp4hi])}

    # Monotonicity violation rate
    viol = tot = 0
    df_tmp = pd.DataFrame({"eid": event_ids, "s": sensors, "p": pred_log})
    for eid, grp in df_tmp.groupby("eid"):
        if len(grp) != 5:
            continue
        row = grp.set_index("s").reindex(SENSOR_ORDER).dropna()
        if len(row) < 2:
            continue
        vals = row["p"].values
        for k in range(len(vals) - 1):
            tot += 1
            viol += int(vals[k] < vals[k + 1])   # increasing = violation
    m["mono_viol_rate"] = viol / tot if tot else 0.0
    return m


def print_metrics(m: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {m['model']}")
    print(f"  RMSE(PGV)={m['rmse_pgv']:.4f}  RMSE(log)={m['rmse_log']:.4f}  "
          f"R²(log)={m['r2_log']:.4f}  MAE(PGV)={m['mae_pgv']:.4f}")
    per = m.get("per_sensor", {})
    for s in SENSOR_ORDER:
        if s in per:
            print(f"    {s}: RMSE(PGV)={per[s]['rmse_pgv']:.4f}  "
                  f"bias={per[s]['bias_pgv']:+.4f}")
    for k in ["pgv_gt4", "pgv_gt8", "mp4_pgv_gt4"]:
        if k in m:
            print(f"  {k}: n={m[k]['n']} rmse_pgv={m[k]['rmse_pgv']:.4f}")
    print(f"  mono_viol_rate={m.get('mono_viol_rate', 'N/A')}")


# ─── TASK 6 — Baseline A1 (direct tabular no physics) ────────────────────────

def train_baseline_a1(df_linec: pd.DataFrame, split_df: pd.DataFrame) -> dict:
    """Quick A1 direct tabular XGBoost (no physics features)."""
    # df_linec may already have split column from main(); avoid double-merge
    if "split" in df_linec.columns:
        df = df_linec.copy()
    else:
        df = df_linec.merge(split_df, on="event_id", how="inner")
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"]

    skip = LEAKAGE | {"event_id", "sensor", "split", "target_log", "target_pgv"}
    base = ["distance", "log_distance", "sensor_code", "track_number",
            "train_speed_kmh", "train_speed_missing", "train_type_code"]
    fo_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in skip and not c.startswith("target_")
        and "pgv" not in c.lower()
    ]
    feat_cols = base + [c for c in fo_cols if c not in base]
    feat_cols = [c for c in feat_cols if c in df.columns]

    best_rmse = 1e9; best_m = None
    for depth, lr in product([3, 4], [0.02, 0.05]):
        m = xgb.XGBRegressor(
            max_depth=depth, learning_rate=lr,
            n_estimators=3000, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", verbosity=0, random_state=42,
            early_stopping_rounds=50, eval_metric="rmse",
        )
        m.fit(tr[feat_cols].values, tr["target_log"].values,
              eval_set=[(va[feat_cols].values, va["target_log"].values)],
              verbose=False)
        vp = m.predict(va[feat_cols].values)
        rmse = float(np.sqrt(np.mean((np.exp(vp) - va["target_pgv"].values) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse; best_m = m

    te_pred = best_m.predict(te[feat_cols].values)
    va_pred = best_m.predict(va[feat_cols].values)
    print(f"[baseline_A1] val RMSE(PGV)={best_rmse:.4f}")
    return {
        "model": best_m,
        "feat_cols": feat_cols,
        "val_pred_log": va_pred,
        "test_pred_log": te_pred,
        "val_sensors": va["sensor"].values,
        "test_sensors": te["sensor"].values,
        "val_event_ids": va["event_id"].values,
        "test_event_ids": te["event_id"].values,
        "val_true_log": va["target_log"].values,
        "test_true_log": te["target_log"].values,
    }


# ─── TASK 7 — Plotting ────────────────────────────────────────────────────────

def _save_fig(path: Path, dpi: int = 150) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_scatter(true_pgv, pred_pgv, label, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_pgv, pred_pgv, alpha=0.3, s=8)
    lim = (0, max(true_pgv.max(), pred_pgv.max()) * 1.05)
    ax.plot(lim, lim, "r--", lw=1.2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True PGV (mm/s)"); ax.set_ylabel("Predicted PGV (mm/s)")
    ax.set_title(f"{label} — measured vs predicted")
    _save_fig(out_path)


def plot_per_sensor_rmse(metrics_list: List[dict], out_path: Path):
    models = [m["model"] for m in metrics_list]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SENSOR_ORDER))
    width = 0.8 / len(models)
    for i, m in enumerate(metrics_list):
        rmses = [m["per_sensor"].get(s, {}).get("rmse_pgv", np.nan) for s in SENSOR_ORDER]
        ax.bar(x + i * width, rmses, width, label=m["model"])
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(SENSOR_ORDER)
    ax.set_ylabel("RMSE(PGV)  [mm/s]")
    ax.set_title("Per-sensor RMSE(PGV) — FO-PhysProfile vs baselines")
    ax.legend(fontsize=8)
    _save_fig(out_path)


def plot_physics_scatter(true_vals, pred_vals, xlabel, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_vals, pred_vals, alpha=0.4, s=8)
    lo = min(true_vals.min(), pred_vals.min())
    hi = max(true_vals.max(), pred_vals.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    r2 = float(1 - np.sum((pred_vals - true_vals) ** 2) /
               max(np.sum((true_vals - true_vals.mean()) ** 2), 1e-12))
    ax.text(0.05, 0.92, f"R²={r2:.3f}", transform=ax.transAxes, fontsize=10)
    _save_fig(out_path)


def plot_pca_components(pca: PCA, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, comp in enumerate(pca.components_):
        ax.plot(SENSOR_ORDER, comp, marker="o", label=f"PC{i+1} (VE={pca.explained_variance_ratio_[i]:.2f})")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Sensor"); ax.set_ylabel("Loading")
    ax.set_title("PCA components of residual profiles")
    ax.legend()
    _save_fig(out_path)


def plot_feature_importance(model: xgb.XGBRegressor, feat_cols: List[str],
                            title: str, out_path: Path, top_n: int = 30):
    imp = pd.DataFrame({
        "feature": feat_cols,
        "importance": model.feature_importances_,
    }).nlargest(top_n, "importance")
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.25)))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    _save_fig(out_path)


def plot_residual_vs_distance(df: pd.DataFrame, residual_col: str,
                               title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in SENSOR_ORDER:
        msk = df["sensor"] == s
        if msk.any():
            ax.scatter(df.loc[msk, "distance"], df.loc[msk, residual_col],
                       alpha=0.25, s=6, label=s)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Distance (m)"); ax.set_ylabel("Residual (log scale)")
    ax.set_title(title); ax.legend(fontsize=8)
    _save_fig(out_path)


def plot_high_pgv_profiles(df: pd.DataFrame, pred_col: str, n_events: int,
                            out_path: Path):
    """Plot the n_events highest-PGV events with true and predicted profiles."""
    top_eids = (
        df.groupby("event_id")["target_pgv"]
        .max()
        .nlargest(n_events)
        .index
    )
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    for ax, eid in zip(axes.flat, top_eids):
        ev = df[df["event_id"] == eid].sort_values("distance")
        ax.plot(ev["distance"], np.exp(ev["target_log"]), "ko-", ms=4, label="true")
        if pred_col in ev.columns:
            ax.plot(ev["distance"], np.exp(ev[pred_col]), "r^--", ms=4, label="pred")
        ax.set_title(str(eid)[:12], fontsize=7)
        ax.set_xlabel("dist (m)", fontsize=7)
        ax.set_ylabel("PGV", fontsize=7)
    axes.flat[0].legend(fontsize=7)
    plt.suptitle("High-PGV profiles — FO-PhysProfile", fontsize=10)
    _save_fig(out_path)


# ─── Leakage audit ────────────────────────────────────────────────────────────

# Full set of leakage token patterns (substring match)
_LEAKAGE_TOKENS = [
    "target_pgv_z_mms", "target_pgv", "target_log",
    "c_target", "c_target_event",
    "residual_log", "scaled_residual",
    "max_pgv", "mp4_pgv",
    "n_event_raw", "n_event_shrunk", "delta_n_target",
    "pc1_target", "pc2_target", "pc3_target",
]


def _check_leakage_tokens(feat_cols: List[str]) -> List[str]:
    """Return any feature column whose name contains a leakage token."""
    found = []
    for col in feat_cols:
        col_low = col.lower()
        for tok in _LEAKAGE_TOKENS:
            if tok in col_low:
                found.append(col)
                break
    return found


def assert_no_leakage(feat_cols: List[str], label: str) -> None:
    """Hard assertion: raise ValueError if any leakage token appears in features."""
    bad = _check_leakage_tokens(feat_cols)
    if bad:
        msg = (f"LEAKAGE DETECTED in {label} feature matrix!\n"
               f"Offending columns: {bad}")
        raise ValueError(msg)
    print(f"[leakage_ok] {label}: {len(feat_cols)} features, no leakage tokens detected.")


def write_leakage_audit(feat_cols: List[str], out_path: Path) -> None:
    bad = _check_leakage_tokens(feat_cols)
    with open(out_path, "w") as f:
        f.write("FO-PhysProfile v1 — Leakage Audit\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total feature columns: {len(feat_cols)}\n")
        f.write(f"Leakage columns detected: {len(bad)}\n\n")
        f.write("Tokens checked:\n")
        for tok in _LEAKAGE_TOKENS:
            f.write(f"  {tok}\n")
        f.write("\n")
        if bad:
            f.write("LEAKAGE COLUMNS:\n")
            for c in bad:
                f.write(f"  {c}\n")
        else:
            f.write("PASS — no leakage columns detected.\n\n")
        f.write("Safe feature list:\n")
        for c in feat_cols:
            f.write(f"  {c}\n")


# ─── SLURM-friendly checkpoint saves ─────────────────────────────────────────

def _save_checkpoint(obj, path: Path, label: str):
    if path.suffix == ".parquet" and isinstance(obj, pd.DataFrame):
        obj.to_parquet(path, index=False)
    elif path.suffix == ".pkl":
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    elif path.suffix == ".csv" and isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    print(f"[ckpt] Saved {label} → {path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    out_dir = make_output_dir(MODELS_ROOT)
    print(f"\nOutput dir: {out_dir}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Load data
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1 — Load data")
    print("=" * 70)

    df_linec = load_linec_parquet()
    wf_array, event_map, actual_fs = load_waveforms()
    p3_dir   = find_p3_dir()
    split_df = get_split_labels(df_linec, p3_dir)

    # Assign split to rows
    df_linec = df_linec.merge(split_df, on="event_id", how="inner")
    print(f"[main] After split merge: {len(df_linec):,} rows  "
          f"splits={df_linec['split'].value_counts().to_dict()}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Feature extraction (TASK 2)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2 — Feature extraction")
    print("=" * 70)

    all_event_ids = df_linec["event_id"].unique().tolist()

    # 2a. Waveform features (spatial + temporal)
    print("\n[2a] Extracting waveform-derived features...")
    try:
        wf_feat_df = build_waveform_feature_df(wf_array, event_map, all_event_ids)
        wf_ok = True
    except Exception as exc:
        print(f"[WARN] Waveform feature extraction failed: {exc}")
        print("[WARN] Continuing with parquet features only.")
        wf_feat_df = pd.DataFrame({"event_id": all_event_ids})
        wf_ok = False

    # 2b. Parquet v2 event-level aggregated features
    print("\n[2b] Aggregating parquet v2 FO features at event level...")
    pq_event_df = build_parquet_event_features(df_linec)

    # 2c. Merge all event features
    event_feat_df = pq_event_df.merge(wf_feat_df, on="event_id", how="left")
    # Add max PGV per event for training weights (allowed, training only)
    max_pgv_ev = df_linec.groupby("event_id")["target_pgv"].max().reset_index()
    max_pgv_ev.columns = ["event_id", "max_pgv_event"]
    event_feat_df = event_feat_df.merge(max_pgv_ev, on="event_id", how="left")

    # Merge split to event-level
    event_feat_df = event_feat_df.merge(split_df, on="event_id", how="left")

    _save_checkpoint(event_feat_df.drop(columns=["max_pgv_event"], errors="ignore"),
                     out_dir / "event_features.parquet", "event_features")

    # Identify event-level FO feature columns (exclude targets + leakage + split)
    skip_ev = LEAKAGE | {"event_id", "split", "max_pgv_event",
                         "track", "track_number", "train_type_code",
                         "train_speed_kmh", "train_speed_missing"}
    meta_ev  = ["train_speed_kmh", "train_speed_missing", "train_type_code", "track_number"]
    fo_event_feat_cols = [
        c for c in event_feat_df.columns
        if c not in skip_ev and pd.api.types.is_numeric_dtype(event_feat_df[c])
        and not c.startswith("target_") and "pgv" not in c.lower()
        and not c.startswith("pc") and c != "c_target_event"
        and "n_event" not in c and "delta_n" not in c
    ]
    fo_event_feat_cols += [c for c in meta_ev if c in event_feat_df.columns
                           and c not in fo_event_feat_cols]

    # Write feature list
    (out_dir / "feature_columns_event.txt").write_text("\n".join(fo_event_feat_cols))
    print(f"[feat] Event-level feature columns: {len(fo_event_feat_cols)}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Physics targets (TASK 3)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3 — Physics / profile targets")
    print("=" * 70)

    target_df, resid_df = compute_physics_targets(df_linec)
    pca, pc_score_df, pca_ve = fit_pca_residuals(target_df, resid_df, split_df)

    # Merge all event targets
    event_full_df = (
        event_feat_df
        .merge(target_df, on="event_id", how="inner")
        .merge(pc_score_df, on="event_id", how="inner")
    )
    _save_checkpoint(event_full_df, out_dir / "profile_targets_event.parquet", "profile_targets")
    _save_checkpoint(pca, out_dir / "residual_profile_pca.pkl", "pca")
    _save_checkpoint(
        pd.DataFrame(pca.components_, columns=SENSOR_ORDER),
        out_dir / "pca_components.csv", "pca_components"
    )

    # Split event-level tables
    ev_tr = event_full_df[event_full_df["split"] == "train"].copy()
    ev_va = event_full_df[event_full_df["split"] == "val"].copy()
    ev_te = event_full_df[event_full_df["split"] == "test"].copy()
    print(f"[split-event] train={len(ev_tr)}  val={len(ev_va)}  test={len(ev_te)}")

    # Prepare event-level arrays
    fo_valid = [c for c in fo_event_feat_cols if c in event_full_df.columns]
    X_tr_ev  = ev_tr[fo_valid].fillna(0.0).values.astype(np.float32)
    X_va_ev  = ev_va[fo_valid].fillna(0.0).values.astype(np.float32)
    X_te_ev  = ev_te[fo_valid].fillna(0.0).values.astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — Event-level models (TASK 4)  —  profile-aware selection + OOF
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 4 — Event-level profile models (profile-aware selection)")
    print("=" * 70)

    from sklearn.model_selection import KFold

    w_tr_ev = _high_pgv_weights(ev_tr)
    df_va_rows = df_linec[df_linec["split"] == "val"].copy()
    df_te_rows = df_linec[df_linec["split"] == "test"].copy()
    df_tr_rows = df_linec[df_linec["split"] == "train"].copy()

    # Helper: return top-K models from grid sorted by individual val RMSE
    def _train_top_k(X_tr, y_tr, X_va, y_va, w_tr, label, top_k=5):
        """Train full grid, return top-K (model, val_pred) sorted by val RMSE."""
        grid = _event_grid()
        results = []
        for params in grid:
            m = _xgbr_event(params)
            m.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_va, y_va)], verbose=False)
            vp = m.predict(X_va)
            rmse = float(np.sqrt(np.mean((vp - y_va) ** 2)))
            results.append((rmse, m, vp))
        results.sort(key=lambda x: x[0])
        print(f"[{label}] top-1 val RMSE={results[0][0]:.5f}  "
              f"({len(grid)} fits, keeping top {top_k})")
        return results[:top_k]

    # ── Model C: select by individual target RMSE (amplitude) ────────────────
    print("\n[C] Training c_target_event model...")
    c_cands = _train_top_k(X_tr_ev, ev_tr["c_target_event"].values,
                            X_va_ev, ev_va["c_target_event"].values,
                            w_tr_ev, "C", top_k=3)
    m_c, va_c = c_cands[0][1], c_cands[0][2]
    te_c = m_c.predict(X_te_ev)
    r2_c = float(1 - np.sum((va_c - ev_va["c_target_event"].values) ** 2) /
                 max(np.sum((ev_va["c_target_event"].values
                              - ev_va["c_target_event"].values.mean()) ** 2), 1e-12))
    print(f"[C] c_hat vs c_target R²={r2_c:.4f}  (target: > 0.30)")
    pd.DataFrame({"feature": fo_valid,
                  "importance": m_c.feature_importances_}
                 ).sort_values("importance", ascending=False).to_csv(
        out_dir / "feature_importance_c.csv", index=False)

    # ── Model N: select by validation PROFILE RMSE(PGV) ──────────────────────
    print("\n[N] Training delta_n_target model (profile-aware selection)...")
    n_cands = _train_top_k(X_tr_ev, ev_tr["delta_n_target"].values,
                            X_va_ev, ev_va["delta_n_target"].values,
                            w_tr_ev, "N", top_k=5)
    # Zero-pad PC scores for profile RMSE evaluation during N search
    zero_pc = np.zeros((len(ev_va), 2), dtype=np.float32)
    best_n_rmse_prof = 1e9
    best_n_idx = 0
    for k, (_, mn_cand, va_n_cand) in enumerate(n_cands):
        rp = _profile_rmse_pgv(va_c, va_n_cand, zero_pc, pca, ev_va, df_va_rows)
        if rp < best_n_rmse_prof:
            best_n_rmse_prof = rp
            best_n_idx = k
    m_n, va_n = n_cands[best_n_idx][1], n_cands[best_n_idx][2]
    te_n = m_n.predict(X_te_ev)
    print(f"[N] Profile-selected idx={best_n_idx}  "
          f"val profile RMSE(PGV)={best_n_rmse_prof:.4f}")
    pd.DataFrame({"feature": fo_valid,
                  "importance": m_n.feature_importances_}
                 ).sort_values("importance", ascending=False).to_csv(
        out_dir / "feature_importance_n.csv", index=False)

    # ── Model PC1: select by validation PROFILE RMSE(PGV) ────────────────────
    print("\n[PC1] Training pc1_target model (profile-aware selection)...")
    pc1_cands = _train_top_k(X_tr_ev, ev_tr["pc1_target"].values,
                              X_va_ev, ev_va["pc1_target"].values,
                              w_tr_ev, "PC1", top_k=5)
    zero_pc2 = np.zeros(len(ev_va), dtype=np.float32)
    best_pc1_rmse_prof = 1e9
    best_pc1_idx = 0
    for k, (_, mpc1_cand, va_pc1_cand) in enumerate(pc1_cands):
        pc_mat = np.column_stack([va_pc1_cand, zero_pc2])
        rp = _profile_rmse_pgv(va_c, va_n, pc_mat, pca, ev_va, df_va_rows)
        if rp < best_pc1_rmse_prof:
            best_pc1_rmse_prof = rp
            best_pc1_idx = k
    m_pc1, va_pc1 = pc1_cands[best_pc1_idx][1], pc1_cands[best_pc1_idx][2]
    te_pc1 = m_pc1.predict(X_te_ev)
    print(f"[PC1] Profile-selected idx={best_pc1_idx}  "
          f"val profile RMSE(PGV)={best_pc1_rmse_prof:.4f}")

    # ── Model PC2: select by validation PROFILE RMSE(PGV) ────────────────────
    print("\n[PC2] Training pc2_target model (profile-aware selection)...")
    pc2_cands = _train_top_k(X_tr_ev, ev_tr["pc2_target"].values,
                              X_va_ev, ev_va["pc2_target"].values,
                              w_tr_ev, "PC2", top_k=5)
    best_pc2_rmse_prof = 1e9
    best_pc2_idx = 0
    for k, (_, mpc2_cand, va_pc2_cand) in enumerate(pc2_cands):
        pc_mat = np.column_stack([va_pc1, va_pc2_cand])
        rp = _profile_rmse_pgv(va_c, va_n, pc_mat, pca, ev_va, df_va_rows)
        if rp < best_pc2_rmse_prof:
            best_pc2_rmse_prof = rp
            best_pc2_idx = k
    m_pc2, va_pc2 = pc2_cands[best_pc2_idx][1], pc2_cands[best_pc2_idx][2]
    te_pc2 = m_pc2.predict(X_te_ev)
    print(f"[PC2] Profile-selected idx={best_pc2_idx}  "
          f"val profile RMSE(PGV)={best_pc2_rmse_prof:.4f}")

    for m_mod, name in [(m_pc1, "pc1"), (m_pc2, "pc2")]:
        pd.DataFrame({"feature": fo_valid,
                      "importance": m_mod.feature_importances_}
                     ).sort_values("importance", ascending=False).to_csv(
            out_dir / f"feature_importance_{name}.csv", index=False)

    # ── PC3 (optional) ────────────────────────────────────────────────────────
    if PCA_N_COMP >= 3 and "pc3_target" in ev_tr.columns and pca_ve[2] > 0.05:
        print("\n[PC3] Training pc3_target model (VE={:.3f})...".format(pca_ve[2]))
        m_pc3, _, va_pc3 = train_event_model(
            X_tr_ev, ev_tr["pc3_target"].values,
            X_va_ev, ev_va["pc3_target"].values,
            weights_tr=w_tr_ev, label="PC3",
        )
        te_pc3 = m_pc3.predict(X_te_ev)
        n_comp_use = 3
    else:
        m_pc3 = va_pc3 = te_pc3 = None
        n_comp_use = 2

    # ── Build prediction dicts ────────────────────────────────────────────────
    def _pred_dict(df_ev, c_arr, n_arr, pc1_arr, pc2_arr, pc3_arr=None):
        c_by  = dict(zip(df_ev["event_id"].values, c_arr))
        dn_by = dict(zip(df_ev["event_id"].values, n_arr))
        pc_by = {}
        for i, eid in enumerate(df_ev["event_id"].values):
            v = np.array([pc1_arr[i], pc2_arr[i]])
            if pc3_arr is not None:
                v = np.append(v, pc3_arr[i])
            pc_by[eid] = v
        return c_by, dn_by, pc_by

    c_va, dn_va, pc_va = _pred_dict(ev_va, va_c, va_n, va_pc1, va_pc2,
                                     va_pc3 if m_pc3 else None)
    c_te, dn_te, pc_te = _pred_dict(ev_te, te_c, te_n, te_pc1, te_pc2,
                                     te_pc3 if m_pc3 else None)

    # ── Final profile RMSE on val ─────────────────────────────────────────────
    va_profile_rmse_pgv = _profile_rmse_pgv(
        va_c, va_n, np.column_stack([va_pc1, va_pc2]),
        pca, ev_va, df_va_rows,
    )
    print(f"\n[profile] Val RMSE(PGV) profile-only (final combination) = "
          f"{va_profile_rmse_pgv:.4f}")

    # ── Reconstruct val and test splits ───────────────────────────────────────
    def _attach_hat_cols(df_rows, c_by, dn_by, pc_by):
        """Add pred_log_profile / c_hat / n_hat / pc_hat columns in-place."""
        pred = reconstruct_profile_predictions(
            df_rows, df_rows["event_id"].unique(),
            c_by, dn_by, pc_by, pca, n_comp_use=n_comp_use,
        )
        df_rows = df_rows.copy()
        df_rows["pred_log_profile"] = pred
        df_rows["c_hat_profile"]    = df_rows["event_id"].map(c_by).astype(float)
        # n_hat: n_track + delta_n (clipped)
        track_by_eid = df_rows.groupby("event_id")["track_number"].first().astype(int)
        df_rows["n_hat_profile"] = df_rows["event_id"].map(
            lambda e: float(np.clip(
                N_TRACK.get(int(track_by_eid.get(e, 1)), 1.0) + dn_by.get(e, 0.0),
                *N_CLIP))
            if e in dn_by else np.nan
        )
        df_rows["pc1_hat"] = df_rows["event_id"].map(
            {e: v[0] for e, v in pc_by.items()})
        df_rows["pc2_hat"] = df_rows["event_id"].map(
            {e: v[1] if len(v) > 1 else 0.0 for e, v in pc_by.items()})
        return df_rows

    df_va_rows = _attach_hat_cols(df_va_rows, c_va, dn_va, pc_va)
    df_te_rows = _attach_hat_cols(df_te_rows, c_te, dn_te, pc_te)

    # ── Train split: GroupKFold OOF predictions (no leakage) ──────────────────
    # Use 5-fold KFold on train events to get honest OOF profile predictions.
    # These OOF values are used ONLY to compute the residual target for the
    # local residual model (Stage 2).  Full-train models (m_c, m_n, m_pc*) are
    # used for val/test prediction and are NOT retrained here.
    print("\n[OOF] Computing GroupKFold OOF train profile predictions (5 folds)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    n_tr_events = len(ev_tr)
    oof_c    = np.zeros(n_tr_events, dtype=np.float64)
    oof_n    = np.zeros(n_tr_events, dtype=np.float64)
    oof_pc1  = np.zeros(n_tr_events, dtype=np.float64)
    oof_pc2  = np.zeros(n_tr_events, dtype=np.float64)

    for fold_i, (fold_tr, fold_va) in enumerate(kf.split(X_tr_ev)):
        Xf_tr = X_tr_ev[fold_tr];  Xf_va = X_tr_ev[fold_va]
        wf_tr = w_tr_ev[fold_tr]

        def _fit_oof(y_all, label_f):
            mf = _xgbr_event({
                "max_depth": 3, "learning_rate": 0.02,
                "min_child_weight": 5, "reg_lambda": 10,
                "subsample": 0.8, "colsample_bytree": 0.8,
            })
            mf.fit(Xf_tr, y_all[fold_tr], sample_weight=wf_tr,
                   eval_set=[(Xf_va, y_all[fold_va])], verbose=False)
            return mf.predict(Xf_va)

        oof_c[fold_va]   = _fit_oof(ev_tr["c_target_event"].values,   f"C-f{fold_i}")
        oof_n[fold_va]   = _fit_oof(ev_tr["delta_n_target"].values,    f"N-f{fold_i}")
        oof_pc1[fold_va] = _fit_oof(ev_tr["pc1_target"].values,        f"PC1-f{fold_i}")
        oof_pc2[fold_va] = _fit_oof(ev_tr["pc2_target"].values,        f"PC2-f{fold_i}")
        print(f"  fold {fold_i+1}/5  done")

    c_tr_d, dn_tr_d, pc_tr_d = _pred_dict(
        ev_tr, oof_c, oof_n, oof_pc1, oof_pc2,
        np.zeros(n_tr_events) if m_pc3 else None,
    )
    df_tr_rows = _attach_hat_cols(df_tr_rows, c_tr_d, dn_tr_d, pc_tr_d)
    print(f"[OOF] Train OOF pred_log_profile: "
          f"valid={df_tr_rows['pred_log_profile'].notna().sum()} / {len(df_tr_rows)}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Row-level features + local residual (TASK 5)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 5 — Local residual model")
    print("=" * 70)

    # Identify parquet FO feature columns for row-level
    pq_ro_skip = LEAKAGE | {"event_id", "sensor", "sensor_code",
                             "distance", "log_distance", "target_log", "target_pgv",
                             "split", "track_number", "train_type_code",
                             "train_speed_kmh", "train_speed_missing"}
    pq_fo_cols = [
        c for c in df_linec.columns
        if pd.api.types.is_numeric_dtype(df_linec[c])
        and c not in pq_ro_skip
        and not c.startswith("target_")
        and "pgv" not in c.lower()
    ]

    # Build row-level feature DataFrames
    row_tr = build_row_features(df_tr_rows, event_feat_df.drop(
        columns=["split", "max_pgv_event", "c_target_event",
                 "n_event_raw", "n_event_shrunk", "delta_n_target",
                 "n_track", "track", "pc1_target", "pc2_target"]
        + [f"pc{k}_target" for k in range(1, 4)], errors="ignore"))
    row_va = build_row_features(df_va_rows, event_feat_df.drop(
        columns=["split", "max_pgv_event", "c_target_event",
                 "n_event_raw", "n_event_shrunk", "delta_n_target",
                 "n_track", "track", "pc1_target", "pc2_target"]
        + [f"pc{k}_target" for k in range(1, 4)], errors="ignore"))
    row_te = build_row_features(df_te_rows, event_feat_df.drop(
        columns=["split", "max_pgv_event", "c_target_event",
                 "n_event_raw", "n_event_shrunk", "delta_n_target",
                 "n_track", "track", "pc1_target", "pc2_target"]
        + [f"pc{k}_target" for k in range(1, 4)], errors="ignore"))

    # Build feature column list
    feat_cols_row = _row_feature_cols(row_tr, pq_fo_cols)
    (out_dir / "feature_columns_row.txt").write_text("\n".join(feat_cols_row))
    write_leakage_audit(feat_cols_row, out_dir / "leakage_audit.txt")
    # Hard assertion: crash now rather than silently train on leaky data
    assert_no_leakage(feat_cols_row, "row-level")
    assert_no_leakage(fo_valid, "event-level")
    print(f"[row_feat] Row-level feature columns: {len(feat_cols_row)}")

    # Drop rows with missing profile prediction (events with incomplete sensor set)
    row_tr = row_tr.dropna(subset=["pred_log_profile"])
    row_va = row_va.dropna(subset=["pred_log_profile"])
    row_te = row_te.dropna(subset=["pred_log_profile"])

    m_local, local_val_rmse, local_scheme = train_local_residual_model(
        row_tr, row_va, row_te, feat_cols_row,
    )

    pd.DataFrame({
        "feature": [c for c in feat_cols_row if c in row_tr.columns],
        "importance": m_local.feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(
        out_dir / "feature_importance_local_residual.csv", index=False
    )

    # Predict local residuals
    def _local_pred(row_df):
        valid = [c for c in feat_cols_row if c in row_df.columns]
        return m_local.predict(row_df[valid].values.astype(np.float32))

    row_va["local_res_hat"]  = _local_pred(row_va)
    row_te["local_res_hat"]  = _local_pred(row_te)

    row_va["pred_log_final"] = row_va["pred_log_profile"] + row_va["local_res_hat"]
    row_te["pred_log_final"] = row_te["pred_log_profile"] + row_te["local_res_hat"]

    # Monotonic versions
    row_va["pred_log_profile_mono"] = apply_monotonic(row_va, "pred_log_profile")
    row_va["pred_log_final_mono"]   = apply_monotonic(row_va, "pred_log_final")
    row_te["pred_log_profile_mono"] = apply_monotonic(row_te, "pred_log_profile")
    row_te["pred_log_final_mono"]   = apply_monotonic(row_te, "pred_log_final")

    # Save row-level features (without targets)
    feat_save_cols = feat_cols_row + ["event_id", "sensor"]
    feat_save_cols = [c for c in feat_save_cols if c in row_te.columns]
    _save_checkpoint(row_te[feat_save_cols], out_dir / "row_features.parquet", "row_features")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6 — Baselines (TASK 6)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 6 — Train baseline models")
    print("=" * 70)

    a1_result = train_baseline_a1(df_linec, split_df)

    # Load P3 baseline if available
    p3_te_pred = p3_va_pred = None
    if p3_dir is not None:
        p = p3_dir / "all_predictions.parquet"
        if p.exists():
            p3_all = pd.read_parquet(p)
            p3_all["event_id"] = p3_all["event_id"].astype(str)
            p3_te = p3_all[p3_all["split"] == "test"]
            p3_va = p3_all[p3_all["split"] == "val"]
            # Align to row_te order
            p3_te_aligned = row_te.merge(
                p3_te[["event_id", "sensor", "pred_log_p3"]],
                on=["event_id", "sensor"], how="left"
            )["pred_log_p3"].values
            p3_va_aligned = row_va.merge(
                p3_va[["event_id", "sensor", "pred_log_p3"]],
                on=["event_id", "sensor"], how="left"
            )["pred_log_p3"].values
            p3_te_pred = p3_te_aligned
            p3_va_pred = p3_va_aligned
            print(f"[baseline_P3] Loaded P3 predictions  n_test={len(p3_te):,}")

    # Load PXGBR-R2 and PXGBR ensemble top-5 from saved output directories
    pxgbr_preds = load_pxgbr_predictions(row_te)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 7 — Metrics (TASK 6 continued)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 7 — Compute metrics")
    print("=" * 70)

    # Align A1 test predictions to row_te order
    a1_te_df = df_linec[df_linec["split"] == "test"].copy()
    a1_te_df["a1_pred"] = a1_result["model"].predict(
        a1_te_df[[c for c in a1_result["feat_cols"] if c in a1_te_df.columns]]
        .fillna(0.0).values
    )
    a1_te_aligned = row_te.merge(
        a1_te_df[["event_id", "sensor", "a1_pred"]],
        on=["event_id", "sensor"], how="left"
    )["a1_pred"].values

    s_te  = row_te["sensor"].values
    e_te  = row_te["event_id"].values
    tl_te = row_te["target_log"].values

    all_metrics = []

    # FO-PhysProfile: profile only
    msk_ok_profile = np.isfinite(row_te["pred_log_profile"].values)
    if msk_ok_profile.sum() > 10:
        m_prof = compute_metrics(
            row_te["pred_log_profile"].values[msk_ok_profile],
            tl_te[msk_ok_profile],
            s_te[msk_ok_profile], e_te[msk_ok_profile],
            "FO-PhysProfile (profile-only)",
        )
        all_metrics.append(m_prof)
        print_metrics(m_prof)

    # FO-PhysProfile: profile-only monotonic
    m_prof_mono = compute_metrics(
        row_te["pred_log_profile_mono"].values[msk_ok_profile],
        tl_te[msk_ok_profile],
        s_te[msk_ok_profile], e_te[msk_ok_profile],
        "FO-PhysProfile (profile-only mono)",
    )
    all_metrics.append(m_prof_mono)
    print_metrics(m_prof_mono)

    # FO-PhysProfile: final (profile + local residual)
    msk_ok_final = np.isfinite(row_te["pred_log_final"].values)
    m_final = compute_metrics(
        row_te["pred_log_final"].values[msk_ok_final],
        tl_te[msk_ok_final],
        s_te[msk_ok_final], e_te[msk_ok_final],
        "FO-PhysProfile (final)",
    )
    all_metrics.append(m_final)
    print_metrics(m_final)

    # FO-PhysProfile: final monotonic
    m_final_mono = compute_metrics(
        row_te["pred_log_final_mono"].values[msk_ok_final],
        tl_te[msk_ok_final],
        s_te[msk_ok_final], e_te[msk_ok_final],
        "FO-PhysProfile (final mono)",
    )
    all_metrics.append(m_final_mono)
    print_metrics(m_final_mono)

    # Baseline A1
    msk_a1 = np.isfinite(a1_te_aligned)
    if msk_a1.sum() > 10:
        m_a1 = compute_metrics(a1_te_aligned[msk_a1], tl_te[msk_a1],
                                s_te[msk_a1], e_te[msk_a1], "A1_direct_tabular")
        all_metrics.append(m_a1)
        print_metrics(m_a1)

    # Baseline P3
    if p3_te_pred is not None:
        msk_p3 = np.isfinite(p3_te_pred)
        if msk_p3.sum() > 10:
            m_p3 = compute_metrics(p3_te_pred[msk_p3], tl_te[msk_p3],
                                    s_te[msk_p3], e_te[msk_p3], "P3_corrected_n")
            all_metrics.append(m_p3)
            print_metrics(m_p3)

    # Baselines PXGBR-R2 and ensemble top-5
    for bl_label, bl_pred in pxgbr_preds.items():
        if bl_pred is None:
            continue
        msk = np.isfinite(bl_pred)
        if msk.sum() > 10:
            m_bl = compute_metrics(bl_pred[msk], tl_te[msk],
                                    s_te[msk], e_te[msk], bl_label)
            all_metrics.append(m_bl)
            print_metrics(m_bl)

    # Summary table
    rows_table = []
    for m in all_metrics:
        row_t = {
            "model":     m["model"],
            "rmse_pgv":  m["rmse_pgv"],
            "rmse_log":  m["rmse_log"],
            "mae_pgv":   m["mae_pgv"],
            "r2_log":    m["r2_log"],
            "bias_pgv":  m["bias_pgv"],
            "mono_viol": m.get("mono_viol_rate", np.nan),
        }
        for s in SENSOR_ORDER:
            row_t[f"rmse_{s}"] = m["per_sensor"].get(s, {}).get("rmse_pgv", np.nan)
        rows_table.append(row_t)

    metrics_df = pd.DataFrame(rows_table)
    metrics_df = metrics_df.sort_values("rmse_pgv")
    print("\n" + "=" * 70)
    print("METRICS SUMMARY (sorted by RMSE(PGV))")
    print("=" * 70)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    metrics_df.to_csv(out_dir / "metrics_table.csv", index=False)

    # Physics diagnostics
    print("\n" + "=" * 70)
    print("PHYSICS DIAGNOSTICS")
    print("=" * 70)
    # c_hat vs c_target (val)
    c_va_true = ev_va["c_target_event"].values
    r2_c_val = float(1 - np.sum((va_c - c_va_true) ** 2) /
                     max(np.sum((c_va_true - c_va_true.mean()) ** 2), 1e-12))
    print(f"  c_hat vs c_target R² (val) = {r2_c_val:.4f}  "
          f"(>0.30 = success criterion)")
    # n_hat vs n_target (val)
    dn_va_true = ev_va["delta_n_target"].values
    r2_n_val = float(1 - np.sum((va_n - dn_va_true) ** 2) /
                     max(np.sum((dn_va_true - dn_va_true.mean()) ** 2), 1e-12))
    print(f"  n_hat vs delta_n_target R² (val) = {r2_n_val:.4f}")
    # PC1, PC2
    r2_pc1 = float(1 - np.sum((va_pc1 - ev_va["pc1_target"].values) ** 2) /
                   max(np.sum((ev_va["pc1_target"].values
                                - ev_va["pc1_target"].values.mean()) ** 2), 1e-12))
    r2_pc2 = float(1 - np.sum((va_pc2 - ev_va["pc2_target"].values) ** 2) /
                   max(np.sum((ev_va["pc2_target"].values
                                - ev_va["pc2_target"].values.mean()) ** 2), 1e-12))
    print(f"  PC1 prediction R² (val) = {r2_pc1:.4f}")
    print(f"  PC2 prediction R² (val) = {r2_pc2:.4f}")
    print(f"  PCA VE: {pca_ve.round(3).tolist()}")
    print(f"  Waveform features available: {wf_ok}")

    phys_diag = {
        "c_hat_r2_val":         r2_c_val,
        "n_hat_r2_val":         r2_n_val,
        "pc1_r2_val":           r2_pc1,
        "pc2_r2_val":           r2_pc2,
        "pca_ve":               pca_ve.tolist(),
        "pca_ve_cumsum":        float(pca_ve.cumsum()[-1]),
        "wf_features_ok":       wf_ok,
        "profile_rmse_pgv_val": va_profile_rmse_pgv,
    }
    (out_dir / "physics_diagnostics.json").write_text(json.dumps(phys_diag, indent=2))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 8 — Save predictions and plots (TASK 7)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 8 — Save predictions and plots")
    print("=" * 70)

    # Test predictions
    pred_cols = [
        "event_id", "sensor", "track_number", "distance",
        "target_log", "target_pgv",
        "pred_log_profile", "pred_log_profile_mono",
        "pred_log_final",   "pred_log_final_mono",
        "c_hat_profile", "n_hat_profile", "pc1_hat", "pc2_hat",
    ]
    pred_cols = [c for c in pred_cols if c in row_te.columns]
    row_te[pred_cols].to_parquet(out_dir / "predictions_test.parquet", index=False)

    # Event-level predictions (test)
    ev_pred_cols = [
        "event_id", "c_target_event", "n_event_shrunk", "delta_n_target",
        "pc1_target", "pc2_target",
    ]
    ev_pred_cols += [f"pc{k}_target" for k in range(3, PCA_N_COMP + 1)]
    ev_pred_cols = [c for c in ev_pred_cols if c in ev_te.columns]
    ev_te_out = ev_te[ev_pred_cols].copy()
    ev_te_out["c_hat"]    = te_c
    ev_te_out["delta_n_hat"] = te_n
    ev_te_out["pc1_hat"]  = te_pc1
    ev_te_out["pc2_hat"]  = te_pc2
    ev_te_out.to_parquet(out_dir / "event_predictions_test.parquet", index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[plots] Generating figures...")

    # 1. Measured vs predicted scatter
    ok = np.isfinite(row_te["pred_log_final"].values)
    plot_scatter(
        row_te["target_pgv"].values[ok],
        np.exp(row_te["pred_log_final"].values[ok]),
        "FO-PhysProfile (final)",
        out_dir / "measured_vs_predicted_fo_physprofile.png",
    )

    # 2. Per-sensor RMSE comparison
    metrics_for_plot = [m for m in all_metrics if m["model"] in [
        "FO-PhysProfile (profile-only)", "FO-PhysProfile (final)",
        "FO-PhysProfile (final mono)", "A1_direct_tabular", "P3_corrected_n",
    ]]
    if metrics_for_plot:
        plot_per_sensor_rmse(metrics_for_plot,
                             out_dir / "per_sensor_rmse_fo_physprofile_vs_baselines.png")

    # 3. High-PGV profiles
    ok_events = row_te.dropna(subset=["pred_log_final"])
    if len(ok_events) > 4:
        plot_high_pgv_profiles(ok_events, "pred_log_final", 16,
                               out_dir / "high_pgv_profiles_fo_physprofile.png")

    # 4. Failure profiles (largest errors)
    if msk_ok_final.sum() > 4:
        err_df = row_te.copy()
        err_df["abs_err_pgv"] = np.abs(
            np.exp(err_df["pred_log_final"]) - err_df["target_pgv"]
        )
        worst_eids = err_df.groupby("event_id")["abs_err_pgv"].max().nlargest(16).index
        fail_df = err_df[err_df["event_id"].isin(worst_eids)]
        plot_high_pgv_profiles(fail_df, "pred_log_final", 16,
                               out_dir / "failure_profiles_fo_physprofile.png")

    # 5. c_hat vs c_target
    c_te_true = ev_te["c_target_event"].values
    plot_physics_scatter(c_te_true, te_c,
                         "c_target", "c_hat",
                         f"c_hat vs c_target  (test, R²={r2_c_val:.3f} val)",
                         out_dir / "c_target_vs_c_hat.png")

    # 6. n_hat vs delta_n_target
    dn_te_true = ev_te["delta_n_target"].values
    plot_physics_scatter(dn_te_true, te_n,
                         "delta_n_target", "delta_n_hat",
                         "n_hat vs delta_n_target (test)",
                         out_dir / "n_target_vs_n_hat.png")

    # 7. PCA components
    plot_pca_components(pca, out_dir / "residual_profile_pca_components.png")

    # 8. PC scores true vs predicted (val, PC1 and PC2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (va_pci, tgt_col, label) in zip(axes, [
        (va_pc1, "pc1_target", "PC1"),
        (va_pc2, "pc2_target", "PC2"),
    ]):
        true_v = ev_va[tgt_col].values
        ax.scatter(true_v, va_pci, alpha=0.4, s=8)
        lo = min(true_v.min(), va_pci.min())
        hi = max(true_v.max(), va_pci.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        r2_v = float(1 - np.sum((va_pci - true_v) ** 2) /
                     max(np.sum((true_v - true_v.mean()) ** 2), 1e-12))
        ax.set_title(f"{label} true vs pred  R²={r2_v:.3f}")
        ax.set_xlabel(f"{label} true"); ax.set_ylabel(f"{label} pred")
    plt.suptitle("PC score prediction quality (val)")
    _save_fig(out_dir / "pc_scores_true_vs_predicted.png")

    # 9. Feature importance C top 30
    plot_feature_importance(m_c, fo_valid,
                            "Feature importance — C model (c_target)",
                            out_dir / "feature_importance_c_top30.png")

    # 10. Feature importance local residual top 30
    valid_loc = [c for c in feat_cols_row if c in row_tr.columns]
    plot_feature_importance(m_local, valid_loc,
                            "Feature importance — local residual",
                            out_dir / "feature_importance_local_residual_top30.png")

    # 11. Residual vs distance
    row_te_ok = row_te.dropna(subset=["pred_log_final"])
    row_te_ok = row_te_ok.copy()
    row_te_ok["final_residual"] = row_te_ok["target_log"] - row_te_ok["pred_log_final"]
    plot_residual_vs_distance(row_te_ok, "final_residual",
                              "Final residual vs distance (test)",
                              out_dir / "residual_vs_distance_final.png")

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    best_row = metrics_df.iloc[0]
    print(f"  Best model: {best_row['model']}")
    print(f"  RMSE(PGV) = {best_row['rmse_pgv']:.4f}  "
          f"RMSE(log) = {best_row['rmse_log']:.4f}")
    if best_row["rmse_pgv"] < 2.20:
        print("  ✓ TARGET MET: RMSE(PGV) < 2.20")
    elif best_row["rmse_pgv"] < 2.24:
        print(f"  ~ BEATS PXGBR_ens_top5 (2.2473): RMSE(PGV) = {best_row['rmse_pgv']:.4f}")
    else:
        print(f"  ✗ Does not beat PXGBR_ens_top5 (2.2473): "
              f"RMSE(PGV) = {best_row['rmse_pgv']:.4f}")
    print(f"\n  c_hat vs c_target R² (val) = {r2_c_val:.4f}")
    if r2_c_val < 0.30:
        print(f"  ✗ SUCCESS CRITERION FAILED: c_hat R² < 0.30  "
              f"(fresh targets may need more FO signal correlation)")
    else:
        print(f"  ✓ SUCCESS CRITERION MET: c_hat R² >= 0.30")

    print(f"\n  Output dir: {out_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
