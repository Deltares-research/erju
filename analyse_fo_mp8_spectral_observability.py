"""
analyse_fo_mp8_spectral_observability.py
==========================================
Band-by-band FO–MP8 vertical-velocity spectral observability analysis.

Step 0 : Patch spectral metadata (track_number, train_type, speed, split)
         using aligned_events.parquet as authoritative source.
         Saves corrected spectral to holten_spectral_targets_v002_corrected/.

Steps 1-9: FO band-RMS computation + observability analysis.

SignalProcessingTools usage
---------------------------
NOT USED for band RMS (same reasons as spectral build):
  fft()                 amplitude normalization, not power-consistent
  psd()                 Welch; bypassed by direct rfft
  one_third_octave_bands() base-2, not base-10

USED: v_eff_SBR — not needed here (analysis only uses band RMS)

Spectral normalization: numpy.fft.rfft, Parseval-consistent.
Padding correction: band_rms_corrected = band_rms * sqrt(N_total / n_valid_250hz)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

def _root():
    return Path(r"P:\11210978-erju-ai") if os.name == "nt" else Path("/p/11210978-erju-ai")

ROOT    = _root()
SPEC_V1 = ROOT / "holten_spectral_targets_v001_full_trace"
SPEC_V2 = ROOT / "holten_spectral_targets_v002_corrected"
ALIGN   = ROOT / "holten_models" / "outputs" / "spectral_fo_alignment_v001"
WV_DIR  = ROOT / "holten_waveform" / "holten_waveform_v003_ch51_20260626_094551"
OUT_DIR = ROOT / "holten_models" / "outputs" / "fo_mp8_spectral_observability_v001"
BANDS_PATH = _REPO / "spectral_definitions" / "bands.parquet"

# FO configuration
FS_WF      = 250.0        # Hz (waveform sampling rate)
N_WF       = 7500         # samples (30 s at 250 Hz)
FS_ORIG    = 1000.0       # Hz (original accelerometer sampling rate)
CH_LC      = 1194         # Line-C centre channel absolute number
CH_LO      = 1165         # first channel in stored array
IDX_LC     = CH_LC - CH_LO  # = 29
IDX_WIN_LO = max(0, IDX_LC - 5)    # 11-channel window: ±5
IDX_WIN_HI = min(51, IDX_LC + 6)   # exclusive upper index
WF_SCALE   = 1.0e6        # strain → microstrain
FLOOR_FO   = 1e-10        # microstrain, floor for log

# MP8 parameters
FLOOR_MP8  = 1e-10        # mm/s, floor for log


def section(s): print(f"\n{'='*70}\n{s}\n{'='*70}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 0  Patch spectral metadata
# ──────────────────────────────────────────────────────────────────────────────
def patch_spectral_metadata() -> Path:
    section("STEP 0: Patch spectral metadata (v001 → v002)")
    SPEC_V2.mkdir(parents=True, exist_ok=True)

    aligned = pd.read_parquet(ALIGN / "aligned_events.parquet",
                               columns=["event_id", "track_number", "train_type",
                                         "train_speed_kmh", "split"])
    aligned["event_id"] = aligned["event_id"].astype(str)

    # Patch spectral_targets.parquet
    tgt = pd.read_parquet(SPEC_V1 / "spectral_targets.parquet")
    tgt["event_id"] = tgt["event_id"].astype(str)
    # Remove incorrect metadata columns
    drop_cols = [c for c in ["track_number","train_type","train_speed_kmh","train_direction"]
                 if c in tgt.columns]
    tgt = tgt.drop(columns=drop_cols)
    tgt = tgt.merge(aligned, on="event_id", how="inner")   # inner: aligned 1697 events only
    tgt.to_parquet(SPEC_V2 / "spectral_targets.parquet", index=False)
    print(f"  spectral_targets: {len(tgt):,} rows  track dist: {dict(tgt['track_number'].value_counts().sort_index())}")

    # Patch events.parquet
    ev = pd.read_parquet(SPEC_V1 / "events.parquet")
    ev["event_id"] = ev["event_id"].astype(str)
    ev = ev.drop(columns=[c for c in ["track_number","train_type","train_speed_kmh"] if c in ev.columns])
    ev = ev.merge(aligned, on="event_id", how="inner")
    ev.to_parquet(SPEC_V2 / "events.parquet", index=False)

    # Copy bands and sensors unchanged
    import shutil
    for fname in ["bands.parquet","sensors.parquet"]:
        shutil.copy2(SPEC_V1 / fname, SPEC_V2 / fname)
    (SPEC_V2 / "build_metadata.json").write_text(json.dumps({
        "description": "Corrected version of v001: track_number/train_type/speed/split patched from aligned_events.parquet",
        "parent": "holten_spectral_targets_v001_full_trace",
        "correction": "track_number was -1 for all events in v001 due to a metadata read bug. Now taken from parquet_v002 via aligned_events.parquet.",
        "n_events": len(ev), "created": datetime.now().isoformat(timespec="seconds")
    }, indent=2), encoding="utf-8")
    print(f"  Saved to: {SPEC_V2}")
    return SPEC_V2


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1  Load all inputs
# ──────────────────────────────────────────────────────────────────────────────
def load_inputs(spec_dir: Path) -> dict:
    section("STEP 1: Loading inputs")
    aligned = pd.read_parquet(ALIGN / "aligned_events.parquet")
    aligned["event_id"] = aligned["event_id"].astype(str)

    bands = pd.read_parquet(BANDS_PATH)
    # Normalise column name: bands.parquet uses "nominal_hz"; spectral targets use "band_nominal_hz"
    if "nominal_hz" in bands.columns and "band_nominal_hz" not in bands.columns:
        bands = bands.rename(columns={"nominal_hz": "band_nominal_hz"})
    primary = bands[bands["fully_inside"]].copy()
    print(f"  Aligned events: {len(aligned):,}  |  Primary bands: {len(primary)}")

    # MP8 spectral targets for aligned events only
    mp8 = pd.read_parquet(spec_dir / "spectral_targets.parquet",
                           columns=["event_id","sensor_id","band_index","band_nominal_hz",
                                     "velocity_band_rms_mms","track_number","split",
                                     "train_type","train_speed_kmh"])
    mp8 = mp8[(mp8["sensor_id"]=="MP8") & (mp8["event_id"].isin(aligned["event_id"]))]
    print(f"  MP8 spectral rows: {len(mp8):,}  events: {mp8['event_id'].nunique()}")

    # Waveform event index
    ei = pd.read_parquet(WV_DIR / "event_index.parquet",
                          columns=["event_id","waveform_row_idx","n_samples_original",
                                   "was_padded","was_cropped"])
    ei["event_id"] = ei["event_id"].astype(str)
    ei = ei.merge(aligned[["event_id","track_number","train_type","train_speed_kmh","split"]],
                   on="event_id", how="inner")

    # Valid sample count at 250 Hz
    ei["n_valid_250hz"] = (ei["n_samples_original"] * FS_WF / FS_ORIG).round().astype(int).clip(1, N_WF)
    ei["duration_valid_s"] = ei["n_valid_250hz"] / FS_WF

    print(f"  Waveform events: {len(ei):,}")
    print(f"  Padded: {ei['was_padded'].sum()}  Cropped: {ei['was_cropped'].sum()}")
    print(f"  Valid duration: mean={ei['duration_valid_s'].mean():.1f}s  "
          f"min={ei['duration_valid_s'].min():.1f}s  max={ei['duration_valid_s'].max():.1f}s")

    wf = np.load(WV_DIR / "waveforms.npy", mmap_mode="r")
    print(f"  Waveform array: {wf.shape}  dtype={wf.dtype}")
    return {"aligned": aligned, "bands": bands, "primary": primary,
            "mp8": mp8, "ei": ei, "wf": wf}


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2  Compute FO band RMS
# ──────────────────────────────────────────────────────────────────────────────
def compute_fo_bands(data: dict, out: Path) -> pd.DataFrame:
    section("STEP 2: Computing FO band RMS per event")
    ei    = data["ei"]
    wf    = data["wf"]
    bands = data["bands"]

    fl_arr  = bands["fl_hz"].values
    fu_arr  = bands["fu_hz"].values
    k_arr   = bands["band_index_k"].values
    nom_arr = bands["band_nominal_hz"].values
    n_bands = len(bands)

    freqs = np.fft.rfftfreq(N_WF, d=1.0 / FS_WF)
    df_hz = FS_WF / N_WF

    # Pre-compute band masks
    band_masks = [(freqs >= fl_arr[bi]) & (freqs < fu_arr[bi]) for bi in range(n_bands)]

    aggregations = ["lc_ch1", "lc_win11", "all51"]  # 3 spatial aggregations
    # Multi-channel reductions: mean/median/max energy across channels
    mc_reductions = ["mean", "median", "max"]

    rows = []
    dur_rows = []
    n_total = len(ei)
    t0 = time.time()

    for idx, ev_row in ei.iterrows():
        if (idx+1) % 200 == 0 or idx+1 == n_total:
            print(f"  [{idx+1:4d}/{n_total}]  {time.time()-t0:.0f}s")

        eid     = ev_row["event_id"]
        row_idx = int(ev_row["waveform_row_idx"])
        n_valid = int(ev_row["n_valid_250hz"])
        corr    = np.sqrt(N_WF / n_valid)  # padding correction factor

        # Load all 51 channels for this event
        block = wf[row_idx].astype(np.float64) * WF_SCALE  # (51, 7500)

        # Duration/padding information
        dur_rows.append({
            "event_id":          eid,
            "waveform_row_idx":  row_idx,
            "n_samples_orig_1khz": int(ev_row["n_samples_original"]),
            "n_valid_250hz":     n_valid,
            "duration_valid_s":  float(ev_row["duration_valid_s"]),
            "waveform_duration_s": N_WF / FS_WF,
            "was_padded":        bool(ev_row["was_padded"]),
            "was_cropped":       bool(ev_row["was_cropped"]),
            "padding_correction_factor": float(corr),
        })

        # Compute band RMS for each aggregation
        def _band_rms_ch(sig_ch: np.ndarray) -> np.ndarray:
            """One-sided Parseval-consistent band RMS for a single channel."""
            X = np.fft.rfft(sig_ch)
            psd = np.abs(X)**2 / (N_WF * FS_WF)
            psd[1:-1] *= 2.0
            return np.array([float(np.sqrt(max(np.sum(psd[band_masks[bi]]) * df_hz, 0)))
                             for bi in range(n_bands)])

        # Single channel: Line-C centre (index 29)
        rms_lc1 = _band_rms_ch(block[IDX_LC]) * corr

        # 11-channel window
        blk_win11 = block[IDX_WIN_LO:IDX_WIN_HI]  # (11, 7500)
        rms_per_ch_win11 = np.array([_band_rms_ch(blk_win11[c]) for c in range(blk_win11.shape[0])])
        rms_win11_mean   = rms_per_ch_win11.mean(axis=0) * corr
        rms_win11_median = np.median(rms_per_ch_win11, axis=0) * corr
        rms_win11_max    = rms_per_ch_win11.max(axis=0) * corr

        # All 51 channels
        rms_per_ch_all = np.array([_band_rms_ch(block[c]) for c in range(block.shape[0])])
        rms_all_mean   = rms_per_ch_all.mean(axis=0) * corr
        rms_all_median = np.median(rms_per_ch_all, axis=0) * corr
        rms_all_max    = rms_per_ch_all.max(axis=0) * corr

        # Parseval check on Line-C channel
        sig_lc = block[IDX_LC]
        ms_full = float(np.mean(sig_lc**2))
        ms_sum_bands = float(np.sum((rms_lc1 / corr)**2))
        parseval_err = abs(ms_sum_bands - ms_full) / (ms_full + 1e-300)

        for bi in range(n_bands):
            rows.append({
                "event_id":         eid,
                "band_index":       int(k_arr[bi]),
                "band_nominal_hz":  float(nom_arr[bi]),
                # Single channel (padding-corrected)
                "fo_rms_lc1_ue":        float(rms_lc1[bi]),          # microstrain
                # 11-channel window
                "fo_rms_win11_mean":    float(rms_win11_mean[bi]),
                "fo_rms_win11_median":  float(rms_win11_median[bi]),
                "fo_rms_win11_max":     float(rms_win11_max[bi]),
                # All 51 channels
                "fo_rms_all51_mean":    float(rms_all_mean[bi]),
                "fo_rms_all51_median":  float(rms_all_median[bi]),
                "fo_rms_all51_max":     float(rms_all_max[bi]),
                # Metadata
                "parseval_err_lc1":     float(parseval_err),
                "was_padded":           bool(ev_row["was_padded"]),
                "padding_corr":         float(corr),
            })

    fo_df  = pd.DataFrame(rows)
    dur_df = pd.DataFrame(dur_rows)
    dur_df.to_parquet(out / "duration_and_padding_checks.parquet", index=False)

    print(f"\n  FO band rows: {len(fo_df):,}")
    print(f"  Parseval error: max={fo_df['parseval_err_lc1'].max():.2e}  "
          f"mean={fo_df['parseval_err_lc1'].mean():.2e}")
    print(f"  Padding correction factor: median={fo_df['padding_corr'].median():.4f}  "
          f"max={fo_df['padding_corr'].max():.4f}")
    return fo_df, dur_df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3  Build analysis table
# ──────────────────────────────────────────────────────────────────────────────
def build_analysis_table(data: dict, fo_df: pd.DataFrame) -> pd.DataFrame:
    section("STEP 3: Building combined analysis table")
    mp8 = data["mp8"].copy()
    aligned = data["aligned"][["event_id","track_number","train_type","train_speed_kmh","split"]]

    # Merge FO with MP8 on event_id × band_index
    combo = fo_df.merge(mp8[["event_id","band_index","velocity_band_rms_mms",
                               "track_number","split","train_type","train_speed_kmh"]],
                         on=["event_id","band_index"], how="inner")

    # Log-transform (floor to prevent log(0))
    fo_cols = ["fo_rms_lc1_ue","fo_rms_win11_mean","fo_rms_win11_median","fo_rms_win11_max",
               "fo_rms_all51_mean","fo_rms_all51_median","fo_rms_all51_max"]
    for col in fo_cols:
        combo[f"log_{col}"] = np.log(combo[col].clip(lower=FLOOR_FO))
    combo["log_mp8_rms"] = np.log(combo["velocity_band_rms_mms"].clip(lower=FLOOR_MP8))

    # Train type family (simplified)
    from src.db.parquet.parquet_v3_utils import assign_train_type_family
    combo["train_family"] = combo["train_type"].apply(assign_train_type_family)

    print(f"  Combined rows: {len(combo):,}  events: {combo['event_id'].nunique()}")
    print(f"  Track dist: {dict(combo['track_number'].value_counts().sort_index())}")
    print(f"  Split dist: {dict(combo['split'].value_counts())}")
    return combo


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4  Band-wise correlation analysis
# ──────────────────────────────────────────────────────────────────────────────
def _bootstrap_ci(x, y, n_boot=1000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            r = pearsonr(x[idx], y[idx])[0]
        except Exception:
            r = float("nan")
        boots.append(r)
    boots = np.array(boots)
    return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def bandwise_correlations(combo: pd.DataFrame, bands: pd.DataFrame,
                           fo_col: str = "log_fo_rms_lc1_ue") -> pd.DataFrame:
    """Compute per-band Pearson/Spearman correlations: FO vs MP8."""
    rows = []
    band_nom = sorted(combo["band_nominal_hz"].unique())
    for nom in band_nom:
        band_info = bands[bands["band_nominal_hz"]==nom].iloc[0]
        for track_label, track_num in [("all", None), ("1", 1), ("2", 2)]:
            sub = combo[combo["band_nominal_hz"]==nom]
            if track_num is not None:
                sub = sub[sub["track_number"]==track_num]
            x = sub[fo_col].values
            y = sub["log_mp8_rms"].values
            m = np.isfinite(x) & np.isfinite(y)
            n = int(m.sum())
            if n < 10:
                continue
            r_p, p_p = pearsonr(x[m], y[m])
            r_s, p_s = spearmanr(x[m], y[m])
            ci_lo, ci_hi = _bootstrap_ci(x[m], y[m])
            rows.append({
                "band_nominal_hz":  float(nom),
                "band_index":       int(band_info["band_index_k"]),
                "fully_inside":     bool(band_info["fully_inside"]),
                "track":            track_label,
                "n":                n,
                "pearson_r":        round(float(r_p),4),
                "pearson_p":        round(float(p_p),6),
                "spearman_r":       round(float(r_s),4),
                "spearman_p":       round(float(p_s),6),
                "boot_ci95_lo":     round(ci_lo,4),
                "boot_ci95_hi":     round(ci_hi,4),
                "fo_representation": fo_col.replace("log_",""),
            })
    return pd.DataFrame(rows)


def run_all_bandwise(combo: pd.DataFrame, bands: pd.DataFrame, out: Path) -> pd.DataFrame:
    section("STEP 4: Band-wise correlations (all FO representations)")
    fo_cols = ["log_fo_rms_lc1_ue", "log_fo_rms_win11_mean", "log_fo_rms_win11_max",
               "log_fo_rms_all51_mean", "log_fo_rms_all51_max"]
    all_corr = []
    for col in fo_cols:
        df = bandwise_correlations(combo, bands, fo_col=col)
        all_corr.append(df)
        best_all = df[(df["track"]=="all") & df["fully_inside"]]["pearson_r"].abs().max()
        print(f"  {col.replace('log_',''):<28} best r={best_all:.4f}")
    corr_df = pd.concat(all_corr, ignore_index=True)
    corr_df.to_parquet(out / "bandwise_correlations.parquet", index=False)
    print(f"  Saved {len(corr_df):,} rows")
    return corr_df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5  Cross-band correlation matrix
# ──────────────────────────────────────────────────────────────────────────────
def cross_band_matrix(combo: pd.DataFrame, bands: pd.DataFrame,
                       fo_col: str, out: Path) -> pd.DataFrame:
    section("STEP 5: Cross-band FO × MP8 correlation matrix")
    primary = bands[bands["fully_inside"]]["band_nominal_hz"].values
    # Pivot: each row = one event, columns = FO band RMS for each nominal freq
    combo_p = combo[combo["band_nominal_hz"].isin(primary)].copy()
    # FO pivot
    fo_piv = combo_p.pivot_table(index="event_id", columns="band_nominal_hz",
                                   values=fo_col, aggfunc="first")
    # MP8 pivot
    mp8_piv = combo_p.pivot_table(index="event_id", columns="band_nominal_hz",
                                    values="log_mp8_rms", aggfunc="first")
    common_ev = fo_piv.index.intersection(mp8_piv.index)
    fo_piv  = fo_piv.loc[common_ev].dropna(axis=1)
    mp8_piv = mp8_piv.loc[common_ev].dropna(axis=1)

    fo_bands  = fo_piv.columns.tolist()
    mp8_bands = mp8_piv.columns.tolist()
    matrix = np.zeros((len(fo_bands), len(mp8_bands)))
    for i, fb in enumerate(fo_bands):
        for j, mb in enumerate(mp8_bands):
            x = fo_piv[fb].values
            y = mp8_piv[mb].values
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 10:
                matrix[i, j] = pearsonr(x[m], y[m])[0]

    df_matrix = pd.DataFrame(matrix, index=fo_bands, columns=mp8_bands)
    df_matrix.index.name = "fo_band_hz"; df_matrix.columns.name = "mp8_band_hz"
    df_matrix.reset_index().to_parquet(out / "cross_band_correlations.parquet", index=False)
    print(f"  Matrix: {len(fo_bands)} FO bands × {len(mp8_bands)} MP8 bands")
    print(f"  Max |r| on diagonal: {abs(np.diag(matrix)).max():.3f}")
    print(f"  Max |r| off-diagonal: {abs(matrix - np.diag(np.diag(matrix))).max():.3f}")
    return df_matrix


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6  Train-type correlations
# ──────────────────────────────────────────────────────────────────────────────
def train_type_correlations(combo: pd.DataFrame, bands: pd.DataFrame,
                              fo_col: str, out: Path) -> pd.DataFrame:
    section("STEP 6: Train-type correlations for top bands")
    primary = bands[bands["fully_inside"]]["band_nominal_hz"].values
    # Find top 5 primary bands by pooled Pearson r
    pool = (combo[combo["band_nominal_hz"].isin(primary)]
            .groupby("band_nominal_hz")
            .apply(lambda g: pearsonr(g[fo_col].values, g["log_mp8_rms"].values)[0]
                   if g[fo_col].notna().sum() >= 10 else 0))
    top_bands = pool.abs().nlargest(5).index.tolist()
    print(f"  Top 5 bands: {top_bands}")

    MIN_N = 20
    rows = []
    families = ["ICM","ICR","SNG","SPR","GO","DDZ","Locomotive","Other"]
    for nom in top_bands:
        sub = combo[combo["band_nominal_hz"]==nom]
        for fam in families:
            fs = sub[sub["train_family"]==fam]
            x = fs[fo_col].values; y = fs["log_mp8_rms"].values
            m = np.isfinite(x) & np.isfinite(y)
            n = int(m.sum())
            rows.append({
                "band_nominal_hz": float(nom), "train_family": fam, "n": n,
                "pearson_r": float(pearsonr(x[m],y[m])[0]) if n>=MIN_N else float("nan"),
                "spearman_r": float(spearmanr(x[m],y[m])[0]) if n>=MIN_N else float("nan"),
                "sufficient_n": n>=MIN_N,
            })

    tt_df = pd.DataFrame(rows)
    tt_df.to_parquet(out / "train_type_correlations.parquet", index=False)
    tt_valid = tt_df[tt_df["sufficient_n"]]
    print(tt_valid[["band_nominal_hz","train_family","n","pearson_r"]].to_string(index=False))
    return tt_df, top_bands


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7  Metadata confounding
# ──────────────────────────────────────────────────────────────────────────────
def confounding_check(combo: pd.DataFrame, bands: pd.DataFrame,
                       fo_col: str, top_bands: list) -> pd.DataFrame:
    section("STEP 7: Metadata confounding check")
    from sklearn.linear_model import LinearRegression
    rows = []
    for nom in top_bands:
        sub = combo[combo["band_nominal_hz"]==nom].copy().dropna(subset=[fo_col,"log_mp8_rms"])
        x = sub[fo_col].values; y = sub["log_mp8_rms"].values
        m = np.isfinite(x) & np.isfinite(y)
        sub = sub[m]; x = x[m]; y = y[m]
        n = len(sub)
        if n < 20:
            continue
        r_raw = float(pearsonr(x, y)[0])
        # Partial correlation: residualise both x and y on metadata
        meta_cols = []
        if "track_number" in sub.columns and sub["track_number"].nunique()>1:
            meta_cols.append("track_number")
        if "train_speed_kmh" in sub.columns:
            sub["speed_finite"] = sub["train_speed_kmh"].fillna(sub["train_speed_kmh"].median())
            meta_cols.append("speed_finite")
        # train type dummies
        tt_dummies = pd.get_dummies(sub["train_family"], prefix="tt", drop_first=True)
        sub = pd.concat([sub, tt_dummies], axis=1)
        meta_cols += list(tt_dummies.columns)
        if meta_cols:
            M = sub[meta_cols].fillna(0).values
            reg = LinearRegression().fit(M, x)
            x_resid = x - reg.predict(M)
            reg2 = LinearRegression().fit(M, y)
            y_resid = y - reg2.predict(M)
            r_partial = float(pearsonr(x_resid, y_resid)[0])
        else:
            r_partial = r_raw
        rows.append({"band_nominal_hz": float(nom), "n": n,
                      "pearson_r_raw": round(r_raw,4),
                      "pearson_r_partial": round(r_partial,4),
                      "drop_in_r": round(r_raw - r_partial, 4)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 8  Plots
# ──────────────────────────────────────────────────────────────────────────────
def make_plots(corr_df: pd.DataFrame, matrix_df: pd.DataFrame,
               combo: pd.DataFrame, bands: pd.DataFrame, out: Path) -> None:
    section("STEP 8: Generating plots")
    (out/"plots").mkdir(exist_ok=True)
    primary = bands[bands["fully_inside"]]["band_nominal_hz"].values
    COLORS = {1:"tab:blue", 2:"tab:red", "all":"tab:gray"}

    def _save(fig, name):
        try: fig.savefig(out/"plots"/name, dpi=140, bbox_inches="tight")
        except Exception as e: print(f"  [WARN] {name}: {e}")
        finally: plt.close(fig)

    # ── 1. Pearson r vs frequency (lc1 + win11_mean + all51_mean, track-pooled) ──
    fig, ax = plt.subplots(figsize=(10, 4))
    for fo_rep, col, lw in [("fo_rms_lc1_ue","#e41a1c",2.0),
                              ("fo_rms_win11_mean","#ff7f00",1.5),
                              ("fo_rms_all51_mean","#377eb8",1.5)]:
        sub = corr_df[(corr_df["fo_representation"]==fo_rep) & (corr_df["track"]=="all")]
        sub = sub.sort_values("band_nominal_hz")
        ax.semilogx(sub["band_nominal_hz"], sub["pearson_r"], "o-", lw=lw,
                    ms=5, color=col, label=fo_rep.replace("fo_rms_",""))
        # CI ribbon for lc1
        if fo_rep == "fo_rms_lc1_ue":
            ax.fill_between(sub["band_nominal_hz"], sub["boot_ci95_lo"], sub["boot_ci95_hi"],
                             alpha=0.15, color=col)
    ax.axvline(1, color="gray", lw=0.7, linestyle=":"); ax.axvline(100, color="gray", lw=0.7, linestyle=":")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("Nominal frequency (Hz)"); ax.set_ylabel("Pearson r")
    ax.set_title("FO → MP8 Pearson correlation by frequency (shaded=95% CI, single-ch)")
    ax.set_xticks([1,2,4,8,16,31.5,63,100]); ax.set_xticklabels(["1","2","4","8","16","31.5","63","100"],fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.25)
    _save(fig, "01_pearson_r_vs_frequency.png")

    # ── 2. Spearman r vs frequency ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    sub = corr_df[(corr_df["fo_representation"]=="fo_rms_lc1_ue") & (corr_df["track"]=="all")].sort_values("band_nominal_hz")
    ax.semilogx(sub["band_nominal_hz"], sub["spearman_r"], "s-", color="#377eb8", ms=5, lw=1.5, label="Spearman")
    ax.semilogx(sub["band_nominal_hz"], sub["pearson_r"], "o--", color="#e41a1c", ms=5, lw=1.5, label="Pearson")
    ax.axhline(0, color="k", lw=0.7); ax.set_xlabel("Hz"); ax.set_ylabel("r")
    ax.set_title("Pearson vs Spearman correlation (single Line-C channel)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.25)
    ax.set_xticks([1,2,4,8,16,31.5,63,100]); ax.set_xticklabels(["1","2","4","8","16","31.5","63","100"],fontsize=8)
    _save(fig, "02_spearman_vs_pearson.png")

    # ── 3. Cross-band heatmap ─────────────────────────────────────────────────
    if isinstance(matrix_df, pd.DataFrame) and len(matrix_df):
        matrix_df2 = matrix_df.set_index("fo_band_hz") if "fo_band_hz" in matrix_df.columns else matrix_df
        fig, ax = plt.subplots(figsize=(9, 8))
        mat = matrix_df2.values
        im = ax.imshow(mat, aspect="auto", vmin=-0.5, vmax=0.5, cmap="RdBu_r")
        ax.set_xticks(range(mat.shape[1])); ax.set_yticks(range(mat.shape[0]))
        ax.set_xticklabels([f"{c:.0f}" for c in matrix_df2.columns], rotation=90, fontsize=7)
        ax.set_yticklabels([f"{r:.0f}" for r in matrix_df2.index], fontsize=7)
        ax.set_xlabel("MP8 band (Hz)"); ax.set_ylabel("FO band (Hz)")
        ax.set_title("Cross-band Pearson r: FO band × MP8 band (single Line-C channel)")
        plt.colorbar(im, ax=ax, label="r")
        _save(fig, "03_cross_band_heatmap.png")

    # ── 4. Local vs global FO ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for fo_rep, col, label in [
        ("fo_rms_lc1_ue","#e41a1c","Single channel (ch 1194)"),
        ("fo_rms_win11_mean","#ff7f00","11-ch window mean"),
        ("fo_rms_win11_max","#d95f02","11-ch window max"),
        ("fo_rms_all51_mean","#377eb8","51-ch mean"),
        ("fo_rms_all51_max","#1b7837","51-ch max"),
    ]:
        sub = corr_df[(corr_df["fo_representation"]==fo_rep) & (corr_df["track"]=="all")].sort_values("band_nominal_hz")
        prim = sub[sub["fully_inside"]]
        ax.semilogx(prim["band_nominal_hz"], prim["pearson_r"], "o-", ms=4, lw=1.5, color=col, label=label)
    ax.axhline(0, color="k", lw=0.7); ax.set_xlabel("Hz"); ax.set_ylabel("Pearson r")
    ax.set_title("FO spatial aggregation comparison (primary bands only)")
    ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.25)
    ax.set_xticks([1.25,2.5,5,10,20,40,80]); ax.set_xticklabels(["1.25","2.5","5","10","20","40","80"],fontsize=8)
    _save(fig, "04_local_vs_global_fo.png")

    # ── 5. Track 1 vs Track 2 ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for track, col in [(1,"tab:blue"),(2,"tab:red"),("all","tab:gray")]:
        sub = corr_df[(corr_df["fo_representation"]=="fo_rms_lc1_ue") &
                       (corr_df["track"]==track)].sort_values("band_nominal_hz")
        prim = sub[sub["fully_inside"]]
        ax.semilogx(prim["band_nominal_hz"], prim["pearson_r"], "o-", ms=4, lw=1.5, color=col, label=f"Track {track}")
    ax.axhline(0, color="k", lw=0.7); ax.set_xlabel("Hz"); ax.set_ylabel("Pearson r")
    ax.set_title("Track 1 vs Track 2 FO–MP8 Pearson correlation")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.25)
    ax.set_xticks([1.25,2.5,5,10,20,40,80]); ax.set_xticklabels(["1.25","2.5","5","10","20","40","80"],fontsize=8)
    _save(fig, "05_track1_vs_track2.png")

    # ── 6. ICR vs non-ICR for top 5 bands ────────────────────────────────────
    top5 = (corr_df[(corr_df["track"]=="all") & (corr_df["fully_inside"]) &
                     (corr_df["fo_representation"]=="fo_rms_lc1_ue")]
            .set_index("band_nominal_hz")["pearson_r"].abs().nlargest(5).index.tolist())
    fig, ax = plt.subplots(figsize=(8, 4))
    icr  = combo[combo["train_family"]=="ICR"]
    nicr = combo[combo["train_family"]!="ICR"]
    xs = range(len(top5))
    for bands_x, color, label in [(icr,"#e41a1c","ICR"),(nicr,"#377eb8","non-ICR")]:
        rs = []
        for nom in top5:
            sub = bands_x[bands_x["band_nominal_hz"]==nom]
            x = sub["log_fo_rms_lc1_ue"].values; y = sub["log_mp8_rms"].values
            m = np.isfinite(x) & np.isfinite(y)
            rs.append(float(pearsonr(x[m],y[m])[0]) if m.sum()>=10 else float("nan"))
        ax.bar([xi + (0.2 if color=="#e41a1c" else -0.2) for xi in xs], rs,
               0.35, color=color, alpha=0.8, label=f"{label} (n={len(bands_x['event_id'].unique())})")
    ax.set_xticks(list(xs)); ax.set_xticklabels([f"{f:.1f}Hz" for f in top5])
    ax.axhline(0,color='k',lw=0.7); ax.set_ylabel("Pearson r"); ax.legend(fontsize=8)
    ax.set_title("ICR vs non-ICR: top 5 primary bands (single channel)")
    _save(fig, "06_icr_vs_nonicr_top_bands.png")

    # ── 7. Representative spectra ─────────────────────────────────────────────
    # Pick 1 low / 1 high PGV event to show FO + MP8 spectra together
    combo_lc = combo[combo["fo_representation"]=="fo_rms_lc1_ue"] if "fo_representation" in combo.columns else combo
    prim_combo = combo[combo["band_nominal_hz"].isin(primary)].copy()
    ev_pgv = prim_combo.groupby("event_id")["velocity_band_rms_mms"].sum().reset_index()
    ev_pgv.columns = ["event_id","total_mp8_rms"]
    sample_eids = [ev_pgv.nsmallest(1,"total_mp8_rms")["event_id"].iloc[0],
                   ev_pgv.nlargest(1,"total_mp8_rms")["event_id"].iloc[0]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax2, eid, label in zip(axes, sample_eids, ["Low vibration","High vibration"]):
        sub = prim_combo[prim_combo["event_id"]==eid].sort_values("band_nominal_hz")
        ax2_right = ax2.twinx()
        ax2.semilogx(sub["band_nominal_hz"], sub["fo_rms_lc1_ue"], "o-", color="#e41a1c", ms=5, label="FO (µstr)")
        ax2_right.semilogx(sub["band_nominal_hz"], sub["velocity_band_rms_mms"], "s--", color="#377eb8", ms=5, label="MP8 (mm/s)")
        ax2.set_xlabel("Hz"); ax2.set_ylabel("FO band RMS (µstr)", color="#e41a1c")
        ax2_right.set_ylabel("MP8 band RMS (mm/s)", color="#377eb8")
        ax2.set_title(f"{label}: {eid[:15]}")
        ax2.set_xticks([1.25,2.5,5,10,20,40,80]); ax2.set_xticklabels(["1.25","2.5","5","10","20","40","80"],fontsize=7)
    _save(fig, "07_representative_spectra.png")
    print("  7 plots saved")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 9  Summary and report
# ──────────────────────────────────────────────────────────────────────────────
def write_summary(corr_df: pd.DataFrame, bands: pd.DataFrame,
                   confounder_df: pd.DataFrame, dur_df: pd.DataFrame,
                   out: Path) -> None:
    section("STEP 9: Summary")
    primary_corr = corr_df[(corr_df["track"]=="all") & (corr_df["fully_inside"]) &
                            (corr_df["fo_representation"]=="fo_rms_lc1_ue")].copy()
    primary_corr = primary_corr.sort_values("pearson_r", ascending=False, key=abs)

    print("\n  Top 10 primary bands by |Pearson r| (single channel, pooled tracks):")
    print(f"  {'Hz':>7} {'Pearson r':>10} {'Spearman r':>11} {'CI95':>18} {'n':>5}")
    for _, row in primary_corr.head(10).iterrows():
        print(f"  {row['band_nominal_hz']:>7.1f} {row['pearson_r']:>10.4f} {row['spearman_r']:>11.4f} "
              f"  [{row['boot_ci95_lo']:.3f},{row['boot_ci95_hi']:.3f}]  {row['n']:>5}")

    best_band = primary_corr.iloc[0]["band_nominal_hz"]
    best_r    = primary_corr.iloc[0]["pearson_r"]

    # Spatial comparison at best band
    best_reps = corr_df[(corr_df["band_nominal_hz"]==best_band) & (corr_df["track"]=="all")]
    print(f"\n  Spatial comparison at {best_band:.1f} Hz:")
    for _, row in best_reps.iterrows():
        print(f"    {row['fo_representation']:<30} r={row['pearson_r']:.4f}")

    # Padding effect summary
    n_padded = dur_df["was_padded"].sum()
    dur_range = f"{dur_df['duration_valid_s'].min():.1f}–{dur_df['duration_valid_s'].max():.1f}s"
    print(f"\n  Padding: {n_padded}/{len(dur_df)} events padded, valid duration {dur_range}")

    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_aligned_events": int(corr_df[(corr_df["track"]=="all")&(corr_df["fully_inside"])].iloc[0]["n"]),
        "top_3_primary_bands_hz": primary_corr.head(3)["band_nominal_hz"].tolist(),
        "best_band_hz": float(best_band),
        "best_pearson_r": float(best_r),
        "best_pearson_r2": float(best_r**2),
        "answers": {
            "1_most_observable_bands": f"See bandwise_correlations.parquet. Best: {best_band:.1f}Hz (r={best_r:.3f})",
            "2_same_or_cross_frequency": "To be read from cross_band_correlations.parquet heatmap",
            "3_local_vs_global": f"See 04_local_vs_global_fo.png",
            "4_after_confounding": f"See confounder table in summary",
            "5_icr_vs_others": "See 06_icr_vs_nonicr_top_bands.png and train_type_correlations.parquet",
            "6_strong_enough_for_modelling": f"Best r²={best_r**2:.3f} ({100*best_r**2:.0f}% variance explained at best band)",
            "7_target_bands": primary_corr.head(7)["band_nominal_hz"].tolist(),
        },
    }
    if len(confounder_df):
        summary["confounding"] = confounder_df[["band_nominal_hz","pearson_r_raw","pearson_r_partial"]].to_dict("records")

    (out / "observability_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n  observability_summary.json saved")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Started: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Output: {OUT_DIR}")

    # Step 0: patch metadata
    spec_dir = patch_spectral_metadata()

    # Steps 1-3: load and compute
    data   = load_inputs(spec_dir)
    fo_df, dur_df = compute_fo_bands(data, OUT_DIR)
    combo  = build_analysis_table(data, fo_df)
    combo.to_parquet(OUT_DIR / "_combined_analysis_table.parquet", index=False)

    # Step 4: bandwise correlations
    corr_df = run_all_bandwise(combo, data["bands"], OUT_DIR)

    # Step 5: cross-band matrix
    matrix_df = cross_band_matrix(combo, data["bands"], "log_fo_rms_lc1_ue", OUT_DIR)

    # Step 6: train-type
    tt_df, top_bands = train_type_correlations(combo, data["bands"], "log_fo_rms_lc1_ue", OUT_DIR)

    # Step 7: confounding
    conf_df = confounding_check(combo, data["bands"], "log_fo_rms_lc1_ue", top_bands)

    # Step 8: plots
    make_plots(corr_df, matrix_df, combo, data["bands"], OUT_DIR)

    # Step 9: summary
    write_summary(corr_df, data["bands"], conf_df, dur_df, OUT_DIR)

    print(f"\nFinished: {datetime.now().isoformat(timespec='seconds')}")
    print(f"All outputs in: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
