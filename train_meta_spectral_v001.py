"""
train_meta_spectral_v001.py
===========================
Metadata-only baseline: predict the 19 primary one-third-octave MP8 band
levels (velocity_band_level_db, dB re 1 nm/s) from train metadata alone.

Features
--------
  train_family   physics-based 8-class grouping via assign_train_type_family()
                 (GO, ICR, ICM, SNG, SPR, DDZ, Locomotive, Other)
  track_number   binary (1 / 2)
  train_speed_kmh  continuous

Note: train_direction is absent from aligned_events (all NaN) and is excluded.

Target
------
  velocity_band_level_db = 20*log10(rms_mms / 1e-6)
  for 19 primary IEC 61260 bands (fully_inside=True), nominal 1.25–80 Hz.
  Boundary bands 1 Hz and 100 Hz are excluded.

Model
-----
  Ridge regression (sklearn, multi-output native):
    OHE (drop='first') for train_family + track_number
    StandardScaler for train_speed_kmh
  Alpha grid-searched on validation macro-RMSE.
  Preprocessing fit on training set only — no leakage.

Split
-----
  Taken from aligned_events.parquet (authoritative):
    train=1103  val=254  test=340
  Final model fit on training set; test set evaluated once at the end.

Seed: 42
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────────

def _root():
    return Path(r"P:\11210978-erju-ai") if os.name == "nt" else Path("/p/11210978-erju-ai")

ROOT        = _root()
SPEC_V2     = ROOT / "holten_spectral_targets_v002_corrected"
ALIGN       = ROOT / "holten_models" / "outputs" / "spectral_fo_alignment_v001"
BANDS_PATH  = _REPO / "spectral_definitions" / "bands.parquet"
MODELS_ROOT = ROOT / "holten_models" / "outputs"

# ── Constants ─────────────────────────────────────────────────────────────────

SENSOR_ID   = "MP8"
RANDOM_SEED = 42
V_REF       = 1e-6   # mm/s reference for dB (= 1 nm/s)

# Categorical features after family mapping; no train_direction (all NaN in data)
CAT_COLS = ["train_family", "track_number"]
NUM_COLS = ["train_speed_kmh"]

# Ridge alpha candidates
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4]


# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def db_to_rms_mms(db: np.ndarray) -> np.ndarray:
    """dB (re 1 nm/s = 1e-6 mm/s) → RMS mm/s."""
    return V_REF * np.power(10.0, np.asarray(db) / 20.0)


def total_rms_from_db(band_db: np.ndarray) -> np.ndarray:
    """Sum-of-squares total RMS (mm/s) from per-band dB array (n_events, n_bands)."""
    return np.sqrt(np.sum(db_to_rms_mms(band_db) ** 2, axis=1))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(pearsonr(a, b)[0])


# ── Step 1: Load and prepare data ────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, list[float], list[str]]:
    section("STEP 1: Loading data")

    from src.db.parquet.parquet_v3_utils import assign_train_type_family

    # Authoritative metadata + split (train_direction absent → excluded)
    aligned = pd.read_parquet(
        ALIGN / "aligned_events.parquet",
        columns=["event_id", "train_type", "train_speed_kmh", "track_number", "split"],
    )
    aligned["event_id"] = aligned["event_id"].astype(str)
    aligned["track_number"] = aligned["track_number"].astype(str)   # OHE wants strings
    aligned["train_family"] = aligned["train_type"].apply(assign_train_type_family)

    print(f"  Events: {len(aligned):,}")
    print(f"  Split : {dict(aligned['split'].value_counts().sort_index())}")
    print(f"  Families: {dict(aligned['train_family'].value_counts().sort_index())}")
    print(f"  Tracks  : {dict(aligned['track_number'].value_counts().sort_index())}")
    print(f"  Speed   : min={aligned['train_speed_kmh'].min():.0f}  "
          f"max={aligned['train_speed_kmh'].max():.0f}  "
          f"median={aligned['train_speed_kmh'].median():.0f}  "
          f"std={aligned['train_speed_kmh'].std():.0f} km/h")

    # Band definitions
    bands = pd.read_parquet(BANDS_PATH)
    if "nominal_hz" in bands.columns and "band_nominal_hz" not in bands.columns:
        bands = bands.rename(columns={"nominal_hz": "band_nominal_hz"})
    primary = bands[bands["fully_inside"]].copy()
    primary_nominals = sorted(primary["band_nominal_hz"].tolist())
    print(f"  Primary bands: {len(primary_nominals)} → {primary_nominals}")

    # Spectral targets: MP8, primary bands only
    spec = pd.read_parquet(
        SPEC_V2 / "spectral_targets.parquet",
        columns=["event_id", "sensor_id", "band_nominal_hz",
                 "velocity_band_level_db", "fully_inside_valid_range"],
    )
    spec["event_id"] = spec["event_id"].astype(str)
    mp8 = spec[
        (spec["sensor_id"] == SENSOR_ID) & (spec["fully_inside_valid_range"])
    ].copy()
    mp8 = mp8[mp8["band_nominal_hz"].isin(primary_nominals)]
    print(f"  MP8 primary-band rows: {len(mp8):,}  events: {mp8['event_id'].nunique():,}")

    # Sanity: NaN / ±inf in targets
    n_nan = int(mp8["velocity_band_level_db"].isna().sum())
    n_inf = int(np.isinf(mp8["velocity_band_level_db"]).sum())
    print(f"  dB NaN={n_nan}  ±inf={n_inf}  "
          f"range=[{mp8['velocity_band_level_db'].min():.1f}, "
          f"{mp8['velocity_band_level_db'].max():.1f}] dB")

    # Pivot to wide: one row per event, 19 band columns
    band_col = {hz: f"b{hz:.4g}hz" for hz in primary_nominals}   # e.g. "b12.5hz"
    mp8["col"] = mp8["band_nominal_hz"].map(band_col)
    wide = mp8.pivot(index="event_id", columns="col",
                     values="velocity_band_level_db").reset_index()
    target_cols = [band_col[hz] for hz in primary_nominals]   # sorted by Hz

    # Merge metadata
    df = wide.merge(aligned[["event_id"] + CAT_COLS + NUM_COLS + ["split"]],
                    on="event_id", how="inner")
    print(f"  Wide shape: {df.shape}  (events × 1_id + {len(target_cols)}_bands + features)")

    # Check for any NaN in target columns
    n_target_nan = df[target_cols].isna().sum().sum()
    if n_target_nan > 0:
        print(f"  WARNING: {n_target_nan} NaN values in target bands — "
              f"events with any missing band will be dropped.")
        df = df.dropna(subset=target_cols).reset_index(drop=True)
        print(f"  After drop: {len(df):,} events")

    print(f"\n  Per-band dB statistics (mean ± std):")
    for col, hz in zip(target_cols, primary_nominals):
        v = df[col].values
        print(f"    {hz:6.4g} Hz: {v.mean():7.2f} ± {v.std():.2f} dB  "
              f"[{v.min():.1f}, {v.max():.1f}]")

    return df, primary_nominals, target_cols


# ── Step 2: Split ─────────────────────────────────────────────────────────────

def build_splits(df: pd.DataFrame, target_cols: list[str]) -> dict:
    section("STEP 2: Building train / val / test splits")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    print(f"  Train: {len(train_df):,} events")
    print(f"  Val  : {len(val_df):,} events")
    print(f"  Test : {len(test_df):,} events")

    feat_cols = CAT_COLS + NUM_COLS
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_miss = sdf[feat_cols].isna().sum().sum()
        if n_miss:
            print(f"  WARNING: {n_miss} missing feature values in {name}")

    def _arrays(sdf):
        X = sdf[feat_cols].copy()
        for c in CAT_COLS:
            X[c] = X[c].astype(str).fillna("Unknown")
        Y = sdf[target_cols].values.astype(np.float64)
        return X, Y

    X_tr, Y_tr = _arrays(train_df)
    X_va, Y_va = _arrays(val_df)
    X_te, Y_te = _arrays(test_df)

    return dict(
        X_train=X_tr, Y_train=Y_tr,
        X_val=X_va,   Y_val=Y_va,
        X_test=X_te,  Y_test=Y_te,
        train_df=train_df, val_df=val_df, test_df=test_df,
    )


# ── Step 3: Preprocessing (fit on training data only) ───────────────────────

def fit_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    section("STEP 3: Fitting preprocessor on training data")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False,
                                  handle_unknown="ignore"), CAT_COLS),
            ("num", SKPipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale",  StandardScaler()),
            ]), NUM_COLS),
        ],
        remainder="drop",
    )
    preprocessor.fit(X_train)

    ohe = preprocessor.named_transformers_["cat"]
    ohe_names = list(ohe.get_feature_names_out(CAT_COLS))
    all_names = ohe_names + NUM_COLS
    print(f"  Features ({len(all_names)}): {all_names}")

    # Report dropped (reference) categories
    for i, col in enumerate(CAT_COLS):
        cats = list(ohe.categories_[i])
        dropped = ohe.drop_idx_[i]
        print(f"  OHE [{col}]: categories={cats}  "
              f"reference (dropped)='{cats[dropped]}'")

    return preprocessor


def _transform(prep: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    Xc = X.copy()
    for c in CAT_COLS:
        Xc[c] = Xc[c].astype(str).fillna("Unknown")
    return prep.transform(Xc).astype(np.float64)


# ── Step 4: Alpha selection on validation ───────────────────────────────────

def select_alpha(prep: ColumnTransformer, splits: dict) -> float:
    section("STEP 4: Alpha selection on validation set")

    X_tr = _transform(prep, splits["X_train"])
    X_va = _transform(prep, splits["X_val"])
    Y_tr = splits["Y_train"]
    Y_va = splits["Y_val"]
    n_bands = Y_tr.shape[1]

    print(f"  {'Alpha':>8}  {'Val macro-RMSE (dB)':>22}  {'Val macro-R²':>14}")
    print(f"  {'-'*8}  {'-'*22}  {'-'*14}")

    best_alpha, best_val_rmse = ALPHA_GRID[0], np.inf
    for alpha in ALPHA_GRID:
        m = Ridge(alpha=alpha)
        m.fit(X_tr, Y_tr)
        pred = m.predict(X_va)
        macro_rmse = float(np.mean([_rmse(Y_va[:, i], pred[:, i])
                                    for i in range(n_bands)]))
        macro_r2   = float(np.mean([_r2(Y_va[:, i], pred[:, i])
                                    for i in range(n_bands)]))
        marker = " ◄" if macro_rmse < best_val_rmse else ""
        print(f"  {alpha:>8.3g}  {macro_rmse:>22.4f}  {macro_r2:>14.4f}{marker}")
        if macro_rmse < best_val_rmse:
            best_alpha, best_val_rmse = alpha, macro_rmse

    print(f"\n  Best alpha: {best_alpha}  (val macro-RMSE = {best_val_rmse:.4f} dB)")
    return best_alpha


# ── Step 5: Fit final model on training data ────────────────────────────────

def fit_model(prep: ColumnTransformer, splits: dict, alpha: float) -> Ridge:
    section("STEP 5: Fitting final model on training data")

    X_tr = _transform(prep, splits["X_train"])
    Y_tr = splits["Y_train"]

    model = Ridge(alpha=alpha)
    model.fit(X_tr, Y_tr)

    # Training-set sanity (never reported as performance)
    pred_tr = model.predict(X_tr)
    n_bands = Y_tr.shape[1]
    tr_macro_rmse = float(np.mean([_rmse(Y_tr[:, i], pred_tr[:, i])
                                   for i in range(n_bands)]))
    tr_macro_r2 = float(np.mean([_r2(Y_tr[:, i], pred_tr[:, i])
                                  for i in range(n_bands)]))

    print(f"  Ridge(alpha={alpha})  n_train={len(X_tr):,}  n_features={X_tr.shape[1]}")
    print(f"  Train macro-RMSE = {tr_macro_rmse:.4f} dB  "
          f"macro-R² = {tr_macro_r2:.4f}  (reference only — not the reported metric)")
    return model


# ── Step 6: Test-set evaluation ──────────────────────────────────────────────

def evaluate_test(
    model: Ridge,
    prep: ColumnTransformer,
    splits: dict,
    primary_nominals: list[float],
    target_cols: list[str],
) -> dict:
    section("STEP 6: Test-set evaluation")

    X_te = _transform(prep, splits["X_test"])
    Y_te = splits["Y_test"]
    Y_pred = model.predict(X_te)
    n_bands = Y_te.shape[1]

    # Per-band metrics
    print(f"\n  {'Hz':>6}  {'RMSE (dB)':>10}  {'MAE (dB)':>10}  "
          f"{'R²':>8}  {'σ_true':>8}  {'RMSE/σ':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")

    per_band = []
    for i, (hz, col) in enumerate(zip(primary_nominals, target_cols)):
        yt = Y_te[:, i]
        yp = Y_pred[:, i]
        rm  = _rmse(yt, yp)
        ma  = _mae(yt, yp)
        r2_ = _r2(yt, yp)
        sig = float(np.std(yt))
        per_band.append({
            "band_hz": hz, "rmse_db": rm, "mae_db": ma,
            "r2": r2_, "std_true_db": sig,
            "rmse_over_sigma": rm / (sig + 1e-9),
        })
        print(f"  {hz:6.4g}  {rm:10.4f}  {ma:10.4f}  {r2_:8.4f}  {sig:8.4f}  "
              f"{rm/(sig+1e-9):8.4f}")

    macro_rmse = float(np.mean([b["rmse_db"] for b in per_band]))
    macro_mae  = float(np.mean([b["mae_db"]  for b in per_band]))
    macro_r2   = float(np.mean([b["r2"]      for b in per_band]))

    print(f"\n  Macro (mean across {n_bands} bands):")
    print(f"    RMSE = {macro_rmse:.4f} dB")
    print(f"    MAE  = {macro_mae:.4f} dB")
    print(f"    R²   = {macro_r2:.4f}")

    # Total RMS reconstruction from 19 primary bands
    true_total = total_rms_from_db(Y_te)
    pred_total = total_rms_from_db(Y_pred)
    tot_rmse = _rmse(true_total, pred_total)
    tot_mae  = _mae(true_total, pred_total)
    tot_r2   = _r2(true_total, pred_total)
    tot_r    = _pearson(true_total, pred_total)

    print(f"\n  Total RMS (sum-of-squares from 19 primary bands):")
    print(f"    RMSE      = {tot_rmse:.4f} mm/s")
    print(f"    MAE       = {tot_mae:.4f} mm/s")
    print(f"    R²        = {tot_r2:.4f}")
    print(f"    Pearson r = {tot_r:.4f}")
    print(f"    True  total RMS: mean={true_total.mean():.4f}  "
          f"std={true_total.std():.4f} mm/s")
    print(f"    Pred  total RMS: mean={pred_total.mean():.4f}  "
          f"std={pred_total.std():.4f} mm/s")

    # Per-event RMSE distribution (across all 19 bands)
    per_ev_rmse = np.sqrt(np.mean((Y_te - Y_pred) ** 2, axis=1))
    print(f"\n  Per-event spectral RMSE across 19 bands:")
    print(f"    median = {np.median(per_ev_rmse):.4f} dB")
    print(f"    p90    = {np.percentile(per_ev_rmse, 90):.4f} dB")
    print(f"    max    = {per_ev_rmse.max():.4f} dB")

    return {
        "per_band": per_band,
        "macro": {"rmse_db": macro_rmse, "mae_db": macro_mae, "r2": macro_r2},
        "total_rms": {
            "rmse_mms": tot_rmse, "mae_mms": tot_mae,
            "r2": tot_r2, "pearson_r": tot_r,
        },
        "residuals_summary": {
            "median_per_event_rmse_db":  float(np.median(per_ev_rmse)),
            "p90_per_event_rmse_db":     float(np.percentile(per_ev_rmse, 90)),
            "max_per_event_rmse_db":     float(per_ev_rmse.max()),
        },
        "Y_pred": Y_pred,
        "Y_true": Y_te,
        "true_total": true_total,
        "pred_total": pred_total,
    }


# ── Step 7: Subgroup analysis ────────────────────────────────────────────────

def subgroup_analysis(results: dict, splits: dict, target_cols: list[str]) -> dict:
    section("STEP 7: Subgroup analysis")

    Y_pred = results["Y_pred"]
    Y_true = results["Y_true"]
    true_tot = results["true_total"]
    pred_tot = results["pred_total"]

    test_df = splits["test_df"].copy()
    test_df = test_df.reset_index(drop=True)
    n_bands = Y_true.shape[1]

    def _row_metrics(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        yt_s, yp_s = Y_true[mask], Y_pred[mask]
        mac_rm = float(np.mean([_rmse(yt_s[:, i], yp_s[:, i]) for i in range(n_bands)]))
        mac_r2 = float(np.mean([_r2(yt_s[:, i], yp_s[:, i]) for i in range(n_bands)]))
        tt, pt = true_tot[mask], pred_tot[mask]
        t_rm, t_r = _rmse(tt, pt), _pearson(tt, pt)
        return {"n": n, "macro_rmse_db": mac_rm, "macro_r2": mac_r2,
                "total_rmse_mms": t_rm, "pearson_r_total": t_r}

    sub = {}

    # By track
    print(f"\n  By track:")
    print(f"  {'Track':>8}  {'n':>5}  {'MacroRMSE(dB)':>14}  "
          f"{'MacroR²':>9}  {'TotalRMSE(mm/s)':>16}  {'r_total':>8}")
    for tv in sorted(test_df["track_number"].unique()):
        mask = (test_df["track_number"] == str(tv)).values
        if mask.sum() < 5:
            continue
        m = _row_metrics(mask)
        sub[f"track_{tv}"] = m
        print(f"  {str(tv):>8}  {m['n']:>5}  {m['macro_rmse_db']:>14.4f}  "
              f"{m['macro_r2']:>9.4f}  {m['total_rmse_mms']:>16.4f}  "
              f"{m['pearson_r_total']:>8.4f}")

    # By train family
    print(f"\n  By train family (n≥20):")
    print(f"  {'Family':>15}  {'n':>5}  {'MacroRMSE(dB)':>14}  "
          f"{'MacroR²':>9}  {'TotalRMSE(mm/s)':>16}  {'r_total':>8}")
    for fam in sorted(test_df["train_family"].unique()):
        mask = (test_df["train_family"] == fam).values
        n = int(mask.sum())
        if n < 20:
            print(f"  {fam:>15}  {n:>5}  — skip (n<20)")
            continue
        m = _row_metrics(mask)
        sub[f"family_{fam}"] = m
        print(f"  {fam:>15}  {m['n']:>5}  {m['macro_rmse_db']:>14.4f}  "
              f"{m['macro_r2']:>9.4f}  {m['total_rmse_mms']:>16.4f}  "
              f"{m['pearson_r_total']:>8.4f}")

    return sub


# ── Step 8: Plots ─────────────────────────────────────────────────────────────

def make_plots(
    results: dict,
    model: Ridge,
    prep: ColumnTransformer,
    primary_nominals: list[float],
    target_cols: list[str],
    splits: dict,
    out_dir: Path,
) -> None:
    section("STEP 8: Generating plots")

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    per_band  = results["per_band"]
    hz_vals   = [b["band_hz"]     for b in per_band]
    rmse_vals = [b["rmse_db"]     for b in per_band]
    r2_vals   = [b["r2"]          for b in per_band]
    std_vals  = [b["std_true_db"] for b in per_band]
    xticklabels = [f"{hz:.4g}" for hz in hz_vals]
    xpos = np.arange(len(hz_vals))

    # ── Plot 1: Per-band RMSE + R² ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax = axes[0]
    ax.bar(xpos, std_vals,  color="lightgray",  alpha=0.6, label="σ true (dB)", zorder=1)
    ax.bar(xpos, rmse_vals, color="steelblue",  alpha=0.85, label="RMSE (dB)",  zorder=2)
    ax.set_ylabel("dB")
    ax.set_title("Per-band test RMSE vs true standard deviation — Ridge metadata baseline")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    colors = ["steelblue" if v >= 0 else "tomato" for v in r2_vals]
    ax.bar(xpos, r2_vals, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("R²")
    ax.set_title("Per-band R² on test set")
    ax.set_xticks(xpos)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.set_xlabel("Band nominal frequency (Hz)")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(plots_dir / "per_band_rmse_r2.png", dpi=120)
    plt.close(fig)
    print("  Saved: per_band_rmse_r2.png")

    # ── Plot 2: Predicted vs actual total RMS ─────────────────────────────────
    true_tot = results["true_total"]
    pred_tot = results["pred_total"]
    tot = results["total_rms"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_tot, pred_tot, alpha=0.45, s=14, color="steelblue", edgecolors="none")
    lo = min(true_tot.min(), pred_tot.min()) * 0.9
    hi = max(true_tot.max(), pred_tot.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, label="1:1")
    ax.set_xlabel("True total RMS (mm/s, 19 bands)")
    ax.set_ylabel("Predicted total RMS (mm/s)")
    ax.set_title(
        f"Total RMS — test set\n"
        f"RMSE={tot['rmse_mms']:.4f} mm/s  R²={tot['r2']:.3f}  "
        f"r={tot['pearson_r']:.3f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "total_rms_scatter.png", dpi=120)
    plt.close(fig)
    print("  Saved: total_rms_scatter.png")

    # ── Plot 3: Mean spectrum true vs predicted ───────────────────────────────
    Y_te   = results["Y_true"]
    Y_pred = results["Y_pred"]
    Y_tr   = splits["Y_train"]

    mean_tr_true = Y_tr.mean(axis=0)
    mean_te_true = Y_te.mean(axis=0)
    mean_te_pred = Y_pred.mean(axis=0)
    std_te_true  = Y_te.std(axis=0)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(xpos, mean_tr_true,  "o-",  color="gray",      linewidth=1.2,
            label="Train mean (true)")
    ax.plot(xpos, mean_te_true,  "o-",  color="black",     linewidth=1.5,
            label="Test mean (true)")
    ax.plot(xpos, mean_te_pred,  "s--", color="steelblue", linewidth=1.5,
            label="Test mean (predicted)")
    ax.fill_between(xpos,
                    mean_te_true - std_te_true,
                    mean_te_true + std_te_true,
                    alpha=0.12, color="black", label="Test ±1σ (true)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.set_xlabel("Band nominal frequency (Hz)")
    ax.set_ylabel("Velocity band level (dB re 1 nm/s)")
    ax.set_title("Mean one-third-octave spectrum: true vs metadata-predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "mean_spectrum.png", dpi=120)
    plt.close(fig)
    print("  Saved: mean_spectrum.png")

    # ── Plot 4: Ridge coefficient heatmap ─────────────────────────────────────
    ohe = prep.named_transformers_["cat"]
    feat_names = list(ohe.get_feature_names_out(CAT_COLS)) + NUM_COLS
    coef = model.coef_   # (n_bands, n_features)
    vmax = np.abs(coef).max()

    fig, ax = plt.subplots(figsize=(max(8, len(feat_names) * 0.7), 7))
    im = ax.imshow(coef, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Coefficient (dB per unit / dB per σ_speed)")
    ax.set_yticks(range(len(primary_nominals)))
    ax.set_yticklabels([f"{hz:.4g} Hz" for hz in primary_nominals], fontsize=8)
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=8)
    ax.set_title("Ridge regression coefficients: metadata → dB per band")
    plt.tight_layout()
    fig.savefig(plots_dir / "ridge_coefficients.png", dpi=120)
    plt.close(fig)
    print("  Saved: ridge_coefficients.png")

    print(f"  Plots → {plots_dir}")


# ── Step 9: Save artifacts ───────────────────────────────────────────────────

def save_artifacts(
    model: Ridge,
    prep: ColumnTransformer,
    results: dict,
    subgroup: dict,
    splits: dict,
    target_cols: list[str],
    primary_nominals: list[float],
    best_alpha: float,
    out_dir: Path,
) -> None:
    section("STEP 9: Saving artifacts")

    # Config
    config = {
        "version": "meta_spectral_v001",
        "created": datetime.now().isoformat(timespec="seconds"),
        "model": "Ridge",
        "alpha": best_alpha,
        "random_seed": RANDOM_SEED,
        "features": {
            "categorical": CAT_COLS,
            "numerical": NUM_COLS,
            "note": "train_direction excluded — all values NaN in aligned_events.parquet",
            "ohe_drop": "first",
        },
        "target": "velocity_band_level_db (dB re 1 nm/s)",
        "sensor": SENSOR_ID,
        "n_primary_bands": len(primary_nominals),
        "primary_nominals_hz": primary_nominals,
        "target_cols": target_cols,
        "split_sizes": {
            "train": len(splits["train_df"]),
            "val":   len(splits["val_df"]),
            "test":  len(splits["test_df"]),
        },
        "spec_source": str(SPEC_V2),
        "align_source": str(ALIGN),
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    # Per-band metrics
    pb_df = pd.DataFrame(results["per_band"])
    pb_df.to_csv(out_dir / "metrics_per_band.csv", index=False)

    # Full metrics
    full_metrics = {
        "per_band": results["per_band"],
        "macro": results["macro"],
        "total_rms": results["total_rms"],
        "residuals_summary": results["residuals_summary"],
        "subgroup": subgroup,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(full_metrics, indent=2), encoding="utf-8"
    )

    # Predictions on test set
    test_df = splits["test_df"].copy().reset_index(drop=True)
    for i, col in enumerate(target_cols):
        test_df[f"pred_{col}"] = results["Y_pred"][:, i]
        test_df[f"true_{col}"] = results["Y_true"][:, i]
        test_df[f"resid_{col}"] = results["Y_true"][:, i] - results["Y_pred"][:, i]
    test_df["pred_total_rms_mms"] = results["pred_total"]
    test_df["true_total_rms_mms"] = results["true_total"]
    out_cols = (
        ["event_id", "split", "train_type", "train_family",
         "train_speed_kmh", "track_number"]
        + [f"pred_{c}" for c in target_cols]
        + [f"true_{c}" for c in target_cols]
        + [f"resid_{c}" for c in target_cols]
        + ["pred_total_rms_mms", "true_total_rms_mms"]
    )
    test_df[[c for c in out_cols if c in test_df.columns]].to_parquet(
        out_dir / "predictions_test.parquet", index=False
    )

    # Model + preprocessor
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "preprocessor": prep,
                "target_cols": target_cols,
                "primary_nominals": primary_nominals,
                "cat_cols": CAT_COLS,
                "num_cols": NUM_COLS,
            },
            f,
        )

    # Ridge coefficients as CSV (for inspection)
    ohe = prep.named_transformers_["cat"]
    feat_names = list(ohe.get_feature_names_out(CAT_COLS)) + NUM_COLS
    coef_df = pd.DataFrame(
        model.coef_,
        columns=feat_names,
        index=[f"{hz:.4g}hz" for hz in primary_nominals],
    )
    coef_df.index.name = "band"
    coef_df["intercept"] = model.intercept_
    coef_df.to_csv(out_dir / "ridge_coefficients.csv")

    # Summary
    best_b  = min(results["per_band"], key=lambda b: b["rmse_db"])
    worst_b = max(results["per_band"], key=lambda b: b["rmse_db"])
    summary = {
        "headline": (
            f"Ridge metadata baseline: "
            f"macro-RMSE={results['macro']['rmse_db']:.2f} dB, "
            f"macro-R²={results['macro']['r2']:.3f}, "
            f"total-RMS r={results['total_rms']['pearson_r']:.3f}"
        ),
        "best_band":  best_b,
        "worst_band": worst_b,
        "macro": results["macro"],
        "total_rms": results["total_rms"],
        "residuals_summary": results["residuals_summary"],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"  config.json, metrics.json, metrics_per_band.csv,")
    print(f"  predictions_test.parquet, model.pkl, ridge_coefficients.csv,")
    print(f"  summary.json  →  {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Started: {datetime.now().isoformat(timespec='seconds')}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = MODELS_ROOT / f"meta_spectral_baseline_v001_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output : {out_dir}")

    df, primary_nominals, target_cols = load_data()
    splits     = build_splits(df, target_cols)
    prep       = fit_preprocessor(splits["X_train"])
    best_alpha = select_alpha(prep, splits)
    model      = fit_model(prep, splits, best_alpha)
    results    = evaluate_test(model, prep, splits, primary_nominals, target_cols)
    subgroup   = subgroup_analysis(results, splits, target_cols)
    make_plots(results, model, prep, primary_nominals, target_cols, splits, out_dir)
    save_artifacts(model, prep, results, subgroup, splits, target_cols,
                   primary_nominals, best_alpha, out_dir)

    print(f"\n{'='*70}")
    print(f"RESULT  {summary_line(results)}")
    print(f"{'='*70}")
    print(f"\nFinished: {datetime.now().isoformat(timespec='seconds')}")
    print(f"All outputs: {out_dir}")


def summary_line(results: dict) -> str:
    m  = results["macro"]
    tr = results["total_rms"]
    return (
        f"macro-RMSE={m['rmse_db']:.2f} dB  "
        f"macro-MAE={m['mae_db']:.2f} dB  "
        f"macro-R²={m['r2']:.3f}  |  "
        f"total-RMS RMSE={tr['rmse_mms']:.4f} mm/s  "
        f"r={tr['pearson_r']:.3f}  R²={tr['r2']:.3f}"
    )


if __name__ == "__main__":
    main()
