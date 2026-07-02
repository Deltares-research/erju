"""
train_fo_physprofile_linec_v2_geometry.py  —  FO-PhysProfile v2 Geometry

Ablation experiment: explicit FO-relative geometry added to v1 local residual model.

Coordinate system (from sites/holten.json):
  FO cable: Y = 0    Track 1: Y = 4.0 m    Track 2: Y = 8.0 m
  MP4  Y=+1.5 → between FO and track (fo_axis_pos=0.375)
  MP8  Y= 0.0 → on FO cable          (fo_axis_pos=0.000)
  MP10 Y=-4.0 → behind FO            (fo_axis_pos=-1.00)
  MP1  Y=-12. → behind FO far        (fo_axis_pos=-3.00)
  MP2  Y=-19. → behind FO far        (fo_axis_pos=-4.75)

Variants (local residual only; all share the same v1 profile predictions):
  V1_ref                         v1 feature set (reference)
  V2_geom                        v1 + geometry + geometry×FO interactions
  V2_geom_no_sc                  V2_geom without sensor_code + feat_sc_x_*
  V2_geom_no_sensor_no_trackcat  V2_geom_no_sc without track_number

Selection policies per variant:
  global_best    val RMSE(PGV)
  mp4_best       val MP4 RMSE(PGV)
  balanced_best  val RMSE(PGV) + 0.15×MP4 RMSE(PGV)

Weighting schemes:
  V1_ref:       uniform, agg_hi, mp4_hi
  V2 variants:  + between_hi, between_mp4_hi, mp4_extreme_hi

Baselines:
  FO-PhysProfile v1:  RMSE(PGV)=2.1882  MP4=4.5592  <- beat this
  PXGBR_ens_top5:     RMSE(PGV)=2.2473  MP4=4.6680
  PXGBR-R2:           RMSE(PGV)=2.2718  MP4=4.7389
  P3:                 RMSE(PGV)=2.4006  MP4=5.0185

Patches applied:
  PATCH 1  Prefers saved v1 test predictions (clean ablation)
  PATCH 2  PC3 disabled — n_comp_use always = 2
  PATCH 3  No duplicate column merges in row feature builder
  PATCH 4  No-sensor-code ablation removes feat_sc_x_* + sensor_code
  PATCH 5  No-sensor-no-trackcat ablation
  PATCH 6  6 weighting schemes incl. between_hi / mp4_extreme_hi
  PATCH 7  3 selection policies per variant
  PATCH 8  Geometry assertions for Track 1 expected values
  PATCH 9  Geometry zone metrics CSV
  PATCH 10 Feature importance CSVs + residual vs axis pos plots
  PATCH 11 Comprehensive interpretation block
  PATCH 12 Smoke test geometry assertions
"""

from __future__ import annotations

import argparse
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
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

import train_fo_physprofile_linec_v1 as _v1
from src.utils.geometry_utils import apply_corrected_distances


# ─── Constants ────────────────────────────────────────────────────────────────

def _root() -> Path:
    return Path(r"P:\11210978-erju-ai") if os.name == "nt" else Path("/p/11210978-erju-ai")

MODELS_ROOT  = _root() / "holten_models"
PARQUET_ROOT = _root() / "holten_parquet"
WAVE_ROOT    = _root() / "holten_waveform"

SENSOR_ORDER = ["MP4", "MP8", "MP10", "MP1", "MP2"]
SENSOR_CODE  = {s: i for i, s in enumerate(SENSOR_ORDER)}

N_TRACK    = {1: 1.0655, 2: 1.3246}
R0         = 10.0
N_SHRINK_W = 0.3
N_CLIP     = (0.5, 2.0)
PCA_N_COMP = 3          # fit up to 3; n_comp_use always = 2 in v2

SITE_CONFIG_PATH = _REPO / "sites" / "holten.json"
_LEAKAGE_TOKENS  = _v1._LEAKAGE_TOKENS

# ── Variant definitions ────────────────────────────────────────────────────────

_GEO_SCHEMES  = ["uniform", "agg_hi", "mp4_hi",
                  "between_hi", "between_mp4_hi", "mp4_extreme_hi"]
_BASE_SCHEMES = ["uniform", "agg_hi", "mp4_hi"]

VARIANT_CONFIGS: Dict[str, dict] = {
    "V1_ref": {
        "use_geometry":     False,
        "use_sensor_code":  True,
        "use_track_number": True,
        "schemes": _BASE_SCHEMES,
    },
    "V2_geom": {
        "use_geometry":     True,
        "use_sensor_code":  True,
        "use_track_number": True,
        "schemes": _GEO_SCHEMES,
    },
    "V2_geom_no_sc": {
        "use_geometry":     True,
        "use_sensor_code":  False,
        "use_track_number": True,
        "schemes": _GEO_SCHEMES,
    },
    "V2_geom_no_sensor_no_trackcat": {
        "use_geometry":     True,
        "use_sensor_code":  False,
        "use_track_number": False,
        "schemes": _GEO_SCHEMES,
    },
}


# ─── Path helpers ──────────────────────────────────────────────────────────────

def make_output_dir(base: Path) -> Path:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (base / "outputs" / "fo_physprofile_linec_v2_geometry"
           / f"fo_physprofile_linec_v002_geom_{ts}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def find_v1_output() -> Optional[Path]:
    root = MODELS_ROOT / "outputs" / "fo_physprofile_linec_v1"
    if not root.exists():
        return None
    for h in reversed(sorted(root.glob("fo_physprofile_linec_v001_*"))):
        if (h / "predictions_test.parquet").exists():
            return h
    return None


# ─── PATCH 1: V1 artifact loading ─────────────────────────────────────────────

_V1_REQUIRED = {"event_id", "sensor", "pred_log_profile", "target_log",
                "target_pgv", "distance", "track_number",
                "c_hat_profile", "n_hat_profile", "pc1_hat", "pc2_hat"}


def load_v1_artifacts(v1_dir: Optional[Path]) -> Optional[dict]:
    """Load files from a completed v1 output directory."""
    if v1_dir is None or not v1_dir.exists():
        return None
    arts: dict = {"v1_dir": v1_dir}

    pp = v1_dir / "predictions_test.parquet"
    if pp.exists():
        df = pd.read_parquet(pp)
        df["event_id"] = df["event_id"].astype(str)
        miss = _V1_REQUIRED - set(df.columns)
        if not miss:
            arts["predictions_test"] = df
            print(f"[v1_load] predictions_test.parquet: {len(df):,} rows  ✓")
        else:
            print(f"[v1_load] predictions_test missing cols: {miss}")
    else:
        print(f"[v1_load] predictions_test.parquet not found in {v1_dir.name}")

    for fname, key in [
        ("event_features.parquet",      "event_features"),
        ("row_features.parquet",         "row_features_test"),
        ("residual_profile_pca.pkl",     None),
        ("metrics_table.csv",            "metrics_table"),
    ]:
        p = v1_dir / fname
        if not p.exists():
            continue
        if fname.endswith(".pkl"):
            with open(p, "rb") as f:
                arts["pca"] = pickle.load(f)
            print(f"[v1_load] {fname}  ✓")
        elif fname.endswith(".csv"):
            arts[key] = pd.read_csv(p)
        else:
            df2 = pd.read_parquet(p)
            df2["event_id"] = df2["event_id"].astype(str)
            arts[key] = df2
            print(f"[v1_load] {fname}: {len(df2):,} rows  ✓")

    arts["can_reuse_test"] = "predictions_test" in arts
    print(f"[v1_load] can_reuse_test={arts['can_reuse_test']}")
    return arts if len(arts) > 1 else None


# ─── Site geometry ─────────────────────────────────────────────────────────────

def load_site_config(path: Path = SITE_CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    with open(path) as f:
        return json.load(f)


def build_sensor_geometry_table(config: dict) -> pd.DataFrame:
    coords = config["accelerometer"]["sensor_coordinates_m"]
    fo     = config["fibre_optics"]
    rows = []
    for s in SENSOR_ORDER:
        if s not in coords:
            continue
        rows.append({
            "sensor":     s,
            "sensor_x_m": float(coords[s]["x"]),
            "sensor_y_m": float(coords[s]["y"]),
            "fo_y_m":     0.0,
            "track1_y_m": float(fo["track_1_to_fo_m"]),
            "track2_y_m": float(fo["track_2_to_fo_m"]),
        })
    return pd.DataFrame(rows).set_index("sensor")


def print_geometry_diagnostic(geom_df: pd.DataFrame) -> None:
    fo_y = 0.0
    print("\n" + "=" * 80)
    print("GEOMETRY DIAGNOSTIC — FO-Relative Coordinates (holten.json)")
    print("=" * 80)
    print(f"{'Sensor':>6}  {'y_m':>6}  {'Track':>5}  {'track_y':>7}  "
          f"{'axis_pos':>9}  {'to_track':>8}  {'to_fo':>6}  {'ratio':>5}  Zone")
    print("-" * 80)
    for s in SENSOR_ORDER:
        if s not in geom_df.index:
            continue
        sy = float(geom_df.loc[s, "sensor_y_m"])
        for tn, ty in [(1, float(geom_df.loc[s, "track1_y_m"])),
                        (2, float(geom_df.loc[s, "track2_y_m"]))]:
            ap    = (sy - fo_y) / (ty - fo_y)
            ratio = abs(sy - ty) / abs(ty - fo_y)
            if   ap > 1.0:               zone = "beyond_track"
            elif ap < 0.0:               zone = "beyond_fo_away"
            elif abs(sy - fo_y) < 0.75:  zone = "on_fo_line"
            else:                        zone = "between_fo_track"
            print(f"{s:>6}  {sy:>6.1f}  T{tn:>4}  {ty:>7.1f}  "
                  f"{ap:>9.4f}  {abs(sy-ty):>8.2f}  "
                  f"{abs(sy-fo_y):>6.2f}  {ratio:>5.3f}  {zone}")
    print("=" * 80)


# ─── PATCH 8: Geometry assertions ─────────────────────────────────────────────

def assert_geometry_values(geom_df: pd.DataFrame) -> None:
    """Assert expected FO-relative geometry values for Track 1."""
    test_df = pd.DataFrame({
        "event_id":     [f"e{i}" for i in range(5)],
        "sensor":       SENSOR_ORDER,
        "track_number": [1] * 5,
        "distance":     [2.5, 4.0, 8.0, 16.0, 23.0],
        "log_distance": [0.0] * 5,
        "sensor_code":  list(range(5)),
        "pred_log_profile": [0.0] * 5,
    })
    g = add_fo_relative_geometry_features(test_df, geom_df)
    expected = {
        "MP4":  {"fo_to_track_axis_pos": 0.375, "acc_to_active_track_m": 2.5,
                 "is_between_track_and_fo": 1.0, "acc_track_to_fo_ratio": 0.625},
        "MP8":  {"fo_to_track_axis_pos": 0.0,   "acc_to_active_track_m": 4.0,
                 "is_on_fo_line": 1.0,            "acc_track_to_fo_ratio": 1.0},
        "MP10": {"fo_to_track_axis_pos": -1.0,   "acc_to_active_track_m": 8.0,
                 "is_beyond_fo_away_from_track": 1.0, "acc_track_to_fo_ratio": 2.0},
        "MP1":  {"fo_to_track_axis_pos": -3.0},
        "MP2":  {"fo_to_track_axis_pos": -4.75},
    }
    errs = []
    for s, checks in expected.items():
        row = g[g["sensor"] == s].iloc[0]
        for col, exp in checks.items():
            actual = float(row[col])
            if abs(actual - exp) > 1e-6:
                errs.append(f"  {s}.{col}: expected {exp}, got {actual:.6f}")
    if errs:
        raise AssertionError("Geometry assertions failed:\n" + "\n".join(errs))
    print("[geometry_assert] Track-1 values correct  ✓")


# ─── FO-relative geometry features ────────────────────────────────────────────

_GEOM_COLS = [
    "fo_to_track_axis_pos", "acc_track_to_fo_ratio",
    "is_between_track_and_fo", "is_on_fo_line",
    "is_beyond_fo_away_from_track", "acc_to_fo_m",
    "signed_acc_from_fo", "signed_acc_from_active_track",
    "log_acc_to_active_track", "log_acc_track_to_fo_ratio",
    "acc_to_fo_over_track_to_fo", "active_track_to_fo_m",
    "sensor_x_m", "sensor_y_m", "acc_to_active_track_m",
]

_GEOM_WF_PAIRS = [
    ("fo_to_track_axis_pos",    "wf_ch_energy_max"),
    ("fo_to_track_axis_pos",    "wf_global_energy"),
    ("fo_to_track_axis_pos",    "wf_global_rms"),
    ("fo_to_track_axis_pos",    "wf_high_low_energy_ratio"),
    ("fo_to_track_axis_pos",    "wf_spectral_centroid"),
    ("acc_track_to_fo_ratio",   "wf_ch_energy_max"),
    ("acc_track_to_fo_ratio",   "wf_high_low_energy_ratio"),
    ("is_between_track_and_fo", "wf_ch_energy_max"),
    ("is_between_track_and_fo", "wf_global_energy"),
    ("is_between_track_and_fo", "wf_high_low_energy_ratio"),
]
_GEOM_OCT_PREFIXES = [
    ("fo_to_track_axis_pos",    "fo_oct_063hz"),
    ("fo_to_track_axis_pos",    "fo_oct_1_00hz"),
    ("fo_to_track_axis_pos",    "fo_oct_2_00hz"),
    ("fo_to_track_axis_pos",    "fo_oct_5_00hz"),
    ("fo_to_track_axis_pos",    "fo_oct_8_00hz"),
    ("acc_track_to_fo_ratio",   "fo_oct_063hz"),
    ("acc_track_to_fo_ratio",   "fo_oct_1_00hz"),
    ("is_between_track_and_fo", "fo_oct_063hz"),
    ("is_between_track_and_fo", "fo_oct_1_00hz"),
    ("is_between_track_and_fo", "fo_oct_5_00hz"),
]
_GEOM_PROFILE_PAIRS = [
    ("is_between_track_and_fo", "pred_log_profile"),
    ("acc_track_to_fo_ratio",   "pred_log_profile"),
    ("fo_to_track_axis_pos",    "pred_log_profile"),
]


def add_fo_relative_geometry_features(df: pd.DataFrame, geom_df: pd.DataFrame,
                                        r0: float = R0) -> pd.DataFrame:
    df   = df.copy()
    fo_y = 0.0
    t1y  = float(geom_df["track1_y_m"].iloc[0])
    t2y  = float(geom_df["track2_y_m"].iloc[0])
    sy   = df["sensor"].map(geom_df["sensor_y_m"].to_dict()).astype(float).values
    sx   = df["sensor"].map(geom_df["sensor_x_m"].to_dict()).astype(float).values
    tn   = df["track_number"].values.astype(int)
    tray = np.where(tn == 1, t1y, t2y)

    df["sensor_x_m"]                  = sx
    df["sensor_y_m"]                  = sy
    df["fo_y_m"]                      = fo_y
    df["active_track_y_m"]            = tray
    df["active_track_to_fo_m"]        = np.abs(tray - fo_y)
    df["acc_to_active_track_m"]       = np.abs(sy - tray)
    df["signed_acc_from_active_track"] = sy - tray
    df["log_acc_to_active_track"]     = np.log(np.maximum(np.abs(sy-tray), 0.1) / r0)
    df["acc_to_fo_m"]                 = np.abs(sy - fo_y)
    df["signed_acc_from_fo"]          = sy - fo_y
    denom = tray - fo_y
    df["fo_to_track_axis_pos"]        = (sy - fo_y) / denom
    axis  = df["fo_to_track_axis_pos"].values
    df["is_between_track_and_fo"]      = ((axis > 0) & (axis < 1)).astype(float)
    df["is_on_fo_line"]                = (np.abs(sy - fo_y) <= 0.75).astype(float)
    df["is_beyond_fo_away_from_track"] = (axis < 0).astype(float)
    df["is_beyond_track"]              = (axis > 1).astype(float)
    df["is_track_side_of_fo"]          = df["is_between_track_and_fo"]
    dtrack = np.maximum(np.abs(sy - tray), 0.1)
    dfo    = np.maximum(np.abs(sy - fo_y), 0.1)
    d_t2fo = np.maximum(np.abs(tray - fo_y), 0.1)
    df["acc_track_to_fo_ratio"]        = dtrack / d_t2fo
    df["log_acc_track_to_fo_ratio"]    = np.log(np.maximum(dtrack / d_t2fo, 1e-3))
    df["acc_to_fo_over_track_to_fo"]   = dfo / d_t2fo
    return df


def add_geometry_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for gcol, wcol in _GEOM_WF_PAIRS:
        if gcol in df.columns and wcol in df.columns:
            df[f"gi_{gcol}_x_{wcol}"] = df[gcol] * df[wcol]
    for gcol, pfx in _GEOM_OCT_PREFIXES:
        if gcol not in df.columns:
            continue
        for wcol in [c for c in df.columns if c.startswith(pfx)]:
            df[f"gi_{gcol}_x_{wcol}"] = df[gcol] * df[wcol]
    for gcol, pcol in _GEOM_PROFILE_PAIRS:
        if gcol in df.columns and pcol in df.columns:
            df[f"gi_{gcol}_x_{pcol}"] = df[gcol] * df[pcol]
    return df


# ─── PATCH 3: Row feature builder ─────────────────────────────────────────────

def _check_no_suffix_cols(df: pd.DataFrame, label: str = "") -> None:
    bad = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    if bad:
        print(f"[WARN{' '+label if label else ''}] suffix cols: {bad[:5]}")


def _build_row_features_v2(
    df_rows: pd.DataFrame,
    event_feat_df: pd.DataFrame,
    geom_df: pd.DataFrame,
    df_linec: Optional[pd.DataFrame] = None,
    add_geometry: bool = True,
) -> pd.DataFrame:
    """Build row-level feature matrix avoiding duplicate column merges."""
    drop_ev = ["split", "max_pgv_event", "c_target_event",
               "n_event_raw", "n_event_shrunk", "delta_n_target",
               "n_track", "track"] + [f"pc{k}_target" for k in range(1, 4)]
    ev_feats = event_feat_df.drop(columns=drop_ev, errors="ignore")

    row_df = df_rows.copy()

    # ── Optional: parquet row-level FO features from df_linec ─────────────────
    n_pq = 0
    if df_linec is not None:
        pq_skip = _v1.LEAKAGE | {
            "event_id", "sensor", "sensor_code",
            "distance", "log_distance", "target_log", "target_pgv",
            "split", "track_number", "train_type_code",
            "train_speed_kmh", "train_speed_missing",
        }
        pq_ro = [c for c in df_linec.columns
                 if pd.api.types.is_numeric_dtype(df_linec[c])
                 and c not in pq_skip
                 and not c.startswith("target_") and "pgv" not in c.lower()]
        missing = [c for c in pq_ro if c not in row_df.columns]
        if missing:
            row_df = row_df.merge(
                df_linec[["event_id", "sensor"] + missing],
                on=["event_id", "sensor"], how="left")
            n_pq = len(missing)

    # ── Event-level FO features (avoid duplicates) ─────────────────────────────
    ev_new = [c for c in ev_feats.columns
              if c != "event_id" and c not in row_df.columns]
    n_ev = 0
    if ev_new:
        row_df = row_df.merge(ev_feats[["event_id"] + ev_new],
                               on="event_id", how="left")
        n_ev = len(ev_new)

    _check_no_suffix_cols(row_df, "after_event_join")

    # ── Distance-frequency interactions ───────────────────────────────────────
    log_d = row_df["log_distance"]
    for wcol in ["wf_spectral_centroid", "wf_high_low_energy_ratio",
                 "wf_band_1_5", "wf_band_5_10", "wf_band_10_20",
                 "wf_band_20_40", "wf_band_40_80"]:
        if wcol in row_df.columns:
            row_df[f"feat_logd_x_{wcol}"] = log_d * row_df[wcol]
    for wcol in ["train_speed_kmh", "train_type_code"]:
        if wcol in row_df.columns:
            row_df[f"feat_logd_x_{wcol}"] = log_d * row_df[wcol]
    if "sensor_code" in row_df.columns:
        sc = row_df["sensor_code"]
        for wcol in ["wf_spectral_centroid", "wf_high_low_energy_ratio"]:
            if wcol in row_df.columns:
                row_df[f"feat_sc_x_{wcol}"] = sc * row_df[wcol]

    # ── Geometry features ──────────────────────────────────────────────────────
    n_geom = n_gi = 0
    if add_geometry:
        row_df = add_fo_relative_geometry_features(row_df, geom_df)
        n_geom = len(_GEOM_COLS)
        row_df = add_geometry_interactions(row_df)
        n_gi   = len([c for c in row_df.columns if c.startswith("gi_")])

    print(f"  pq={n_pq}  ev_new={n_ev}  geom={n_geom}  gi={n_gi}  "
          f"total_cols={len(row_df.columns)}")
    return row_df


# ─── PATCH 4/5: Feature column selection ──────────────────────────────────────

# ALL columns that are geometry-derived. fo_* prefix matching in _get_feat_cols
# would otherwise pull fo_y_m / fo_to_track_axis_pos into V1_ref (Issue 1 fix).
_GEOM_ALL_COLS: frozenset = frozenset(
    list(_GEOM_COLS) + [
        "fo_y_m", "active_track_y_m",
        "is_beyond_track", "is_track_side_of_fo",
        "fo_to_track_axis_pos", "acc_track_to_fo_ratio",
        "is_between_track_and_fo", "is_on_fo_line",
        "is_beyond_fo_away_from_track", "acc_to_fo_m",
        "signed_acc_from_fo", "signed_acc_from_active_track",
        "log_acc_to_active_track", "log_acc_track_to_fo_ratio",
        "acc_to_fo_over_track_to_fo", "active_track_to_fo_m",
        "sensor_x_m", "sensor_y_m", "acc_to_active_track_m",
    ]
)


def _get_feat_cols(row_df: pd.DataFrame,
                    use_sensor_code: bool = True,
                    use_geometry: bool = True,
                    use_track_number: bool = True) -> List[str]:
    explicit = [
        "pred_log_profile", "c_hat_profile", "n_hat_profile",
        "pc1_hat", "pc2_hat",
        "distance", "log_distance",
        "train_speed_kmh", "train_speed_missing", "train_type_code",
    ]
    if use_sensor_code:
        explicit.append("sensor_code")
    if use_track_number:
        explicit.append("track_number")

    geom_cols = _GEOM_COLS if use_geometry else []

    pq_cols   = [c for c in row_df.columns if c.startswith("fo_")]
    wf_cols   = [c for c in row_df.columns if c.startswith("wf_")]
    feat_cols = [c for c in row_df.columns if c.startswith("feat_")]
    gi_cols   = [c for c in row_df.columns if c.startswith("gi_")] if use_geometry else []

    # ISSUE 1 FIX: purge geometry-derived columns from every group when
    # use_geometry=False.  Without this, pq_cols picks up fo_y_m and
    # fo_to_track_axis_pos (both start with "fo_"), contaminating V1_ref.
    if not use_geometry:
        pq_cols   = [c for c in pq_cols   if c not in _GEOM_ALL_COLS]
        wf_cols   = [c for c in wf_cols   if c not in _GEOM_ALL_COLS]
        feat_cols = [c for c in feat_cols if c not in _GEOM_ALL_COLS]
        gi_cols   = []   # never include geometry interactions in no-geometry variants

    # PATCH 4: remove sensor_code-derived columns when use_sensor_code=False
    if not use_sensor_code:
        feat_cols = [c for c in feat_cols if not c.startswith("feat_sc_x_")]
        gi_cols   = [c for c in gi_cols   if "sensor_code" not in c]

    all_cols = explicit + geom_cols + feat_cols + gi_cols + wf_cols + pq_cols

    seen = set(); out = []
    for c in all_cols:
        if c in seen or c not in row_df.columns:
            continue
        cl = c.lower()
        if any(tok in cl for tok in _LEAKAGE_TOKENS):
            continue
        if not pd.api.types.is_numeric_dtype(row_df[c]):
            continue
        if c.startswith("target_") or "pgv" in cl:
            continue
        seen.add(c); out.append(c)
    return out


# ─── PATCH 6: Local residual training with extended schemes ───────────────────

def _local_grid_v2() -> List[dict]:
    grid = []
    for depth, lr, mcw, rl in product(
        [3, 4, 5], [0.01, 0.02, 0.05], [1, 5], [1, 10, 20]
    ):
        grid.append({"max_depth": depth, "learning_rate": lr,
                     "min_child_weight": mcw, "reg_lambda": rl,
                     "subsample": 0.8, "colsample_bytree": 0.8})
    return grid


def _local_weights_v2(df: pd.DataFrame, scheme: str) -> np.ndarray:
    s  = df["sensor"].values
    tp = df["target_pgv"].values
    w  = np.ones(len(df), dtype=np.float32)
    if scheme == "agg_hi":
        w += 1.0*(s=="MP4") + 2.0*(tp>4) + 3.0*(tp>8)
    elif scheme == "mp4_hi":
        w += 1.5*(s=="MP4") + 1.0*(tp>4) + 1.0*(tp>8)
    elif scheme == "between_hi":
        btw = df.get("is_between_track_and_fo",
                      pd.Series((s=="MP4").astype(float),
                                 index=df.index)).values
        w += 2.0*btw + 1.0*(tp>4) + 2.0*(tp>8)
    elif scheme == "between_mp4_hi":
        btw = df.get("is_between_track_and_fo",
                      pd.Series((s=="MP4").astype(float),
                                 index=df.index)).values
        w += 2.0*(s=="MP4") + 2.0*btw + 1.0*(tp>4) + 3.0*(tp>8)
    elif scheme == "mp4_extreme_hi":
        w += 2.0*(s=="MP4") + 2.0*(tp>4) + 4.0*(tp>8)
    return w


# ─── PATCH 7: Three selection policies per variant ────────────────────────────

def train_variant(
    row_tr: pd.DataFrame,
    row_va: pd.DataFrame,
    feat_cols: List[str],
    label: str,
    schemes: List[str],
) -> dict:
    """Train local residual; return dict with global_best, mp4_best, balanced_best."""
    valid_cols = [c for c in feat_cols
                  if c in row_tr.columns and c in row_va.columns]
    y_tr  = (row_tr["target_log"] - row_tr["pred_log_profile"]).values
    y_va  = (row_va["target_log"] - row_va["pred_log_profile"]).values
    X_tr  = row_tr[valid_cols].fillna(0.0).values.astype(np.float32)
    X_va  = row_va[valid_cols].fillna(0.0).values.astype(np.float32)
    p0_va = row_va["pred_log_profile"].values
    tgv_va= row_va["target_pgv"].values
    s_va  = row_va["sensor"].values

    grid    = _local_grid_v2()
    results = []

    for scheme in schemes:
        w_tr = _local_weights_v2(row_tr, scheme)
        for params in grid:
            m = xgb.XGBRegressor(
                **params,
                n_estimators=3000, early_stopping_rounds=50,
                eval_metric="rmse", tree_method="hist",
                verbosity=0, random_state=42)
            m.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_va, y_va)], verbose=False)
            res_va   = m.predict(X_va)
            pred_pgv = np.exp(p0_va + res_va)
            rmse     = float(np.sqrt(np.mean((pred_pgv - tgv_va)**2)))
            mp4      = s_va == "MP4"
            mp4_rmse = (float(np.sqrt(np.mean((pred_pgv[mp4] - tgv_va[mp4])**2)))
                        if mp4.any() else 999.0)
            mp4hi    = mp4 & (tgv_va > 4)
            mp4hi_rmse = (float(np.sqrt(np.mean((pred_pgv[mp4hi] - tgv_va[mp4hi])**2)))
                          if mp4hi.any() else 999.0)
            results.append({
                "model": m, "scheme": scheme,
                "rmse_pgv": rmse, "mp4_rmse": mp4_rmse,
                "mp4_hi_rmse": mp4hi_rmse,
                "balanced": rmse + 0.15*mp4_rmse,
                "va_res_pred": res_va,
            })

    global_best   = min(results, key=lambda r: (r["rmse_pgv"], r["mp4_rmse"], r["mp4_hi_rmse"]))
    mp4_best_r    = min(results, key=lambda r: (r["mp4_rmse"], r["rmse_pgv"]))
    balanced_best = min(results, key=lambda r: (r["balanced"], r["mp4_rmse"]))

    print(f"[{label}] ({len(results)} fits)")
    print(f"  global_best:   scheme={global_best['scheme']:22s}  "
          f"RMSE={global_best['rmse_pgv']:.4f}  MP4={global_best['mp4_rmse']:.4f}")
    print(f"  mp4_best:      scheme={mp4_best_r['scheme']:22s}  "
          f"RMSE={mp4_best_r['rmse_pgv']:.4f}  MP4={mp4_best_r['mp4_rmse']:.4f}")
    print(f"  balanced_best: scheme={balanced_best['scheme']:22s}  "
          f"RMSE={balanced_best['rmse_pgv']:.4f}  MP4={balanced_best['mp4_rmse']:.4f}")

    def _pack(r):
        return (r["model"], r["rmse_pgv"], r["mp4_rmse"], r["scheme"], valid_cols)

    return {
        "global_best":   _pack(global_best),
        "mp4_best":      _pack(mp4_best_r),
        "balanced_best": _pack(balanced_best),
        "n_fits": len(results), "valid_cols": valid_cols,
    }


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics_v2(
    pred_log: np.ndarray, true_log: np.ndarray,
    sensors: np.ndarray, event_ids: np.ndarray, label: str,
) -> dict:
    pp = np.exp(pred_log); tp = np.exp(true_log)
    rmse = lambda a, b: float(np.sqrt(np.mean((a-b)**2)))
    bias = lambda a, b: float(np.mean(a-b))
    r2   = lambda pl, tl: float(
        1 - np.sum((tl-pl)**2) / max(np.sum((tl-tl.mean())**2), 1e-12))
    m = {"model": label,
         "rmse_log": rmse(pred_log, true_log), "rmse_pgv": rmse(pp, tp),
         "mae_pgv":  float(np.mean(np.abs(pp-tp))),
         "r2_log":   r2(pred_log, true_log),   "bias_pgv": bias(pp, tp)}
    m["per_sensor"] = {}
    for s in SENSOR_ORDER:
        msk = sensors == s
        if msk.any():
            m["per_sensor"][s] = {
                "rmse_pgv": rmse(pp[msk], tp[msk]),
                "rmse_log": rmse(pred_log[msk], true_log[msk]),
                "bias_pgv": bias(pp[msk], tp[msk]),
                "bias_log": bias(pred_log[msk], true_log[msk]),
            }
    for thr in [4.0, 8.0]:
        hi = tp > thr
        if hi.any():
            m[f"pgv_gt{int(thr)}"] = {"n": int(hi.sum()), "rmse_pgv": rmse(pp[hi], tp[hi])}
        mp4hi = (sensors == "MP4") & hi
        if mp4hi.any():
            m[f"mp4_pgv_gt{int(thr)}"] = {
                "n": int(mp4hi.sum()), "rmse_pgv": rmse(pp[mp4hi], tp[mp4hi])}
    viol = tot = 0
    df_tmp = pd.DataFrame({"eid": event_ids, "s": sensors, "p": pred_log})
    for eid, grp in df_tmp.groupby("eid"):
        if len(grp) != 5: continue
        row = grp.set_index("s").reindex(SENSOR_ORDER).dropna()
        vals = row["p"].values
        for k in range(len(vals)-1):
            tot += 1; viol += int(vals[k] < vals[k+1])
    m["mono_viol_rate"] = viol / tot if tot else 0.0
    return m


# ─── PATCH 9: Geometry zone metrics ───────────────────────────────────────────

def compute_geometry_zone_metrics(
    row_df: pd.DataFrame, pred_col: str, label: str,
) -> List[dict]:
    if pred_col not in row_df.columns:
        return []
    valid = row_df.dropna(subset=[pred_col])
    pl = valid[pred_col].values; tl = valid["target_log"].values
    pp = np.exp(pl); tp = np.exp(tl)
    sv = valid["sensor"].values
    rmse = lambda a, b: float(np.sqrt(np.mean((a-b)**2)))
    bias = lambda a, b: float(np.mean(a-b))
    rows = []

    def _row(zone, msk):
        if msk.sum() < 2: return
        rows.append({"model": label, "zone": zone, "n": int(msk.sum()),
                     "rmse_pgv": rmse(pp[msk], tp[msk]),
                     "rmse_log": rmse(pl[msk], tl[msk]),
                     "bias_pgv": bias(pp[msk], tp[msk]),
                     "bias_log": bias(pl[msk], tl[msk])})

    for col, name in [("is_between_track_and_fo", "between"),
                       ("is_on_fo_line", "on_fo_line"),
                       ("is_beyond_fo_away_from_track", "beyond_fo_away"),
                       ("is_beyond_track", "beyond_track")]:
        if col in valid.columns:
            _row(name, valid[col].values == 1)

    if "fo_to_track_axis_pos" in valid.columns:
        ax = valid["fo_to_track_axis_pos"].values
        for lo, hi, nm in [(-99,0,"axis_lt0"),(0,0.3,"axis_near0"),
                             (0.3,1.0,"axis_0to1"),(1.0,99,"axis_gt1")]:
            _row(nm, (ax>=lo) & (ax<hi))

    for ss in SENSOR_ORDER:
        _row(f"sensor_{ss}", sv==ss)
    for thr in [4.0, 8.0]:
        msk = (sv=="MP4") & (tp>thr)
        _row(f"MP4_pgv_gt{int(thr)}", msk)
    if "is_between_track_and_fo" in valid.columns:
        btw = valid["is_between_track_and_fo"].values == 1
        for thr in [4.0, 8.0]:
            _row(f"between_pgv_gt{int(thr)}", btw & (tp>thr))
    return rows


# ─── Plots ────────────────────────────────────────────────────────────────────

def _save_fig(path: Path, dpi: int = 150) -> None:
    plt.tight_layout(); plt.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close()


def plot_per_sensor_rmse(metrics_dict: Dict[str, dict], out: Path):
    models = list(metrics_dict.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SENSOR_ORDER)); w = 0.8 / len(models)
    for i, (lbl, m) in enumerate(metrics_dict.items()):
        rmses = [m["per_sensor"].get(s, {}).get("rmse_pgv", np.nan) for s in SENSOR_ORDER]
        ax.bar(x + i*w, rmses, w, label=lbl)
    ax.set_xticks(x + w*(len(models)-1)/2); ax.set_xticklabels(SENSOR_ORDER)
    ax.set_ylabel("RMSE(PGV) [mm/s]")
    ax.set_title("Per-sensor RMSE — v1 vs v2_geometry"); ax.legend(fontsize=7)
    _save_fig(out)


def plot_residual_vs_axis_pos_v1_vs_v2(row_df, v1_col, v2_col, out):
    if "fo_to_track_axis_pos" not in row_df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, pcol, title in [(axes[0], v1_col, "V1_ref"),
                              (axes[1], v2_col, "V2_geom")]:
        if pcol not in row_df.columns:
            ax.set_title(f"{title} (no data)"); continue
        valid = row_df.dropna(subset=[pcol, "fo_to_track_axis_pos"])
        res   = valid["target_log"] - valid[pcol]
        axis  = valid["fo_to_track_axis_pos"]
        for s in SENSOR_ORDER:
            msk = valid["sensor"] == s
            ax.scatter(axis[msk], res[msk], alpha=0.25, s=6, label=s)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.axvline(0, color="gray", lw=0.6, ls=":"); ax.axvline(1, color="gray", lw=0.6, ls=":")
        ax.set_xlabel("fo_to_track_axis_pos"); ax.set_ylabel("Residual (log)")
        ax.set_title(f"{title}"); ax.legend(fontsize=7)
    plt.suptitle("Residual vs FO-relative axis position", fontsize=10)
    _save_fig(out)


def plot_residual_box_by_zone_v1_vs_v2(row_df, v1_col, v2_col, out):
    zones = [("is_between_track_and_fo", "between"),
             ("is_on_fo_line", "on FO"),
             ("is_beyond_fo_away_from_track", "beyond FO")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, pcol, title in [(axes[0], v1_col, "V1_ref"),
                              (axes[1], v2_col, "V2_geom")]:
        if pcol not in row_df.columns:
            ax.set_title(f"{title} (no data)"); continue
        valid = row_df.dropna(subset=[pcol])
        res   = valid["target_log"] - valid[pcol]
        data, labs = [], []
        for col, lbl in zones:
            if col not in valid.columns: continue
            msk = valid[col].values == 1
            if msk.sum() > 2:
                data.append(res[msk].values); labs.append(f"{lbl}\n(n={msk.sum()})")
        if data:
            ax.boxplot(data, labels=labs, patch_artist=True,
                       medianprops={"color": "red"})
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_ylabel("Residual (log)"); ax.set_title(f"Zone boxplot — {title}")
    _save_fig(out)


def plot_mp4_profiles_v1_vs_v2(row_df, v1_col, v2_col, n, out):
    if v1_col not in row_df.columns or v2_col not in row_df.columns:
        return
    top_e = (row_df[row_df["sensor"]=="MP4"]
             .groupby("event_id")["target_pgv"].max().nlargest(n).index)
    nrows = max(1, (n+3)//4)
    fig, axes = plt.subplots(nrows, 4, figsize=(14, 3*nrows))
    for ax, eid in zip(np.array(axes).flat, top_e):
        ev = row_df[row_df["event_id"]==eid].sort_values("distance")
        ax.plot(ev["distance"], np.exp(ev["target_log"]), "ko-", ms=4, label="true")
        ax.plot(ev["distance"], np.exp(ev[v1_col].fillna(np.nan)), "b.:", ms=4, label="v1")
        ax.plot(ev["distance"], np.exp(ev[v2_col].fillna(np.nan)), "r^--", ms=4, label="v2")
        ax.set_title(str(eid)[:12], fontsize=7)
        ax.set_xlabel("dist(m)", fontsize=7); ax.set_ylabel("PGV", fontsize=7)
    np.array(axes).flat[0].legend(fontsize=6)
    plt.suptitle("High-PGV MP4 events — v1 vs v2", fontsize=10)
    _save_fig(out)


def plot_scatter(true_pgv, pred_pgv, label, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_pgv, pred_pgv, alpha=0.3, s=8)
    lim = (0, max(true_pgv.max(), pred_pgv.max()) * 1.05)
    ax.plot(lim, lim, "r--", lw=1.2); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True PGV"); ax.set_ylabel("Pred PGV")
    ax.set_title(f"{label} — measured vs predicted")
    _save_fig(out_path)


def plot_feat_imp(model, feat_cols, title, out_path, top_n=30):
    if len(feat_cols) != len(model.feature_importances_):
        return
    imp = pd.DataFrame({"feature": feat_cols,
                         "importance": model.feature_importances_}
                       ).nlargest(top_n, "importance")
    fig, ax = plt.subplots(figsize=(8, max(4, top_n*0.25)))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1])
    ax.set_xlabel("Importance"); ax.set_title(title)
    _save_fig(out_path)


def plot_geometry_table(geom_df, out):
    rows = []
    fo_y = 0.0
    for s in SENSOR_ORDER:
        if s not in geom_df.index: continue
        sy = float(geom_df.loc[s, "sensor_y_m"])
        for tn, ty in [(1, float(geom_df.loc[s, "track1_y_m"])),
                        (2, float(geom_df.loc[s, "track2_y_m"]))]:
            ap = (sy-fo_y)/(ty-fo_y)
            rows.append({"S": s, "T": tn, "y": sy, "ty": ty,
                          "axis": round(ap,4),
                          "d_track": round(abs(sy-ty),2),
                          "d_fo": round(abs(sy-fo_y),2),
                          "ratio": round(abs(sy-ty)/abs(ty-fo_y),3)})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, len(df)*0.4+1.5))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))
    plt.title("Holten Line-C FO-Relative Geometry", pad=10)
    _save_fig(out)


# ─── V1 pipeline (PATCH 1: v1 artifact reuse; PATCH 2: PC3 disabled) ──────────

def run_v1_pipeline(v1_artifacts: Optional[dict] = None, smoke: bool = False) -> dict:
    """
    Run v1 profile reconstruction pipeline.
    PATCH 1: if v1_artifacts has predictions_test, uses saved test predictions.
    PATCH 2: PC3 is disabled; n_comp_use = 2 always.
    """
    print("\n" + "=" * 70)
    print("Running v1 profile reconstruction pipeline")
    print("=" * 70)

    df_linec = _v1.load_linec_parquet()
    wf_array, event_map, _ = _v1.load_waveforms()
    p3_dir   = _v1.find_p3_dir()
    split_df = _v1.get_split_labels(df_linec, p3_dir)
    df_linec = df_linec.merge(split_df, on="event_id", how="inner")

    if smoke:
        ev_s = sorted(df_linec["event_id"].unique())[:30]
        df_linec = df_linec[df_linec["event_id"].isin(ev_s)].copy()

    all_eids = df_linec["event_id"].unique().tolist()
    try:
        wf_feat_df = _v1.build_waveform_feature_df(wf_array, event_map, all_eids)
    except Exception as exc:
        print(f"[WARN] Waveform features failed: {exc}")
        wf_feat_df = pd.DataFrame({"event_id": all_eids})

    pq_ev_df  = _v1.build_parquet_event_features(df_linec)
    ev_fd     = pq_ev_df.merge(wf_feat_df, on="event_id", how="left")
    mpgv_ev   = (df_linec.groupby("event_id")["target_pgv"].max()
                 .reset_index().rename(columns={"target_pgv": "max_pgv_event"}))
    ev_fd     = ev_fd.merge(mpgv_ev,   on="event_id", how="left")
    ev_fd     = ev_fd.merge(split_df,  on="event_id", how="left")

    skip_ev  = _v1.LEAKAGE | {"event_id","split","max_pgv_event","track","track_number",
                               "train_type_code","train_speed_kmh","train_speed_missing"}
    meta_ev  = ["train_speed_kmh","train_speed_missing","train_type_code","track_number"]
    fo_ev_fc = [c for c in ev_fd.columns
                if c not in skip_ev
                and pd.api.types.is_numeric_dtype(ev_fd[c])
                and not c.startswith("target_") and "pgv" not in c.lower()
                and not c.startswith("pc") and c != "c_target_event"
                and "n_event" not in c and "delta_n" not in c]
    fo_ev_fc += [c for c in meta_ev if c in ev_fd.columns and c not in fo_ev_fc]

    target_df, resid_df = _v1.compute_physics_targets(df_linec)
    pca, pc_score_df, pca_ve = _v1.fit_pca_residuals(
        target_df, resid_df, split_df, n_comp=PCA_N_COMP)
    ev_full = (ev_fd
               .merge(target_df,   on="event_id", how="inner")
               .merge(pc_score_df, on="event_id", how="inner"))

    ev_tr = ev_full[ev_full["split"]=="train"].copy()
    ev_va = ev_full[ev_full["split"]=="val"].copy()
    ev_te = ev_full[ev_full["split"]=="test"].copy()
    fo_valid = [c for c in fo_ev_fc if c in ev_full.columns]

    X_tr = ev_tr[fo_valid].fillna(0.0).values.astype(np.float32)
    X_va = ev_va[fo_valid].fillna(0.0).values.astype(np.float32)
    X_te = ev_te[fo_valid].fillna(0.0).values.astype(np.float32)

    w_tr       = _v1._high_pgv_weights(ev_tr)
    df_va_rows = df_linec[df_linec["split"]=="val"].copy()
    df_te_rows = df_linec[df_linec["split"]=="test"].copy()
    df_tr_rows = df_linec[df_linec["split"]=="train"].copy()
    zero_pc    = np.zeros((len(ev_va), 2), dtype=np.float32)

    def _top_k(Xtr, ytr, Xva, yva, wtr, lbl, k=5):
        cands = []
        for p in _v1._event_grid():
            m = _v1._xgbr_event(p)
            m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False)
            vp = m.predict(Xva)
            cands.append((float(np.sqrt(np.mean((vp-yva)**2))), m, vp))
        cands.sort(key=lambda x: x[0])
        print(f"  [{lbl}] best RMSE={cands[0][0]:.5f}")
        return cands[:k]

    print("\n[v1_pipe] C model...")
    c_c  = _top_k(X_tr, ev_tr["c_target_event"].values,
                   X_va, ev_va["c_target_event"].values, w_tr, "C", 3)
    m_c, va_c = c_c[0][1], c_c[0][2]
    te_c = m_c.predict(X_te)
    r2_c = float(1 - np.sum((va_c - ev_va["c_target_event"].values)**2) /
                 max(np.sum((ev_va["c_target_event"].values -
                              ev_va["c_target_event"].values.mean())**2), 1e-12))
    print(f"  c_hat R²(val)={r2_c:.4f}")

    print("\n[v1_pipe] N model (profile-aware)...")
    n_c = _top_k(X_tr, ev_tr["delta_n_target"].values,
                  X_va, ev_va["delta_n_target"].values, w_tr, "N", 5)
    b=0; bp=1e9
    for k,(_, mn, vnc) in enumerate(n_c):
        rp = _v1._profile_rmse_pgv(va_c, vnc, zero_pc, pca, ev_va, df_va_rows)
        if rp < bp: bp=rp; b=k
    m_n, va_n = n_c[b][1], n_c[b][2]
    te_n = m_n.predict(X_te)

    print("\n[v1_pipe] PC1 (profile-aware)...")
    pc1_c = _top_k(X_tr, ev_tr["pc1_target"].values,
                    X_va, ev_va["pc1_target"].values, w_tr, "PC1", 5)
    z2=np.zeros(len(ev_va),dtype=np.float32); b=0; bp=1e9
    for k,(_, mp, vp) in enumerate(pc1_c):
        rp = _v1._profile_rmse_pgv(va_c, va_n,
                                    np.column_stack([vp, z2]), pca, ev_va, df_va_rows)
        if rp < bp: bp=rp; b=k
    m_pc1, va_pc1 = pc1_c[b][1], pc1_c[b][2]
    te_pc1 = m_pc1.predict(X_te)

    print("\n[v1_pipe] PC2 (profile-aware)...")
    pc2_c = _top_k(X_tr, ev_tr["pc2_target"].values,
                    X_va, ev_va["pc2_target"].values, w_tr, "PC2", 5)
    b=0; bp=1e9
    for k,(_, mp, vp) in enumerate(pc2_c):
        rp = _v1._profile_rmse_pgv(va_c, va_n,
                                    np.column_stack([va_pc1, vp]), pca, ev_va, df_va_rows)
        if rp < bp: bp=rp; b=k
    m_pc2, va_pc2 = pc2_c[b][1], pc2_c[b][2]
    te_pc2 = m_pc2.predict(X_te)

    # PATCH 2: PC3 disabled
    n_comp_use = 2

    def _pred_dict(df_ev, c_a, n_a, p1_a, p2_a):
        c_by  = dict(zip(df_ev["event_id"].values, c_a))
        dn_by = dict(zip(df_ev["event_id"].values, n_a))
        pc_by = {eid: np.array([p1_a[i], p2_a[i]])
                 for i, eid in enumerate(df_ev["event_id"].values)}
        return c_by, dn_by, pc_by

    c_va_d,  dn_va_d,  pc_va_d  = _pred_dict(ev_va, va_c,  va_n,  va_pc1,  va_pc2)
    c_te_d,  dn_te_d,  pc_te_d  = _pred_dict(ev_te, te_c,  te_n,  te_pc1,  te_pc2)

    def _attach(df_rows, c_by, dn_by, pc_by):
        pred = _v1.reconstruct_profile_predictions(
            df_rows, df_rows["event_id"].unique(),
            c_by, dn_by, pc_by, pca, n_comp_use=n_comp_use)
        df   = df_rows.copy()
        df["pred_log_profile"] = pred
        df["c_hat_profile"]    = df["event_id"].map(c_by).astype(float)
        tb   = df.groupby("event_id")["track_number"].first().astype(int)
        df["n_hat_profile"] = df["event_id"].map(
            lambda e: float(np.clip(
                N_TRACK.get(int(tb.get(e,1)),1.0)+dn_by.get(e,0.), *N_CLIP))
            if e in dn_by else np.nan)
        df["pc1_hat"] = df["event_id"].map({e: v[0] for e,v in pc_by.items()})
        df["pc2_hat"] = df["event_id"].map(
            {e: v[1] if len(v)>1 else 0. for e,v in pc_by.items()})
        return df

    df_va_rows = _attach(df_va_rows, c_va_d, dn_va_d, pc_va_d)

    # PATCH 1: use saved test predictions if available
    used_saved_test = False
    if v1_artifacts and v1_artifacts.get("can_reuse_test"):
        pv1  = v1_artifacts["predictions_test"]
        base = df_linec[df_linec["split"]=="test"].copy()
        base = base.merge(
            pv1[["event_id","sensor","pred_log_profile",
                  "c_hat_profile","n_hat_profile","pc1_hat","pc2_hat"]],
            on=["event_id","sensor"], how="left", suffixes=("","_s"))
        for col in ["pred_log_profile","c_hat_profile","n_hat_profile","pc1_hat","pc2_hat"]:
            sc = f"{col}_s"
            if sc in base.columns:
                base[col] = base[sc].combine_first(base.get(col, np.nan))
                base.drop(columns=[sc], inplace=True)
        df_te_rows    = base
        used_saved_test = True
        print("[v1_pipe] Test: using saved v1 predictions  ✓")
    else:
        df_te_rows = _attach(df_te_rows, c_te_d, dn_te_d, pc_te_d)
        print("[WARNING] v2_geometry is rerunning v1 profile reconstruction; "
              "V1_ref may not exactly reproduce saved v1.")

    # OOF for train
    print("\n[v1_pipe] OOF KFold (5-fold)...")
    kf  = KFold(n_splits=5, shuffle=True, random_state=42)
    nev = len(ev_tr)
    oof_c=np.zeros(nev); oof_n=np.zeros(nev)
    oof_p1=np.zeros(nev); oof_p2=np.zeros(nev)
    fp = {"max_depth":3,"learning_rate":0.02,"min_child_weight":5,
          "reg_lambda":10,"subsample":0.8,"colsample_bytree":0.8}
    for fi, (ftr, fva) in enumerate(kf.split(X_tr)):
        def _ff(y):
            mf = _v1._xgbr_event(fp)
            mf.fit(X_tr[ftr], y[ftr], sample_weight=w_tr[ftr],
                   eval_set=[(X_tr[fva], y[fva])], verbose=False)
            return mf.predict(X_tr[fva])
        oof_c[fva]  = _ff(ev_tr["c_target_event"].values)
        oof_n[fva]  = _ff(ev_tr["delta_n_target"].values)
        oof_p1[fva] = _ff(ev_tr["pc1_target"].values)
        oof_p2[fva] = _ff(ev_tr["pc2_target"].values)
        print(f"  fold {fi+1}/5  done")

    c_tr_d, dn_tr_d, pc_tr_d = _pred_dict(ev_tr, oof_c, oof_n, oof_p1, oof_p2)
    df_tr_rows = _attach(df_tr_rows, c_tr_d, dn_tr_d, pc_tr_d)

    vp_rmse = _v1._profile_rmse_pgv(
        va_c, va_n, np.column_stack([va_pc1, va_pc2]), pca, ev_va, df_va_rows)
    print(f"\n[v1_pipe] Val profile RMSE(PGV) = {vp_rmse:.4f}")

    return {
        "df_linec": df_linec, "split_df": split_df,
        "event_feat_df": ev_fd, "pca": pca, "pca_ve": pca_ve,
        "ev_tr": ev_tr, "ev_va": ev_va, "ev_te": ev_te,
        "df_tr_rows": df_tr_rows, "df_va_rows": df_va_rows, "df_te_rows": df_te_rows,
        "fo_valid": fo_valid, "n_comp_use": n_comp_use, "r2_c_val": r2_c,
        "used_saved_test": used_saved_test,
    }


# ─── PATCH 11: Interpretation ─────────────────────────────────────────────────

def print_interpretation(
    vt_metrics: Dict[str, dict],
    v1_saved_metrics: Optional[dict],
    geom_zone_rows: List[dict],
    feat_cols_dict: Dict[str, List[str]],
    models_dict: Dict[str, xgb.XGBRegressor],
) -> None:
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    def _mp4(m): return m.get("per_sensor",{}).get("MP4",{}).get("rmse_pgv",None)
    def _rmse(m): return m.get("rmse_pgv",None)

    v1r_mp4  = _mp4(vt_metrics.get("V1_ref",{}))
    v2g_mp4  = _mp4(vt_metrics.get("V2_geom",{}))
    v2ns_mp4 = _mp4(vt_metrics.get("V2_geom_no_sc",{}))
    v2nt_mp4 = _mp4(vt_metrics.get("V2_geom_no_sensor_no_trackcat",{}))
    v1r_rmse = _rmse(vt_metrics.get("V1_ref",{}))
    v2g_rmse = _rmse(vt_metrics.get("V2_geom",{}))

    sv1 = v1_saved_metrics.get("rmse_pgv",2.1882) if v1_saved_metrics else 2.1882
    sv1mp4 = (_mp4(v1_saved_metrics) or 4.5592) if v1_saved_metrics else 4.5592

    print("\n── RMSE(PGV) ─────────────────────────────────────────────────────────")
    for nm, val in [("P3_corrected_n",2.4006),("PXGBR-R2",2.2718),
                     ("PXGBR_ens_top5",2.2473),("FO-v1 final (saved)",sv1)]:
        actual = vt_metrics.get(nm,{}).get("rmse_pgv",val)
        print(f"  {nm:45s}  {actual:.4f}")
    for vname in VARIANT_CONFIGS:
        m = vt_metrics.get(vname,{})
        if m: print(f"  {vname:45s}  {m.get('rmse_pgv','?')}")

    print("\n── MP4 RMSE(PGV) ─────────────────────────────────────────────────────")
    for nm, val in [("P3_corrected_n",5.0185),("PXGBR-R2",4.7389),
                     ("PXGBR_ens_top5",4.6680),("FO-v1 final (saved)",sv1mp4)]:
        actual = _mp4(vt_metrics.get(nm,{})) or val
        print(f"  {nm:45s}  {actual:.4f}")
    for vname in VARIANT_CONFIGS:
        m = _mp4(vt_metrics.get(vname,{}))
        if m: print(f"  {vname:45s}  {m:.4f}")

    print("\n── Geometry zone (V2_geom global_best) ────────────────────────────────")
    for row in geom_zone_rows:
        if row["model"]=="V2_geom" and row["zone"] in [
            "between","on_fo_line","beyond_fo_away","sensor_MP4"]:
            print(f"  {row['zone']:30s}  n={row['n']:5d}  "
                  f"rmse_pgv={row['rmse_pgv']:.4f}  bias={row['bias_pgv']:+.4f}")

    print("\n── Geometry feature importance ────────────────────────────────────────")
    for vname, fc_list in feat_cols_dict.items():
        if vname not in models_dict: continue
        mod = models_dict[vname]
        if len(fc_list) != len(mod.feature_importances_): continue
        imp = pd.Series(mod.feature_importances_, index=fc_list)
        gi  = imp[[c for c in fc_list if
                    any(k in c for k in ["fo_to_track","acc_track","between","gi_"])]]
        frac = gi.sum() / (imp.sum() + 1e-12)
        top3 = gi.nlargest(3).index.tolist()
        print(f"  {vname}: geom_frac={frac:.3f}  top={top3}")

    print("\n── Conclusions ────────────────────────────────────────────────────────")
    if v2g_mp4 is not None and v1r_mp4 is not None:
        delta = v2g_mp4 - v1r_mp4
        if delta < -0.05:
            print(f"→ V2 improves MP4 by {-delta:.3f}: FO-relative geometry captures "
                  "interpolation from FO cable back toward the source (track).")
        elif delta < 0:
            print(f"→ V2 modestly improves MP4 ({delta:+.3f}): geometry adds signal "
                  "but sensor_code already captured most of the effect.")
        else:
            print(f"→ V2 does not improve MP4 ({delta:+.3f}): profile model "
                  "is the main bottleneck. Keep v1 final as best model.")

    if v2g_rmse and v1r_rmse:
        if v2g_rmse < v1r_rmse and v2g_mp4 and v2g_mp4 < (v1r_mp4 or 999):
            print("→ V2 improves both global RMSE and MP4: geometry patch is effective.")
        elif v2g_rmse > v1r_rmse + 0.03:
            print("→ V2 worsens global RMSE: geometry interactions may add noise. "
                  "Prefer v1 features for overall accuracy.")

    if v2g_mp4 and v2ns_mp4:
        d = v2ns_mp4 - v2g_mp4
        if abs(d) < 0.10:
            print(f"→ V2_no_sc ≈ V2 (Δ={d:+.3f}): physical geometry can replace "
                  "sensor_code (generalizes to new sensors).")
        else:
            print(f"→ V2_no_sc degrades MP4 by {d:+.3f}: sensor-specific calibration "
                  "effects remain important beyond geometry.")
    if v2nt_mp4 and v2ns_mp4:
        d = v2nt_mp4 - v2ns_mp4
        if abs(d) < 0.10:
            print(f"→ Removing track_number makes little difference (Δ={d:+.3f}): "
                  "active_track_y_m / acc_to_active_track suffice.")
        else:
            print(f"→ Removing track_number degrades by {d:+.3f}: track identity "
                  "carries additional information beyond physical coordinates.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",  action="store_true", help="30-event smoke test")
    parser.add_argument("--v1_dir", type=str, default=None,
                        help="Path to v1 output dir (default: auto-discover)")
    args = parser.parse_args()

    out_dir = make_output_dir(MODELS_ROOT)
    print(f"\nOutput dir: {out_dir}")
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git","rev-parse","--short","HEAD"], text=True).strip()
    except Exception:
        git_hash = "unknown"
    print(f"Git commit: {git_hash}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Geometry (PATCH 8: assertions)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70); print("STEP 1 — Site geometry"); print("=" * 70)
    site_cfg = load_site_config()
    geom_df  = build_sensor_geometry_table(site_cfg)
    print_geometry_diagnostic(geom_df)
    assert_geometry_values(geom_df)   # PATCH 8

    diag_rows = []
    fo_y = 0.0
    for s in SENSOR_ORDER:
        if s not in geom_df.index: continue
        sy = float(geom_df.loc[s,"sensor_y_m"])
        for tn, ty in [(1, float(geom_df.loc[s,"track1_y_m"])),
                        (2, float(geom_df.loc[s,"track2_y_m"]))]:
            ap = (sy-fo_y)/(ty-fo_y)
            diag_rows.append({
                "sensor": s, "track_number": tn,
                "sensor_x_m": float(geom_df.loc[s,"sensor_x_m"]),
                "sensor_y_m": sy, "active_track_y_m": ty,
                "active_track_to_fo_m": abs(ty-fo_y),
                "acc_to_active_track_m": round(abs(sy-ty),2),
                "acc_to_fo_m": round(abs(sy-fo_y),2),
                "signed_acc_from_fo": round(sy-fo_y,2),
                "fo_to_track_axis_pos": round(ap,4),
                "acc_track_to_fo_ratio": round(abs(sy-ty)/abs(ty-fo_y),3),
                "is_between_track_and_fo": int(0<ap<1),
                "is_on_fo_line": int(abs(sy-fo_y)<=0.75),
                "is_beyond_fo_away_from_track": int(ap<0),
            })
    pd.DataFrame(diag_rows).to_csv(out_dir/"geometry_diagnostic_table.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Load v1 artifacts (PATCH 1)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70); print("STEP 2 — v1 artifacts"); print("=" * 70)
    v1_dir_path = Path(args.v1_dir) if args.v1_dir else find_v1_output()
    if v1_dir_path:
        print(f"[v1_dir] {v1_dir_path}")
        v1_arts = load_v1_artifacts(v1_dir_path)
    else:
        print("[v1_dir] None found — computing fresh")
        v1_arts = None

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — v1 pipeline
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70); print("STEP 3 — v1 profile reconstruction"); print("=" * 70)
    v1 = run_v1_pipeline(v1_artifacts=v1_arts, smoke=args.smoke)
    df_tr_rows    = v1["df_tr_rows"]
    df_va_rows    = v1["df_va_rows"]
    df_te_rows    = v1["df_te_rows"]
    df_linec      = v1["df_linec"]
    event_feat_df = v1["event_feat_df"]
    pca           = v1["pca"]
    event_feat_df.to_parquet(out_dir/"event_features.parquet", index=False)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — Row features (PATCH 3)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70); print("STEP 4 — Row feature matrices"); print("=" * 70)
    print("[tr]")
    row_tr = _build_row_features_v2(df_tr_rows, event_feat_df, geom_df,
                                     df_linec, add_geometry=True)
    print("[va]")
    row_va = _build_row_features_v2(df_va_rows, event_feat_df, geom_df,
                                     df_linec, add_geometry=True)
    print("[te]")
    row_te = _build_row_features_v2(df_te_rows, event_feat_df, geom_df,
                                     df_linec, add_geometry=True)
    for split, rd in [("tr", row_tr), ("va", row_va), ("te", row_te)]:
        rd.dropna(subset=["pred_log_profile"], inplace=True)
    print(f"[rows] train={len(row_tr)}  val={len(row_va)}  test={len(row_te)}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Feature columns + leakage (PATCH 4/5/12)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70); print("STEP 5 — Feature columns + leakage"); print("=" * 70)
    fc_by_variant: Dict[str, List[str]] = {}
    for vname, vcfg in VARIANT_CONFIGS.items():
        fc = _get_feat_cols(
            row_tr,
            use_sensor_code=vcfg["use_sensor_code"],
            use_geometry=vcfg["use_geometry"],
            use_track_number=vcfg["use_track_number"],
        )
        _v1.assert_no_leakage(fc, vname)
        # Sensor-code and track-number ablation assertions
        if not vcfg["use_sensor_code"]:
            bad = [c for c in fc if "sensor_code" in c or c.startswith("feat_sc_x_")]
            if bad:
                raise ValueError(f"[{vname}] sensor_code leaked: {bad}")
        if not vcfg["use_track_number"]:
            if "track_number" in fc:
                raise ValueError(f"[{vname}] track_number leaked")
        # ISSUE 1 assertion: no-geometry variants must contain zero geometry columns
        if not vcfg["use_geometry"]:
            bad_geom = [
                c for c in fc
                if c in _GEOM_ALL_COLS
                or c.startswith("gi_")
                or "fo_to_track_axis" in c
                or "acc_track_to_fo" in c
                or "between_track_and_fo" in c
                or "beyond_fo" in c
            ]
            if bad_geom:
                raise ValueError(
                    f"[{vname}] geometry leaked into no-geometry baseline: {bad_geom}"
                )
            n_geom_in_v1ref = len([c for c in fc if c in _GEOM_ALL_COLS])
            print(f"  [{vname}] geometry features present: {n_geom_in_v1ref}  ✓")
        fc_by_variant[vname] = fc
        (out_dir / f"feature_columns_{vname}.txt").write_text("\n".join(fc))

    n_rm_sc = len(fc_by_variant["V2_geom"]) - len(fc_by_variant["V2_geom_no_sc"])
    n_rm_nt = (len(fc_by_variant["V2_geom_no_sc"])
               - len(fc_by_variant["V2_geom_no_sensor_no_trackcat"]))
    print(f"\n  V1_ref:                      {len(fc_by_variant['V1_ref'])} features")
    print(f"  V2_geom:                     {len(fc_by_variant['V2_geom'])} features")
    print(f"  V2_geom_no_sc:               {len(fc_by_variant['V2_geom_no_sc'])} features  "
          f"(removed {n_rm_sc} sensor-code-derived)")
    print(f"  V2_geom_no_sensor_no_trackcat: {len(fc_by_variant['V2_geom_no_sensor_no_trackcat'])} features  "
          f"(also removed {n_rm_nt} track_number)")
    _v1.write_leakage_audit(fc_by_variant["V2_geom"], out_dir/"leakage_audit_geometry.txt")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6 — Train variants (PATCH 6/7)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70); print("STEP 6 — Train local residual variants"); print("=" * 70)
    variant_results: Dict[str, dict] = {}
    for vname, vcfg in VARIANT_CONFIGS.items():
        print(f"\n{'─'*50}\n{vname}\n{'─'*50}")
        res = train_variant(row_tr, row_va, fc_by_variant[vname],
                             vname, vcfg["schemes"])
        variant_results[vname] = res

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 7 — Test predictions
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70); print("STEP 7 — Test predictions"); print("=" * 70)

    def _predict_test(model, fc, rte):
        vc = [c for c in fc if c in rte.columns]
        X  = rte[vc].fillna(0.0).values.astype(np.float32)
        r  = model.predict(X)
        final = rte["pred_log_profile"].values + r
        mono  = _v1.apply_monotonic(rte.assign(_tmp=final), "_tmp").values
        return final, mono

    models_gb: Dict[str, Tuple] = {}  # for feat imp / interpretation
    for vname, res in variant_results.items():
        for policy in ["global_best", "mp4_best", "balanced_best"]:
            model, _, _, scheme, fc_used = res[policy]
            cf = f"pred_{vname}_{policy}"
            cm = f"{cf}_mono"
            final, mono = _predict_test(model, fc_used, row_te)
            row_te[cf] = final; row_te[cm] = mono
        models_gb[vname] = (res["global_best"][0], res["global_best"][4])

    row_te["pred_log_profile_mono"] = _v1.apply_monotonic(row_te, "pred_log_profile")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 8 — Metrics (PATCH 9: geometry zone metrics)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70); print("STEP 8 — Metrics"); print("=" * 70)
    s_te  = row_te["sensor"].values
    e_te  = row_te["event_id"].values
    tl_te = row_te["target_log"].values
    all_metrics: List[dict] = []
    geom_zone_all: List[dict] = []
    vt_metrics: Dict[str, dict] = {}

    def _eval(pred_col, label):
        if pred_col not in row_te.columns: return
        msk = np.isfinite(row_te[pred_col].values)
        if msk.sum() < 10: return
        m = compute_metrics_v2(row_te[pred_col].values[msk], tl_te[msk],
                                s_te[msk], e_te[msk], label)
        all_metrics.append(m)
        ps = m.get("per_sensor",{})
        print(f"  {label:50s}  RMSE={m['rmse_pgv']:.4f}  "
              f"MP4={ps.get('MP4',{}).get('rmse_pgv','?'):.4f}  "
              f"mono={m['mono_viol_rate']:.3f}")
        return m

    _eval("pred_log_profile",      "Profile-only")
    _eval("pred_log_profile_mono", "Profile-only mono")

    # ISSUE 2 FIX: track per-policy test metrics so ablation CSV is correct
    policy_metrics: Dict[Tuple[str, str], dict] = {}

    for vname in VARIANT_CONFIGS:
        for policy in ["global_best", "mp4_best", "balanced_best"]:
            cf = f"pred_{vname}_{policy}"
            m = _eval(cf, f"{vname}_{policy}")
            if m:
                policy_metrics[(vname, policy)] = m
                if policy == "global_best":
                    vt_metrics[vname] = m
                    gz = compute_geometry_zone_metrics(row_te, cf, vname)
                    geom_zone_all.extend(gz)
            _eval(f"{cf}_mono", f"{vname}_{policy}_mono")

    # Baselines
    v1_saved_metrics = None
    if v1_arts and "predictions_test" in v1_arts:
        pv1 = v1_arts["predictions_test"]
        for vcol, vlbl in [("pred_log_final","FO-PhysProfile v1 final"),
                             ("pred_log_final_mono","FO-v1 final mono")]:
            if vcol not in pv1.columns: continue
            al = row_te[["event_id","sensor"]].merge(
                pv1[["event_id","sensor",vcol]],
                on=["event_id","sensor"], how="left")[vcol].values
            msk = np.isfinite(al)
            if msk.sum() > 10:
                m = compute_metrics_v2(al[msk], tl_te[msk],
                                        s_te[msk], e_te[msk], vlbl)
                all_metrics.append(m)
                if vlbl == "FO-PhysProfile v1 final":
                    v1_saved_metrics = m
                    vt_metrics[vlbl] = m

    pxgbr = _v1.load_pxgbr_predictions(row_te)
    for bl, bp in pxgbr.items():
        if bp is not None:
            msk = np.isfinite(bp)
            if msk.sum() > 10:
                m = compute_metrics_v2(bp[msk], tl_te[msk],
                                        s_te[msk], e_te[msk], bl)
                all_metrics.append(m); vt_metrics[bl] = m

    p3d = _v1.find_p3_dir()
    if p3d and (p3d/"all_predictions.parquet").exists():
        p3a = pd.read_parquet(p3d/"all_predictions.parquet")
        p3a["event_id"] = p3a["event_id"].astype(str)
        p3t = p3a[p3a["split"]=="test"]
        p3al = row_te[["event_id","sensor"]].merge(
            p3t[["event_id","sensor","pred_log_p3"]],
            on=["event_id","sensor"],how="left")["pred_log_p3"].values
        msk = np.isfinite(p3al)
        if msk.sum() > 10:
            m = compute_metrics_v2(p3al[msk], tl_te[msk], s_te[msk], e_te[msk],
                                    "P3_corrected_n")
            all_metrics.append(m); vt_metrics["P3_corrected_n"] = m

    # Save metrics
    rows_tbl = []
    for m in all_metrics:
        rt = {"model": m["model"], "rmse_pgv": m["rmse_pgv"],
              "rmse_log": m["rmse_log"], "r2_log": m["r2_log"],
              "bias_pgv": m["bias_pgv"], "mono_viol": m.get("mono_viol_rate",np.nan)}
        for ss in SENSOR_ORDER:
            rt[f"rmse_{ss}"] = m["per_sensor"].get(ss,{}).get("rmse_pgv",np.nan)
        for k in ["pgv_gt4","pgv_gt8","mp4_pgv_gt4","mp4_pgv_gt8"]:
            rt[f"{k}_n"]    = m.get(k,{}).get("n",np.nan)
            rt[f"{k}_rmse"] = m.get(k,{}).get("rmse_pgv",np.nan)
        rows_tbl.append(rt)

    mdf = pd.DataFrame(rows_tbl).sort_values("rmse_pgv")
    mdf.to_csv(out_dir/"metrics_table.csv", index=False)
    pd.DataFrame(geom_zone_all).to_csv(out_dir/"geometry_zone_metrics.csv", index=False)

    # Ablation table — ISSUE 2 FIX: use policy_metrics for per-policy test values
    abl = []
    for vname, res in variant_results.items():
        for policy in ["global_best", "mp4_best", "balanced_best"]:
            _, vr, vmr, sch, _ = res[policy]
            m = policy_metrics.get((vname, policy), {})
            ps = m.get("per_sensor", {})
            abl.append({
                "variant":  vname, "policy": policy, "scheme": sch,
                "val_rmse_pgv":  vr, "val_mp4_rmse": vmr,
                "test_rmse_pgv": m.get("rmse_pgv", np.nan),
                "test_rmse_log": m.get("rmse_log", np.nan),
                "test_mp4_rmse": ps.get("MP4", {}).get("rmse_pgv", np.nan),
                "test_mp4_pgv_gt4_rmse": m.get("mp4_pgv_gt4", {}).get("rmse_pgv", np.nan),
                "test_mp4_pgv_gt8_rmse": m.get("mp4_pgv_gt8", {}).get("rmse_pgv", np.nan),
            })
    pd.DataFrame(abl).to_csv(out_dir/"geometry_ablation_metrics.csv", index=False)

    print("\n" + "=" * 70)
    print("METRICS SUMMARY (sorted by RMSE PGV)")
    print("=" * 70)
    sc = ["model","rmse_pgv","rmse_log","rmse_MP4"]
    sc = [c for c in sc if c in mdf.columns]
    print(mdf[sc].head(12).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 9 — Save outputs (PATCH 10)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70); print("STEP 9 — Save outputs"); print("=" * 70)

    save_c = ["event_id","sensor","track_number","distance",
              "target_log","target_pgv","pred_log_profile","pred_log_profile_mono"]
    for vname in VARIANT_CONFIGS:
        cf = f"pred_{vname}_global_best"
        if cf in row_te.columns:
            save_c += [cf, f"{cf}_mono"]
    gsave = ["sensor_y_m","fo_to_track_axis_pos","acc_track_to_fo_ratio",
             "is_between_track_and_fo","is_on_fo_line","is_beyond_fo_away_from_track"]
    save_c += [c for c in gsave if c in row_te.columns]
    row_te[[c for c in save_c if c in row_te.columns]].to_parquet(
        out_dir/"predictions_test.parquet", index=False)

    gfc = list(dict.fromkeys(fc_by_variant["V2_geom"] + ["event_id","sensor"] + gsave))
    row_te[[c for c in gfc if c in row_te.columns]].to_parquet(
        out_dir/"row_features_geometry.parquet", index=False)

    for vname, (mod, fc) in models_gb.items():
        vc = [c for c in fc if c in row_tr.columns]
        if len(vc) == len(mod.feature_importances_):
            pd.DataFrame({"feature": vc, "importance": mod.feature_importances_}
                         ).sort_values("importance", ascending=False).to_csv(
                out_dir / f"feature_importance_{vname}.csv", index=False)

    print(f"[save] All outputs → {out_dir}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 10 — Plots (PATCH 10)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70); print("STEP 10 — Plots"); print("=" * 70)
    v1c = "pred_V1_ref_global_best"; v2c = "pred_V2_geom_global_best"

    pm = {m["model"]: m for m in all_metrics
          if any(x in m["model"] for x in
                 ["V1_ref_global","V2_geom_global","v1 final","PXGBR-R2","P3"])}
    if pm:
        plot_per_sensor_rmse(pm, out_dir/"per_sensor_rmse_v1_vs_v2_geometry.png")

    plot_residual_vs_axis_pos_v1_vs_v2(
        row_te, v1c, v2c,
        out_dir/"residual_vs_fo_to_track_axis_pos_v1_vs_v2.png")

    plot_residual_box_by_zone_v1_vs_v2(
        row_te, v1c, v2c,
        out_dir/"residual_box_by_geometry_zone_v1_vs_v2.png")

    if v2c in row_te.columns:
        top_e = (row_te.dropna(subset=[v2c])
                 .groupby("event_id")["target_pgv"].max().nlargest(16).index)
        fig, axes = plt.subplots(4, 4, figsize=(14, 12))
        for ax, eid in zip(np.array(axes).flat, top_e):
            ev = row_te[row_te["event_id"]==eid].sort_values("distance")
            ax.plot(ev["distance"], np.exp(ev["target_log"]), "ko-", ms=4)
            ax.plot(ev["distance"], np.exp(ev[v2c].fillna(np.nan)), "r^--", ms=4)
            if v1c in ev.columns:
                ax.plot(ev["distance"], np.exp(ev[v1c].fillna(np.nan)), "b.:", ms=4)
            ax.set_title(str(eid)[:12], fontsize=7)
            ax.set_xlabel("dist", fontsize=7); ax.set_ylabel("PGV", fontsize=7)
        plt.suptitle("High-PGV profiles — v2 geometry", fontsize=10)
        _save_fig(out_dir/"high_pgv_profiles_v2_geometry.png")

    plot_mp4_profiles_v1_vs_v2(row_te, v1c, v2c, 16,
                                 out_dir/"mp4_profiles_v1_vs_v2_geometry.png")

    if v2c in row_te.columns:
        ok = np.isfinite(row_te[v2c].values)
        plot_scatter(row_te["target_pgv"].values[ok],
                     np.exp(row_te[v2c].values[ok]),
                     "V2_geom",
                     out_dir/"measured_vs_predicted_v2_geometry.png")

    for vname, (mod, fc) in models_gb.items():
        vc = [c for c in fc if c in row_tr.columns]
        plot_feat_imp(mod, vc, f"Feature importance — {vname}",
                       out_dir / f"feature_importance_local_residual_{vname}.png")

    plot_geometry_table(geom_df, out_dir/"geometry_diagnostic_layout_table.png")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 11 — Interpretation (PATCH 11)
    # ══════════════════════════════════════════════════════════════════════════
    print_interpretation(
        vt_metrics     = vt_metrics,
        v1_saved_metrics = v1_saved_metrics,
        geom_zone_rows  = geom_zone_all,
        feat_cols_dict  = {v: res["valid_cols"] for v, res in variant_results.items()},
        models_dict     = {v: res["global_best"][0] for v, res in variant_results.items()},
    )

    print(f"\n  v1_dir:             {v1_dir_path}")
    print(f"  v1 artifacts reused: {v1['used_saved_test']}")
    print(f"  git commit:         {git_hash}")
    print(f"  output dir:         {out_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
