"""Audit FO time-series data before building the waveform dataset.

For every event NetCDF file found in the same input folders as
config_parquet_v2, reports:

  - FO strain shape, sampling rate, duration
  - Whether absolute channel 1194 is stored in geometry_fo/fo_channel_id
  - Accelerometer (velocity_mms) PGV_z target availability
  - Train metadata completeness (type, speed, track number)
  - Cross-check against Parquet v2 and v4 event IDs

Recommends candidate fixed lengths T (samples) with counts of events that
would be padded vs. cropped, and prints memory estimates.

Saves a plain-text report alongside this script:
    scripts/audit_fo_timeseries_report.txt

Run from the repository root:
    python scripts/audit_fo_timeseries.py
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import netCDF4 as nc
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure repository root is on sys.path so src.* imports work when this
# script is executed from any working directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.db.parquet.config_parquet_v2 import CONFIG as PARQUET_V2_CFG  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_CHANNEL = 1194          # absolute FO channel to audit
PARQUET_ROOT = Path(r"P:\11210978-erju-ai\holten_parquet")
REPORT_PATH = Path(__file__).parent / "audit_fo_timeseries_report.txt"

# Candidate fixed lengths to evaluate (in samples).
# Labelled for readability; actual recommendation uses the duration data.
_CANDIDATE_T: List[int] = [3000, 5000, 7500, 10000, 15000, 20000, 25000, 30000]


# ---------------------------------------------------------------------------
# Per-file audit helper
# ---------------------------------------------------------------------------

def _to_python_scalar(v: Any) -> Any:
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8")
        except Exception:
            pass
    return v


def _safe_float(v: Any, default: float = np.nan) -> float:
    v = _to_python_scalar(v)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = -1) -> int:
    v = _to_python_scalar(v)
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def audit_one_file(nc_path: Path) -> Dict[str, Any]:
    """Return a dict of audit metrics for a single NetCDF event file.

    All values are safe for aggregation (no exceptions propagate).
    """
    rec: Dict[str, Any] = {
        "file": str(nc_path),
        "event_id": "",
        "site_id": "",
        # FO
        "has_fo_group": False,
        "has_fo_strain": False,
        "has_fs_hz": False,
        "fo_n_samples": -1,
        "fo_n_channels": -1,
        "fs_hz": np.nan,
        "duration_s": np.nan,
        # Channel 1194
        "has_geometry_fo": False,
        "has_fo_channel_id": False,
        "ch1194_available": False,
        "ch1194_index": -1,
        "fo_channel_min": -1,
        "fo_channel_max": -1,
        # Accelerometers
        "n_sensors_total": 0,
        "n_sensors_velocity_mms": 0,
        "n_sensors_finite_pgvz": 0,
        # Metadata
        "train_type": "",
        "train_speed_kmh": np.nan,
        "track_number": -1,
        # Status
        "parse_error": "",
    }

    try:
        with nc.Dataset(nc_path, "r") as ds:
            # --- identifiers ---
            rec["event_id"] = str(_to_python_scalar(getattr(ds, "event_id", "")))
            rec["site_id"] = str(_to_python_scalar(getattr(ds, "site_id", "")))

            # --- FO strain ---
            rec["has_fo_group"] = "fo" in ds.groups
            if rec["has_fo_group"]:
                fo_grp = ds.groups["fo"]
                rec["has_fo_strain"] = "strain" in fo_grp.variables
                rec["has_fs_hz"] = "fs_hz" in fo_grp.variables

                if rec["has_fs_hz"]:
                    rec["fs_hz"] = _safe_float(fo_grp.variables["fs_hz"][()])

                if rec["has_fo_strain"]:
                    shp = fo_grp.variables["strain"].shape
                    if len(shp) == 2:
                        rec["fo_n_samples"] = int(shp[0])
                        rec["fo_n_channels"] = int(shp[1])
                        if np.isfinite(rec["fs_hz"]) and rec["fs_hz"] > 0:
                            rec["duration_s"] = rec["fo_n_samples"] / rec["fs_hz"]

            # --- Channel 1194 via geometry_fo ---
            rec["has_geometry_fo"] = "geometry_fo" in ds.groups
            if rec["has_geometry_fo"]:
                geom = ds.groups["geometry_fo"]
                rec["has_fo_channel_id"] = "fo_channel_id" in geom.variables
                if rec["has_fo_channel_id"]:
                    ch_ids = np.asarray(
                        geom.variables["fo_channel_id"][:], dtype=np.int32
                    )
                    rec["fo_channel_min"] = int(ch_ids.min())
                    rec["fo_channel_max"] = int(ch_ids.max())
                    indices = np.where(ch_ids == TARGET_CHANNEL)[0]
                    if len(indices) > 0:
                        rec["ch1194_available"] = True
                        rec["ch1194_index"] = int(indices[0])

            # --- Accelerometer sensors ---
            if "acc" in ds.groups:
                acc_root = ds.groups["acc"]
                sensor_ids = list(acc_root.groups.keys())
                rec["n_sensors_total"] = len(sensor_ids)
                for sid in sensor_ids:
                    sg = acc_root.groups[sid]
                    # Only free-field sensors (MP1–MP13) store velocity_mms
                    if "velocity_mms" not in sg.variables:
                        continue
                    rec["n_sensors_velocity_mms"] += 1
                    try:
                        vel = np.asarray(sg.variables["velocity_mms"][:],
                                         dtype=np.float64)
                        # z-axis is column 2; use axis_mask to find it if present
                        z_col = 2
                        if "axis_mask" in sg.variables:
                            mask = np.asarray(
                                sg.variables["axis_mask"][:], dtype=np.int32
                            )
                            # axis_mask is [has_x, has_y, has_z]; z at position 2
                            active_cols = np.where(mask)[0]
                            if len(active_cols) >= 3:
                                z_col = int(active_cols[2])
                            elif len(active_cols) > 0:
                                z_col = int(active_cols[-1])
                        if vel.ndim == 2 and vel.shape[1] > z_col:
                            z_trace = vel[:, z_col]
                            pgv = float(np.nanmax(np.abs(z_trace)))
                            if pgv > 0 and np.isfinite(pgv):
                                rec["n_sensors_finite_pgvz"] += 1
                    except Exception:
                        pass

            # --- Train metadata ---
            meta = ds.groups.get("meta_acc") or ds.groups.get("meta")
            if meta is not None:
                def _read(grp, key, default):
                    try:
                        return _to_python_scalar(grp.variables[key][()])
                    except Exception:
                        return default

                rec["train_type"] = str(_read(meta, "train_type", ""))
                rec["train_speed_kmh"] = _safe_float(
                    _read(meta, "train_speed_kmh", np.nan)
                )
                rec["track_number"] = _safe_int(
                    _read(meta, "track_number", -1)
                )

    except Exception as exc:
        rec["parse_error"] = f"{type(exc).__name__}: {exc}"

    return rec


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total > 0 else "n/a"


def _quantiles(values: np.ndarray, qs=(0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0)):
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return {q: np.nan for q in qs}
    return {q: float(np.quantile(vals, q)) for q in qs}


def _histogram_str(values: np.ndarray, n_bins: int = 10) -> str:
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return "  (no data)"
    counts, edges = np.histogram(vals, bins=n_bins)
    lines = []
    bar_max = max(counts) if max(counts) > 0 else 1
    bar_width = 30
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(c / bar_max * bar_width)
        lines.append(f"  [{lo:8.1f}, {hi:8.1f})  {c:5d}  {bar}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    lines: List[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("FO Time-Series Data Audit")
    log(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Target channel : {TARGET_CHANNEL}")
    log("=" * 70)

    # ------------------------------------------------------------------
    # 1. Discover NetCDF files (same folders as Parquet v2 config)
    # ------------------------------------------------------------------
    input_folders = PARQUET_V2_CFG.input_folder_paths()
    netcdf_glob = PARQUET_V2_CFG.netcdf_glob

    log("\n[1/7] Input folders:")
    for p in input_folders:
        status = "OK" if p.exists() else "MISSING"
        log(f"  [{status}] {p}")

    netcdf_files: List[Path] = []
    for folder in input_folders:
        if folder.exists():
            netcdf_files.extend(sorted(folder.glob(netcdf_glob)))
    netcdf_files = sorted(set(netcdf_files))

    log(f"\n  NetCDF files found : {len(netcdf_files):,}")
    if len(netcdf_files) == 0:
        log("  ERROR: no NetCDF files found. Aborting audit.")
        return

    # ------------------------------------------------------------------
    # 2. Per-file audit
    # ------------------------------------------------------------------
    log(f"\n[2/7] Auditing {len(netcdf_files):,} files ...")
    records: List[Dict[str, Any]] = []
    n_errors = 0
    for i, nc_path in enumerate(netcdf_files, 1):
        if i % 100 == 0 or i == len(netcdf_files):
            print(f"  {i}/{len(netcdf_files)} ...", end="\r")
        rec = audit_one_file(nc_path)
        if rec["parse_error"]:
            n_errors += 1
        records.append(rec)
    print()

    df = pd.DataFrame(records)
    n_total = len(df)

    # ------------------------------------------------------------------
    # 3. Cross-check with Parquet v2 / v4
    # ------------------------------------------------------------------
    log("\n[3/7] Cross-checking with Parquet v2 / v4 ...")

    v2_events: set = set()
    v4_events: set = set()
    v2_path_used = ""
    v4_path_used = ""

    if PARQUET_ROOT.exists():
        v2_builds = sorted(PARQUET_ROOT.glob("parquet_v002_*"))
        v4_builds = sorted(PARQUET_ROOT.glob("parquet_v004_*"))

        if v2_builds:
            v2_path = v2_builds[-1] / "dataset.parquet"
            if v2_path.exists():
                _v2 = pd.read_parquet(v2_path, columns=["event_id"])
                v2_events = set(_v2["event_id"].unique())
                v2_path_used = str(v2_path)
                log(f"  Parquet v2 : {len(v2_events):,} unique events  ({v2_path})")
            else:
                log(f"  Parquet v2 : dataset.parquet not found in {v2_builds[-1]}")
        else:
            log("  Parquet v2 : no parquet_v002_* builds found")

        if v4_builds:
            v4_path = v4_builds[-1] / "dataset.parquet"
            if v4_path.exists():
                _v4 = pd.read_parquet(v4_path, columns=["event_id"])
                v4_events = set(_v4["event_id"].unique())
                v4_path_used = str(v4_path)
                log(f"  Parquet v4 : {len(v4_events):,} unique events  ({v4_path})")
            else:
                log(f"  Parquet v4 : dataset.parquet not found in {v4_builds[-1]}")
        else:
            log("  Parquet v4 : no parquet_v004_* builds found")
    else:
        log(f"  Parquet root not accessible: {PARQUET_ROOT}")

    nc_events = set(df["event_id"].unique()) - {""}
    log(f"\n  Unique event_ids in NetCDF files : {len(nc_events):,}")

    if v2_events:
        in_v2_not_nc = v2_events - nc_events
        in_nc_not_v2 = nc_events - v2_events
        log(f"  In Parquet v2 but NOT in NetCDF  : {len(in_v2_not_nc):,}")
        log(f"  In NetCDF but NOT in Parquet v2  : {len(in_nc_not_v2):,}")

    if v4_events:
        in_v4_not_nc = v4_events - nc_events
        log(f"  In Parquet v4 but NOT in NetCDF  : {len(in_v4_not_nc):,}")

    # ------------------------------------------------------------------
    # 4. FO signal quality
    # ------------------------------------------------------------------
    log("\n[4/7] FO signal quality")
    log("-" * 50)

    n_fo_group = int(df["has_fo_group"].sum())
    n_fo_strain = int(df["has_fo_strain"].sum())
    n_fs_hz = int(df["has_fs_hz"].sum())
    n_parse_err = int((df["parse_error"] != "").sum())

    log(f"  Files with /fo group          : {n_fo_group:,} / {n_total:,}  ({_pct(n_fo_group, n_total)})")
    log(f"  Files with /fo/strain         : {n_fo_strain:,} / {n_total:,}  ({_pct(n_fo_strain, n_total)})")
    log(f"  Files with /fo/fs_hz          : {n_fs_hz:,} / {n_total:,}  ({_pct(n_fs_hz, n_total)})")
    log(f"  Files with parse error        : {n_parse_err:,} / {n_total:,}  ({_pct(n_parse_err, n_total)})")
    if n_parse_err > 0:
        errs = df[df["parse_error"] != ""][["file", "parse_error"]].head(5)
        for _, row in errs.iterrows():
            log(f"    {Path(row['file']).name}: {row['parse_error']}")

    # Sampling rate
    fs_vals = df.loc[df["fs_hz"] > 0, "fs_hz"].values
    if len(fs_vals) > 0:
        unique_fs, counts_fs = np.unique(fs_vals, return_counts=True)
        log(f"\n  Sampling rate (fs_hz):")
        for fs, cnt in zip(unique_fs, counts_fs):
            log(f"    {fs:8.1f} Hz  →  {cnt:,} files  ({_pct(cnt, n_total)})")
    else:
        log("\n  No valid fs_hz values found.")

    # ------------------------------------------------------------------
    # 5. Duration and shape distributions
    # ------------------------------------------------------------------
    log("\n[5/7] Duration and shape distributions")
    log("-" * 50)

    dur_vals = df.loc[df["duration_s"] > 0, "duration_s"].values
    samp_vals = df.loc[df["fo_n_samples"] > 0, "fo_n_samples"].values
    ch_vals = df.loc[df["fo_n_channels"] > 0, "fo_n_channels"].values

    if len(dur_vals) > 0:
        qs = _quantiles(dur_vals)
        log(f"\n  Event duration (seconds)  [N={len(dur_vals):,}]")
        log(f"    min     : {qs[0.0]:8.2f} s")
        log(f"    p5      : {qs[0.05]:8.2f} s")
        log(f"    p25     : {qs[0.25]:8.2f} s")
        log(f"    median  : {qs[0.50]:8.2f} s")
        log(f"    p75     : {qs[0.75]:8.2f} s")
        log(f"    p95     : {qs[0.95]:8.2f} s")
        log(f"    max     : {qs[1.0]:8.2f} s")
        log(f"    mean    : {float(np.mean(dur_vals)):8.2f} s")
        log(f"    std     : {float(np.std(dur_vals)):8.2f} s")
        log(f"\n  Duration histogram:")
        log(_histogram_str(dur_vals, n_bins=12))
    else:
        log("  No valid duration values found.")

    if len(samp_vals) > 0:
        qs_s = _quantiles(samp_vals)
        log(f"\n  FO n_samples  [N={len(samp_vals):,}]")
        log(f"    min={int(qs_s[0.0]):,}  p5={int(qs_s[0.05]):,}  p25={int(qs_s[0.25]):,}  "
            f"median={int(qs_s[0.50]):,}  p75={int(qs_s[0.75]):,}  p95={int(qs_s[0.95]):,}  "
            f"max={int(qs_s[1.0]):,}")

    if len(ch_vals) > 0:
        qs_c = _quantiles(ch_vals)
        log(f"\n  FO n_channels  [N={len(ch_vals):,}]")
        log(f"    min={int(qs_c[0.0]):,}  median={int(qs_c[0.50]):,}  max={int(qs_c[1.0]):,}")
        unique_ch, cnt_ch = np.unique(ch_vals.astype(int), return_counts=True)
        for uc, cc in zip(unique_ch, cnt_ch):
            log(f"    n_channels={uc:3d} : {cc:,} files  ({_pct(cc, n_total)})")

    # ------------------------------------------------------------------
    # 6. Channel 1194 availability
    # ------------------------------------------------------------------
    log(f"\n[6/7] Channel {TARGET_CHANNEL} availability")
    log("-" * 50)

    n_geom_fo = int(df["has_geometry_fo"].sum())
    n_ch_id_var = int(df["has_fo_channel_id"].sum())
    n_ch1194 = int(df["ch1194_available"].sum())

    log(f"  Files with geometry_fo group    : {n_geom_fo:,} / {n_total:,}  ({_pct(n_geom_fo, n_total)})")
    log(f"  Files with fo_channel_id var    : {n_ch_id_var:,} / {n_total:,}  ({_pct(n_ch_id_var, n_total)})")
    log(f"  Files where ch {TARGET_CHANNEL} is present : {n_ch1194:,} / {n_total:,}  ({_pct(n_ch1194, n_total)})")
    log(f"  Files where ch {TARGET_CHANNEL} is MISSING : {n_total - n_ch1194:,} / {n_total:,}")

    ch_idx_vals = df.loc[df["ch1194_available"], "ch1194_index"].values
    if len(ch_idx_vals) > 0:
        unique_idx, cnt_idx = np.unique(ch_idx_vals.astype(int), return_counts=True)
        log(f"\n  Array index of ch {TARGET_CHANNEL} in /fo/strain:")
        for ui, ci in zip(unique_idx, cnt_idx):
            log(f"    index={ui:3d} : {ci:,} files  ({_pct(ci, n_total)})")

    ch_min_vals = df.loc[df["fo_channel_min"] >= 0, "fo_channel_min"].values
    ch_max_vals = df.loc[df["fo_channel_max"] >= 0, "fo_channel_max"].values
    if len(ch_min_vals) > 0:
        log(f"\n  Stored channel range in /fo/strain:")
        log(f"    fo_channel_min: min={int(ch_min_vals.min()):,}  max={int(ch_min_vals.max()):,}  "
            f"mode={int(np.bincount(ch_min_vals.astype(int)).argmax()):,}")
        log(f"    fo_channel_max: min={int(ch_max_vals.min()):,}  max={int(ch_max_vals.max()):,}  "
            f"mode={int(np.bincount(ch_max_vals.astype(int)).argmax()):,}")

    # ------------------------------------------------------------------
    # 7. Accelerometer targets
    # ------------------------------------------------------------------
    log("\n[7/7] Accelerometer target coverage and metadata")
    log("-" * 50)

    n_has_sensors = int((df["n_sensors_total"] > 0).sum())
    n_has_vel = int((df["n_sensors_velocity_mms"] > 0).sum())
    n_has_pgvz = int((df["n_sensors_finite_pgvz"] > 0).sum())
    n_no_pgvz = n_total - n_has_pgvz

    log(f"\n  Accelerometers:")
    log(f"    Events with ≥1 acc sensor           : {n_has_sensors:,}  ({_pct(n_has_sensors, n_total)})")
    log(f"    Events with ≥1 velocity_mms sensor  : {n_has_vel:,}  ({_pct(n_has_vel, n_total)})")
    log(f"    Events with ≥1 finite PGV_z         : {n_has_pgvz:,}  ({_pct(n_has_pgvz, n_total)})")
    log(f"    Events with NO valid PGV_z          : {n_no_pgvz:,}  ({_pct(n_no_pgvz, n_total)})")

    sensors_per_event = df.loc[df["n_sensors_velocity_mms"] > 0, "n_sensors_velocity_mms"].values
    if len(sensors_per_event) > 0:
        uq, uqcnt = np.unique(sensors_per_event.astype(int), return_counts=True)
        log(f"\n  Distribution of n_sensors_velocity_mms per event (excluding 0):")
        for u, c in zip(uq, uqcnt):
            log(f"    n_sensors={u:2d} : {c:,} events")

    log(f"\n  Train metadata:")
    n_speed_avail = int(df["train_speed_kmh"].notna().sum())
    n_speed_nan = n_total - n_speed_avail
    n_type_avail = int((df["train_type"] != "").sum())
    n_track_avail = int((df["track_number"] >= 0).sum())
    log(f"    train_type available     : {n_type_avail:,}  ({_pct(n_type_avail, n_total)})")
    log(f"    train_speed_kmh valid    : {n_speed_avail:,}  ({_pct(n_speed_avail, n_total)})")
    log(f"    train_speed_kmh missing  : {n_speed_nan:,}  ({_pct(n_speed_nan, n_total)})")
    log(f"    track_number available   : {n_track_avail:,}  ({_pct(n_track_avail, n_total)})")

    # ------------------------------------------------------------------
    # Fixed-length recommendation
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("FIXED-LENGTH RECOMMENDATION")
    log("=" * 70)

    if len(samp_vals) > 0 and len(fs_vals) > 0:
        # Use the most common fs for converting samples → seconds
        modal_fs = float(np.bincount(np.round(fs_vals).astype(int)).argmax())
        if modal_fs <= 0:
            modal_fs = float(fs_vals[0])

        log(f"\nModal fs = {modal_fs:.0f} Hz")
        log(f"\nCandidate fixed lengths  (events padded / cropped / exact):")
        log(f"  {'T (samples)':>12}  {'Duration':>10}  {'Padded':>8}  {'Cropped':>8}  {'Exact':>8}  {'% coverage'}")
        log(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")

        valid_samp = samp_vals[samp_vals > 0]
        for T in _CANDIDATE_T:
            n_pad = int(np.sum(valid_samp < T))
            n_crop = int(np.sum(valid_samp > T))
            n_exact = int(np.sum(valid_samp == T))
            pct_cov = (n_pad + n_exact) / len(valid_samp) * 100  # no cropping needed
            dur_str = f"{T / modal_fs:.1f} s"
            log(f"  {T:>12,}  {dur_str:>10}  {n_pad:>8,}  {n_crop:>8,}  {n_exact:>8,}  "
                f"{pct_cov:>9.1f}%")

        # p5 recommendation
        p5_samples = int(np.quantile(valid_samp, 0.05))
        p25_samples = int(np.quantile(valid_samp, 0.25))
        median_samples = int(np.quantile(valid_samp, 0.50))

        log(f"\n  p5  duration → {p5_samples:,} samples  ({p5_samples / modal_fs:.1f} s)")
        log(f"  p25 duration → {p25_samples:,} samples  ({p25_samples / modal_fs:.1f} s)")
        log(f"  p50 duration → {median_samples:,} samples  ({median_samples / modal_fs:.1f} s)")
        log(f"\n  NOTE: p5 strategy means ~5% of events would need padding at T=p5.")
        log(f"  If events cluster at a common round length, use that length directly.")
        log(f"  Review the duration histogram above before deciding T.")

    # ------------------------------------------------------------------
    # Memory estimate
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("MEMORY ESTIMATE (waveform dataset, float32)")
    log("=" * 70)

    n_est = len(nc_events) if nc_events else n_total
    log(f"\n  N events = {n_est:,} (unique event_ids)")
    log(f"  dtype   = float32  (4 bytes / sample)")
    log(f"\n  {'T (samples)':>12}  {'Duration':>10}  {'Size (MB)':>12}")
    log(f"  {'-'*12}  {'-'*10}  {'-'*12}")
    if len(fs_vals) > 0:
        modal_fs_m = float(np.bincount(np.round(fs_vals).astype(int)).argmax())
        if modal_fs_m <= 0:
            modal_fs_m = float(fs_vals[0])
    else:
        modal_fs_m = 500.0
    for T in _CANDIDATE_T:
        size_mb = n_est * T * 4 / 1024 / 1024
        dur_str = f"{T / modal_fs_m:.1f} s"
        log(f"  {T:>12,}  {dur_str:>10}  {size_mb:>12.1f}")

    log(f"\n  (This is for the waveforms.npy file only, one row per event.)")
    log(f"  (Sensor-level training pairs are stored in event_index.parquet.)")

    # ------------------------------------------------------------------
    # Summary banner
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  Total NetCDF files          : {n_total:,}")
    log(f"  Parse errors                : {n_parse_err:,}")
    log(f"  Valid FO strain             : {n_fo_strain:,}  ({_pct(n_fo_strain, n_total)})")
    log(f"  Channel {TARGET_CHANNEL} available       : {n_ch1194:,}  ({_pct(n_ch1194, n_total)})")
    log(f"  Events with finite PGV_z    : {n_has_pgvz:,}  ({_pct(n_has_pgvz, n_total)})")
    if len(dur_vals) > 0:
        log(f"  Duration: p5={np.quantile(dur_vals, 0.05):.1f} s  "
            f"median={np.quantile(dur_vals, 0.50):.1f} s  "
            f"p95={np.quantile(dur_vals, 0.95):.1f} s")
    if v2_events:
        log(f"  Parquet v2 event coverage   : {len(v2_events & nc_events):,} / {len(v2_events):,} "
            f"events matched in NetCDF")
    log("=" * 70)

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    report_text = "\n".join(lines)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
