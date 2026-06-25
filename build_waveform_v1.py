"""Build the single-channel FO waveform dataset (waveform v1).

Input
-----
* Event NetCDF files (same folders as Parquet v2).
* Parquet v2 sensor-level dataset — defines the canonical 1,697-event set,
  the train metadata, and (via geometry correction) the distances/targets that
  the training script will join by ``event_id``.

Output  (holten_waveform_v001_<timestamp>/)
-------
* waveforms.npy        — float32 (N_events, T), one channel-1194 waveform per
                         event, processed (1-100 Hz bandpass -> 250 Hz ->
                         30 s energy-centred crop/pad).  No normalization.
* event_index.parquet  — one row per event: event_id, waveform_row_idx, train
                         metadata, and full crop diagnostics.
* build_config.json
* build_log.txt
* plots/crop_diag_*.png — crop diagnostics for representative events.

One waveform is stored per event (NOT per sensor row).  The CNN Dataset maps
each sensor-level target row (from Parquet v2) to the event waveform by
``event_id``.

Run from the repository root:
    python build_waveform_v1.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd

# Ensure repo root on path when run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.db.waveform.config_waveform_v1 import CONFIG
from src.db.waveform.waveform_v1_utils import (
    extract_channel_signal,
    process_event_waveform,
)
from src.utils.geometry_utils import apply_corrected_distances


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_python_scalar(v):
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


def _create_build_folder(output_root: Path, version_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = output_root / f"{version_name}_{ts}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _open_netcdf_with_retry(nc_path: Path, retries: int = 4, delay: float = 1.5):
    """Open a NetCDF file, retrying briefly on transient network-drive errors.

    The Holten NetCDF files live on a network drive (P:) that occasionally
    drops the connection.  A short retry loop prevents a transient blip from
    silently dropping an event from the build.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return nc.Dataset(nc_path, "r")
        except Exception as exc:  # OSError / network errors
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(delay)
    raise last_exc if last_exc is not None else RuntimeError("open failed")


def _load_canonical_events(cfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (event_meta_df, sensor_df) from the latest Parquet v2 build.

    event_meta_df : one row per event with train metadata + event-level PGV.
    sensor_df     : sensor-level rows with corrected distances (for diagnostics
                    and high/low-PGV event selection).
    """
    parquet_root = cfg.parquet_root()
    v2_builds = sorted(parquet_root.glob("parquet_v002_*"), key=lambda p: p.name)
    if not v2_builds:
        raise FileNotFoundError(f"No parquet_v002_* builds found in {parquet_root}")
    v2_path = v2_builds[-1] / "dataset.parquet"
    if not v2_path.exists():
        raise FileNotFoundError(f"dataset.parquet not found in {v2_builds[-1]}")

    df = pd.read_parquet(v2_path)

    # Exclude in-track sensors (MP14-MP19) to match the tabular benchmark.
    if cfg.exclude_sensor_ids and "sensor_id" in df.columns:
        df = df[~df["sensor_id"].isin(cfg.exclude_sensor_ids)].copy()

    # Corrected distances (adds effective_distance_to_active_track_m).
    if "effective_distance_to_active_track_m" not in df.columns:
        df = apply_corrected_distances(df)

    tcfg = cfg.target
    # Event-level metadata: identical across an event's sensor rows -> first.
    meta_cols = [c for c in tcfg.metadata_cols if c in df.columns]
    keep = ["event_id"] + meta_cols
    if "site_id" in df.columns:
        keep.append("site_id")
    event_meta = (
        df[keep].groupby("event_id", sort=True).first().reset_index()
    )

    # Event-level PGV summary for representative-event selection.
    pgv = (
        df.groupby("event_id")[tcfg.pgv_col]
        .agg(["max", "mean", "count"])
        .rename(columns={"max": "pgv_max", "mean": "pgv_mean", "count": "n_sensors"})
        .reset_index()
    )
    event_meta = event_meta.merge(pgv, on="event_id", how="left")

    return event_meta, df


def _read_event_id(ds) -> str:
    return str(_to_python_scalar(getattr(ds, "event_id", "")))


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------


def _plot_crop_diagnostic(
    event_id: str,
    nc_path: Path,
    cfg,
    label: str,
    out_path: Path,
) -> None:
    """Draw a 2-panel crop diagnostic for one event.

    Top    : raw original channel-1194 signal (native fs) with the crop span.
    Bottom : processed (bandpassed + 250 Hz) signal with the crop window and
             energy-centre marker.
    """
    scfg, wcfg = cfg.signal, cfg.window
    with nc.Dataset(nc_path, "r") as ds:
        raw, reason = extract_channel_signal(ds, scfg.target_channel)
        proc, reason2 = process_event_waveform(
            dataset=ds,
            target_channel=scfg.target_channel,
            bandpass_freqmin=scfg.bandpass_freqmin,
            bandpass_freqmax=scfg.bandpass_freqmax,
            bandpass_corners=scfg.bandpass_corners,
            target_fs_hz=scfg.target_fs_hz,
            fixed_length_samples=cfg.fixed_length_samples,
            crop_center_method=wcfg.crop_center_method,
            envelope_smooth_s=wcfg.envelope_smooth_s,
            pad_value=wcfg.pad_value,
        )
    if raw is None or proc is None:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # --- Top: raw original signal ---
    t_raw = np.arange(raw.signal.size) / raw.fs_hz
    axes[0].plot(t_raw, raw.signal, lw=0.4, color="steelblue")
    s0 = proc.crop_start_sample_original
    s1 = proc.crop_end_sample_original
    if proc.was_cropped:
        axes[0].axvspan(s0 / raw.fs_hz, s1 / raw.fs_hz, color="orange", alpha=0.25,
                        label="30 s crop window")
    axes[0].set_title(
        f"{label}  |  event={event_id}  |  dur={proc.duration_original_s:.1f} s  "
        f"|  padded={proc.was_padded}  cropped={proc.was_cropped}"
    )
    axes[0].set_xlabel("time [s]  (original, native fs)")
    axes[0].set_ylabel("strain (ch 1194, raw)")
    if proc.was_cropped:
        axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(True, ls=":", alpha=0.4)

    # --- Bottom: processed signal + crop ---
    fs_out = scfg.target_fs_hz
    t_proc = np.arange(proc.processed_signal.size) / fs_out
    axes[1].plot(t_proc, proc.processed_signal, lw=0.5, color="seagreen",
                 label="processed (1-100 Hz, 250 Hz)")
    cs = proc.crop_start_processed
    ce = proc.crop_end_processed
    if proc.was_cropped:
        axes[1].axvspan(max(cs, 0) / fs_out, ce / fs_out, color="orange", alpha=0.25,
                        label="selected 30 s crop")
    axes[1].axvline(proc.center_index_processed / fs_out, color="red", lw=1.2, ls="--",
                    label="energy centre")
    axes[1].set_xlabel("time [s]  (processed, 250 Hz)")
    axes[1].set_ylabel("strain (ch 1194)")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _select_representative_events(index_df: pd.DataFrame) -> Dict[str, str]:
    """Pick representative event_ids: short / median / ~p95 / very long / hi/lo PGV."""
    ok = index_df[index_df["build_status"] == "ok"].copy()
    if ok.empty:
        return {}
    selected: Dict[str, str] = {}

    dur = ok["duration_original_s"]
    selected["short"] = ok.loc[dur.idxmin(), "event_id"]
    selected["very_long"] = ok.loc[dur.idxmax(), "event_id"]
    median_dur = dur.median()
    selected["median"] = ok.loc[(dur - median_dur).abs().idxmin(), "event_id"]
    p95_dur = dur.quantile(0.95)
    selected["p95_long"] = ok.loc[(dur - p95_dur).abs().idxmin(), "event_id"]

    if "pgv_max" in ok.columns and ok["pgv_max"].notna().any():
        selected["high_pgv"] = ok.loc[ok["pgv_max"].idxmax(), "event_id"]
        selected["low_pgv"] = ok.loc[ok["pgv_max"].idxmin(), "event_id"]

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = CONFIG
    log_lines: List[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        log_lines.append(msg)

    T = cfg.fixed_length_samples
    scfg, wcfg, tcfg = cfg.signal, cfg.window, cfg.target

    log("=" * 70)
    log("Build Waveform v1 — single-channel FO time-series dataset")
    log("=" * 70)
    log(f"Channel      : {scfg.target_channel}")
    log(f"Bandpass     : {scfg.bandpass_freqmin}-{scfg.bandpass_freqmax} Hz "
        f"(order {scfg.bandpass_corners})")
    log(f"Resample     : {scfg.native_fs_hz:.0f} -> {scfg.target_fs_hz:.0f} Hz")
    log(f"Fixed length : {wcfg.fixed_length_s:.0f} s = {T:,} samples")
    log(f"Crop centre  : {wcfg.crop_center_method}")

    # ------------------------------------------------------------------
    # 1. Canonical event set + metadata from Parquet v2
    # ------------------------------------------------------------------
    log("\n[1/5] Loading canonical event set from Parquet v2 ...")
    event_meta, sensor_df = _load_canonical_events(cfg)
    canonical_ids = set(event_meta["event_id"].astype(str))
    meta_by_id = event_meta.set_index("event_id")
    log(f"      Canonical events : {len(canonical_ids):,}")
    log(f"      Sensor rows      : {len(sensor_df):,}")

    # ------------------------------------------------------------------
    # 2. Discover NetCDF files
    # ------------------------------------------------------------------
    log("\n[2/5] Discovering NetCDF files ...")
    netcdf_files: List[Path] = []
    for folder in cfg.input_folder_paths():
        if folder.exists():
            netcdf_files.extend(sorted(folder.glob(cfg.netcdf_glob())))
    netcdf_files = sorted(set(netcdf_files))
    if cfg.max_files > 0:
        netcdf_files = netcdf_files[: cfg.max_files]
        log(f"      [DEBUG] max_files={cfg.max_files}")
    log(f"      NetCDF files found: {len(netcdf_files):,}")

    # ------------------------------------------------------------------
    # 3. Process each canonical event
    # ------------------------------------------------------------------
    log("\n[3/5] Processing waveforms ...")
    wave_by_id: Dict[str, np.ndarray] = {}
    path_by_id: Dict[str, Path] = {}
    records: List[Dict] = []
    skipped_reasons: Dict[str, int] = {}
    open_failures: List[str] = []

    n_seen = 0
    for i, nc_path in enumerate(netcdf_files, 1):
        if i % 100 == 0 or i == len(netcdf_files):
            print(f"      {i}/{len(netcdf_files)} ...", end="\r")

        # Open with retry so a transient network drop does not silently drop
        # the event (which would otherwise be miscounted as "not found").
        try:
            ds = _open_netcdf_with_retry(nc_path)
        except Exception as exc:
            open_failures.append(f"{nc_path} :: {type(exc).__name__}: {exc}")
            continue

        eid = None
        proc = None
        reason = "ok"
        try:
            with ds:
                eid = _read_event_id(ds)
                if eid not in canonical_ids or eid in wave_by_id:
                    continue
                n_seen += 1
                proc, reason = process_event_waveform(
                    dataset=ds,
                    target_channel=scfg.target_channel,
                    bandpass_freqmin=scfg.bandpass_freqmin,
                    bandpass_freqmax=scfg.bandpass_freqmax,
                    bandpass_corners=scfg.bandpass_corners,
                    target_fs_hz=scfg.target_fs_hz,
                    fixed_length_samples=T,
                    crop_center_method=wcfg.crop_center_method,
                    envelope_smooth_s=wcfg.envelope_smooth_s,
                    pad_value=wcfg.pad_value,
                )
        except Exception as exc:
            reason = f"process_failed:{type(exc).__name__}"
            proc = None

        if eid is None or eid not in canonical_ids:
            continue

        if proc is None:
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            records.append({"event_id": eid, "build_status": reason})
            continue

        wave_by_id[eid] = proc.waveform
        path_by_id[eid] = nc_path
        records.append(
            {
                "event_id": eid,
                "build_status": "ok",
                "n_samples_original": proc.n_samples_original,
                "duration_original_s": proc.duration_original_s,
                "channel_index": proc.channel_index,
                "was_padded": proc.was_padded,
                "was_cropped": proc.was_cropped,
                "crop_start_sample_original": proc.crop_start_sample_original,
                "crop_end_sample_original": proc.crop_end_sample_original,
                "crop_center_method": proc.crop_center_method,
            }
        )
    print()

    # Canonical events that were never found in any NetCDF file.
    found_ids = {r["event_id"] for r in records}
    missing_ids = canonical_ids - found_ids
    for eid in missing_ids:
        records.append({"event_id": eid, "build_status": "not_found_in_netcdf"})

    # ------------------------------------------------------------------
    # 4. Assemble arrays + index (sorted by event_id for determinism)
    # ------------------------------------------------------------------
    log("\n[4/5] Assembling dataset ...")
    ok_ids = sorted(wave_by_id.keys())
    waveforms = np.zeros((len(ok_ids), T), dtype=np.float32)
    row_by_id: Dict[str, int] = {}
    for row_idx, eid in enumerate(ok_ids):
        waveforms[row_idx] = wave_by_id[eid]
        row_by_id[eid] = row_idx

    index_df = pd.DataFrame(records)
    index_df["waveform_row_idx"] = index_df["event_id"].map(row_by_id).astype("Int64")

    # Attach metadata (left join on event_id).
    meta_attach_cols = [c for c in meta_by_id.columns]
    index_df = index_df.merge(
        meta_by_id.reset_index()[["event_id"] + meta_attach_cols],
        on="event_id",
        how="left",
    )

    # Missing-speed flag for downstream training (impute at train time only).
    if "train_speed_kmh" in index_df.columns:
        index_df["train_speed_kmh_is_missing"] = (
            index_df["train_speed_kmh"].isna().astype(int)
        )

    # Sort: ok rows first (by row idx), then the rest.
    index_df = index_df.sort_values(
        ["build_status", "waveform_row_idx"], na_position="last"
    ).reset_index(drop=True)

    ok_mask = index_df["build_status"] == "ok"
    n_ok = int(ok_mask.sum())
    n_padded = int((ok_mask & (index_df.get("was_padded") == True)).sum())  # noqa: E712
    n_cropped = int((ok_mask & (index_df.get("was_cropped") == True)).sum())  # noqa: E712

    log(f"      Events with waveform : {n_ok:,}")
    log(f"      Padded (short)       : {n_padded:,}  ({n_padded / max(n_ok,1) * 100:.1f}%)")
    log(f"      Cropped (long)       : {n_cropped:,}  ({n_cropped / max(n_ok,1) * 100:.1f}%)")
    log(f"      Exact length         : {n_ok - n_padded - n_cropped:,}")
    if open_failures:
        log(f"      File open failures   : {len(open_failures):,} (network/IO)")
    if missing_ids:
        log(f"      Canonical not found  : {len(missing_ids):,}")
    if skipped_reasons:
        log("      Skipped reasons:")
        for r, c in sorted(skipped_reasons.items()):
            log(f"        {r}: {c}")

    # Completeness check — a flaky network drive can silently drop events.
    n_canonical = len(canonical_ids)
    coverage = n_ok / max(n_canonical, 1)
    if coverage < 0.99:
        log("")
        log("  " + "!" * 60)
        log(f"  WARNING: only {n_ok:,}/{n_canonical:,} canonical events "
            f"({coverage * 100:.1f}%) have waveforms.")
        log(f"  {len(open_failures):,} file open failures and "
            f"{len(missing_ids):,} not-found suggest a network-drive issue.")
        log("  Re-run the build once the drive is stable for a complete dataset.")
        log("  " + "!" * 60)

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    log("\n[5/5] Saving outputs ...")
    build_dir = _create_build_folder(cfg.output_root_path(), cfg.output.version_name)
    plots_dir = build_dir / cfg.output.plots_subfolder
    plots_dir.mkdir(parents=True, exist_ok=True)

    np.save(build_dir / cfg.output.waveforms_filename, waveforms)
    log(f"      -> {cfg.output.waveforms_filename}  shape={waveforms.shape} "
        f"({waveforms.nbytes / 1024 / 1024:.1f} MB)")

    index_path = build_dir / cfg.output.event_index_filename
    index_df.to_parquet(index_path, index=False)
    log(f"      -> {cfg.output.event_index_filename}  ({len(index_df):,} rows)")

    build_config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_events_with_waveform": n_ok,
        "n_padded": n_padded,
        "n_cropped": n_cropped,
        "fixed_length_samples": T,
        "fixed_length_s": wcfg.fixed_length_s,
        "target_fs_hz": scfg.target_fs_hz,
        "native_fs_hz": scfg.native_fs_hz,
        "target_channel": scfg.target_channel,
        "bandpass_hz": [scfg.bandpass_freqmin, scfg.bandpass_freqmax],
        "bandpass_corners": scfg.bandpass_corners,
        "crop_center_method": wcfg.crop_center_method,
        "envelope_smooth_s": wcfg.envelope_smooth_s,
        "r0_m": tcfg.r0_m,
        "distance_col": tcfg.distance_col,
        "pgv_col": tcfg.pgv_col,
        "exclude_sensor_ids": cfg.exclude_sensor_ids,
        "config": cfg.as_dict(),
        "skipped_reasons": skipped_reasons,
        "n_canonical_not_found": len(missing_ids),
        "n_open_failures": len(open_failures),
        "n_canonical_events": len(canonical_ids),
        "coverage_fraction": n_ok / max(len(canonical_ids), 1),
    }
    with open(build_dir / cfg.output.build_config_filename, "w", encoding="utf-8") as fh:
        json.dump(build_config, fh, indent=2, default=str)
    log(f"      -> {cfg.output.build_config_filename}")

    # Diagnostic crop plots
    log("\n      Drawing crop diagnostic plots ...")
    selected = _select_representative_events(index_df)
    for label, eid in selected.items():
        if eid in path_by_id:
            out_png = plots_dir / f"crop_diag_{label}.png"
            try:
                _plot_crop_diagnostic(eid, path_by_id[eid], cfg, label, out_png)
                log(f"        {label:10s} event={eid} -> {out_png.name}")
            except Exception as exc:
                log(f"        {label:10s} event={eid} FAILED: {exc}")

    with open(build_dir / cfg.output.build_log_filename, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines))

    print()
    print("=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    print(f"Build folder : {build_dir}")
    print(f"Waveforms    : {waveforms.shape}  ({waveforms.nbytes / 1024 / 1024:.1f} MB)")
    print(f"Events ok    : {n_ok:,}  |  padded {n_padded:,}  cropped {n_cropped:,}")


if __name__ == "__main__":
    main()
