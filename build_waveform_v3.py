"""Build the wide-aperture multi-channel FO waveform dataset (waveform v3).

Same engine as v2 but uses the full stored channel block (1165-1215, 50 m) with
an optional channel stride.  Output array shape: (N_events, n_channels, T).

Variants (override at run time):
    stride 1 -> (N, 51, T)   holten_waveform_v003_ch51
    stride 5 -> (N, 11, T)   holten_waveform_v003_ch11

Run:
    python build_waveform_v3.py
    python -c "import build_waveform_v3 as b; b.CONFIG.signal.channel_stride=5; \
               b.CONFIG.output.version_name='holten_waveform_v003_ch11'; b.main()"
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.db.waveform.config_waveform_v3 import CONFIG
from src.db.waveform.waveform_v2_utils import process_event_block
from src.utils.geometry_utils import apply_corrected_distances


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
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return nc.Dataset(nc_path, "r")
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(delay)
    raise last_exc if last_exc is not None else RuntimeError("open failed")


def _read_event_id(ds) -> str:
    return str(_to_python_scalar(getattr(ds, "event_id", "")))


def _load_canonical(cfg):
    parquet_root = cfg.parquet_root()
    v2_builds = sorted(parquet_root.glob("parquet_v002_*"), key=lambda p: p.name)
    if not v2_builds:
        raise FileNotFoundError(f"No parquet_v002_* builds in {parquet_root}")
    df = pd.read_parquet(v2_builds[-1] / "dataset.parquet")
    if cfg.exclude_sensor_ids and "sensor_id" in df.columns:
        df = df[~df["sensor_id"].isin(cfg.exclude_sensor_ids)]
    if "effective_distance_to_active_track_m" not in df.columns:
        df = apply_corrected_distances(df)
    tcfg = cfg.target
    meta_cols = [c for c in tcfg.metadata_cols if c in df.columns]
    keep = ["event_id"] + meta_cols + (["site_id"] if "site_id" in df.columns else [])
    event_meta = df[keep].groupby("event_id", sort=True).first().reset_index()
    pgv = (
        df.groupby("event_id")[tcfg.pgv_col]
        .agg(["max", "mean"]).rename(columns={"max": "pgv_max", "mean": "pgv_mean"})
        .reset_index()
    )
    return event_meta.merge(pgv, on="event_id", how="left")


def _plot_crop_diag(event_id, nc_path, cfg, label, out_path):
    scfg, wcfg = cfg.signal, cfg.window
    with nc.Dataset(nc_path, "r") as ds:
        proc, _ = process_event_block(
            dataset=ds, center_channel=scfg.center_channel,
            channel_lo=scfg.channel_lo, channel_hi=scfg.channel_hi,
            bandpass_freqmin=scfg.bandpass_freqmin, bandpass_freqmax=scfg.bandpass_freqmax,
            bandpass_corners=scfg.bandpass_corners, target_fs_hz=scfg.target_fs_hz,
            fixed_length_samples=cfg.fixed_length_samples,
            crop_center_method=wcfg.crop_center_method,
            envelope_smooth_s=wcfg.envelope_smooth_s, pad_value=wcfg.pad_value,
            expected_n_channels=scfg.n_channels, channel_stride=scfg.channel_stride,
        )
    if proc is None:
        return
    fs = scfg.target_fs_hz
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [2, 1]})
    pb = proc.processed_block
    vmax = np.percentile(np.abs(pb), 99) + 1e-12
    t_full = pb.shape[1] / fs
    axes[0].imshow(pb, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax,
                   extent=[0, t_full, scfg.channel_hi, scfg.channel_lo])
    if proc.was_cropped:
        axes[0].axvline(proc.crop_start_processed / fs, color="lime", lw=1.5)
        axes[0].axvline(proc.crop_end_processed / fs, color="lime", lw=1.5)
    axes[0].set_ylabel("FO channel ID")
    axes[0].set_title(
        f"{label}  |  event={event_id}  |  dur={proc.duration_original_s:.1f} s  "
        f"|  shape={proc.block.shape}  aperture={scfg.aperture_m:.0f} m"
    )
    ctr = pb[proc.center_pos]
    t = np.arange(ctr.size) / fs
    axes[1].plot(t, ctr, lw=0.5, color="seagreen")
    if proc.was_cropped:
        axes[1].axvspan(proc.crop_start_processed / fs, proc.crop_end_processed / fs,
                        color="orange", alpha=0.25)
    axes[1].axvline(proc.center_index_processed / fs, color="red", lw=1.2, ls="--")
    axes[1].set_xlabel("time [s] (250 Hz)")
    axes[1].set_ylabel("centre-channel strain")
    axes[1].grid(True, ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def _select_events(index_df):
    ok = index_df[index_df["build_status"] == "ok"].copy()
    if ok.empty:
        return {}
    sel = {}
    dur = ok["duration_original_s"]
    sel["short"] = ok.loc[dur.idxmin(), "event_id"]
    sel["very_long"] = ok.loc[dur.idxmax(), "event_id"]
    sel["median"] = ok.loc[(dur - dur.median()).abs().idxmin(), "event_id"]
    if "pgv_max" in ok.columns and ok["pgv_max"].notna().any():
        sel["high_pgv"] = ok.loc[ok["pgv_max"].idxmax(), "event_id"]
        sel["low_pgv"] = ok.loc[ok["pgv_max"].idxmin(), "event_id"]
    return sel


def main() -> None:
    cfg = CONFIG
    log_lines: List[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        log_lines.append(msg)

    T = cfg.fixed_length_samples
    scfg, wcfg, tcfg = cfg.signal, cfg.window, cfg.target
    n_ch = scfg.n_channels

    log("=" * 70)
    log("Build Waveform v3 — wide-aperture multi-channel FO dataset")
    log("=" * 70)
    log(f"Channels     : {scfg.channel_lo}-{scfg.channel_hi} stride {scfg.channel_stride} "
        f"-> {n_ch} channels @ {scfg.effective_pitch_m:.0f} m")
    log(f"Aperture     : {scfg.aperture_m:.0f} m  (gauge length {scfg.gauge_length_m:.0f} m)")
    log(f"Resample     : {scfg.native_fs_hz:.0f} -> {scfg.target_fs_hz:.0f} Hz")
    log(f"Fixed length : {wcfg.fixed_length_s:.0f} s = {T:,} samples")
    log(f"Tensor shape : (N_events, {n_ch}, {T})")

    log("\n[1/5] Loading canonical event set ...")
    event_meta = _load_canonical(cfg)
    canonical_ids = set(event_meta["event_id"].astype(str))
    meta_by_id = event_meta.set_index("event_id")
    log(f"      Canonical events : {len(canonical_ids):,}")

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

    log("\n[3/5] Processing blocks ...")
    block_by_id: Dict[str, np.ndarray] = {}
    path_by_id: Dict[str, Path] = {}
    records: List[Dict] = []
    skipped: Dict[str, int] = {}
    open_failures: List[str] = []

    for i, nc_path in enumerate(netcdf_files, 1):
        if i % 100 == 0 or i == len(netcdf_files):
            print(f"      {i}/{len(netcdf_files)} ...", end="\r")
        try:
            ds = _open_netcdf_with_retry(nc_path)
        except Exception as exc:
            open_failures.append(f"{nc_path} :: {exc}")
            continue
        eid = None
        proc = None
        reason = "ok"
        try:
            with ds:
                eid = _read_event_id(ds)
                if eid not in canonical_ids or eid in block_by_id:
                    continue
                proc, reason = process_event_block(
                    dataset=ds, center_channel=scfg.center_channel,
                    channel_lo=scfg.channel_lo, channel_hi=scfg.channel_hi,
                    bandpass_freqmin=scfg.bandpass_freqmin,
                    bandpass_freqmax=scfg.bandpass_freqmax,
                    bandpass_corners=scfg.bandpass_corners,
                    target_fs_hz=scfg.target_fs_hz, fixed_length_samples=T,
                    crop_center_method=wcfg.crop_center_method,
                    envelope_smooth_s=wcfg.envelope_smooth_s,
                    pad_value=wcfg.pad_value, expected_n_channels=n_ch,
                    channel_stride=scfg.channel_stride,
                )
        except Exception as exc:
            reason = f"process_failed:{type(exc).__name__}"
            proc = None
        if eid is None or eid not in canonical_ids:
            continue
        if proc is None:
            skipped[reason] = skipped.get(reason, 0) + 1
            records.append({"event_id": eid, "build_status": reason})
            continue
        block_by_id[eid] = proc.block
        path_by_id[eid] = nc_path
        records.append({
            "event_id": eid, "build_status": "ok",
            "n_samples_original": proc.n_samples_original,
            "duration_original_s": proc.duration_original_s,
            "n_channels": proc.n_channels,
            "was_padded": proc.was_padded, "was_cropped": proc.was_cropped,
            "crop_start_sample_original": proc.crop_start_sample_original,
            "crop_end_sample_original": proc.crop_end_sample_original,
            "crop_center_method": proc.crop_center_method,
        })
    print()

    found_ids = {r["event_id"] for r in records}
    missing_ids = canonical_ids - found_ids
    for eid in missing_ids:
        records.append({"event_id": eid, "build_status": "not_found_in_netcdf"})

    log("\n[4/5] Assembling dataset ...")
    ok_ids = sorted(block_by_id.keys())
    waveforms = np.zeros((len(ok_ids), n_ch, T), dtype=np.float32)
    row_by_id: Dict[str, int] = {}
    for row_idx, eid in enumerate(ok_ids):
        waveforms[row_idx] = block_by_id[eid]
        row_by_id[eid] = row_idx

    index_df = pd.DataFrame(records)
    index_df["waveform_row_idx"] = index_df["event_id"].map(row_by_id).astype("Int64")
    index_df = index_df.merge(
        meta_by_id.reset_index()[["event_id"] + list(meta_by_id.columns)],
        on="event_id", how="left",
    )
    if "train_speed_kmh" in index_df.columns:
        index_df["train_speed_kmh_is_missing"] = index_df["train_speed_kmh"].isna().astype(int)
    index_df = index_df.sort_values(
        ["build_status", "waveform_row_idx"], na_position="last"
    ).reset_index(drop=True)

    ok_mask = index_df["build_status"] == "ok"
    n_ok = int(ok_mask.sum())
    n_padded = int((ok_mask & (index_df.get("was_padded") == True)).sum())  # noqa: E712
    n_cropped = int((ok_mask & (index_df.get("was_cropped") == True)).sum())  # noqa: E712
    log(f"      Events with block : {n_ok:,}")
    log(f"      Padded / Cropped  : {n_padded:,} / {n_cropped:,}")
    if open_failures:
        log(f"      File open failures: {len(open_failures):,}")
    if missing_ids:
        log(f"      Canonical missing : {len(missing_ids):,}")
    if skipped:
        for r, c in sorted(skipped.items()):
            log(f"        {r}: {c}")

    coverage = n_ok / max(len(canonical_ids), 1)
    if coverage < 0.99:
        log("  " + "!" * 60)
        log(f"  WARNING: only {n_ok}/{len(canonical_ids)} events ({coverage*100:.1f}%).")
        log("  " + "!" * 60)

    log("\n[5/5] Saving outputs ...")
    build_dir = _create_build_folder(cfg.output_root_path(), cfg.output.version_name)
    plots_dir = build_dir / cfg.output.plots_subfolder
    plots_dir.mkdir(parents=True, exist_ok=True)

    np.save(build_dir / cfg.output.waveforms_filename, waveforms)
    log(f"      -> {cfg.output.waveforms_filename}  shape={waveforms.shape} "
        f"({waveforms.nbytes / 1024 / 1024:.1f} MB)")
    index_df.to_parquet(build_dir / cfg.output.event_index_filename, index=False)
    log(f"      -> {cfg.output.event_index_filename}  ({len(index_df):,} rows)")

    build_config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_events_with_block": n_ok, "n_padded": n_padded, "n_cropped": n_cropped,
        "fixed_length_samples": T, "fixed_length_s": wcfg.fixed_length_s,
        "target_fs_hz": scfg.target_fs_hz, "center_channel": scfg.center_channel,
        "channel_lo": scfg.channel_lo, "channel_hi": scfg.channel_hi,
        "channel_stride": scfg.channel_stride, "n_channels": n_ch,
        "effective_pitch_m": scfg.effective_pitch_m, "aperture_m": scfg.aperture_m,
        "gauge_length_m": scfg.gauge_length_m,
        "bandpass_hz": [scfg.bandpass_freqmin, scfg.bandpass_freqmax],
        "crop_center_method": wcfg.crop_center_method, "r0_m": tcfg.r0_m,
        "exclude_sensor_ids": cfg.exclude_sensor_ids,
        "skipped_reasons": skipped, "n_open_failures": len(open_failures),
        "n_canonical_events": len(canonical_ids), "coverage_fraction": coverage,
        "config": cfg.as_dict(),
    }
    with open(build_dir / cfg.output.build_config_filename, "w", encoding="utf-8") as fh:
        json.dump(build_config, fh, indent=2, default=str)
    log(f"      -> {cfg.output.build_config_filename}")

    log("\n      Drawing crop diagnostic plots ...")
    for label, eid in _select_events(index_df).items():
        if eid in path_by_id:
            out_png = plots_dir / f"crop_diag_{label}.png"
            try:
                _plot_crop_diag(eid, path_by_id[eid], cfg, label, out_png)
                log(f"        {label:10s} event={eid} -> {out_png.name}")
            except Exception as exc:
                log(f"        {label:10s} event={eid} FAILED: {exc}")

    with open(build_dir / cfg.output.build_log_filename, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines))

    print("\n" + "=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    print(f"Build folder : {build_dir}")
    print(f"Waveforms    : {waveforms.shape}  ({waveforms.nbytes / 1024 / 1024:.1f} MB)")
    print(f"Aperture     : {scfg.aperture_m:.0f} m  ({n_ch} ch @ {scfg.effective_pitch_m:.0f} m)")


if __name__ == "__main__":
    main()
