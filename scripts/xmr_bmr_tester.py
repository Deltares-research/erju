#!/usr/bin/env python3
"""
Review MR3000 monthly vibration data stored in ZIPs (Culemborg project).

Folder structure (user-described):
ROOT/
  metingen Caubergh 2025D_april/
    MT33_background.zip   -> contains MT33/*.BMR
    MT33events.zip        -> contains MT33events/<event_id>/*.XMR
    MT40_background.zip   -> contains MT40/*.BMR
    MT40events.zip        -> contains MT40events/<event_id>/*.XMR

This script:
- Enumerates all month folders under ROOT with name starting "metingen Caubergh "
- For each month folder, processes those 4 ZIPs (if they exist)
- Samples N files at random per ZIP (default 25)
- Reads XMR (binary) directly
- Reads BMR via vendor converter EXE -> TXT -> parse
- Saves plots (1 col x 3 rows) for each sampled file
- Saves a metadata CSV for sampled files + a consistency report per ZIP
- Writes a run log
- Writes a FINAL SUMMARY at the end to log + to final_summary_*.txt

Requirements:
  pip install numpy pandas matplotlib
"""

from __future__ import annotations

import csv
import logging
import random
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Logging
# ============================================================


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mr3000_review")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    ch = logging.StreamHandler()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ============================================================
# XMR (binary) reader
# ============================================================


def _bcd_to_int(byte_val: int) -> int:
    hi = (byte_val >> 4) & 0xF
    lo = byte_val & 0xF
    return hi * 10 + lo


def _read_bcd_datetime(f) -> datetime:
    secs = _bcd_to_int(int.from_bytes(f.read(1), "little"))
    mins = _bcd_to_int(int.from_bytes(f.read(1), "little"))
    hour = _bcd_to_int(int.from_bytes(f.read(1), "little"))
    day = _bcd_to_int(int.from_bytes(f.read(1), "little"))
    mon = _bcd_to_int(int.from_bytes(f.read(1), "little"))
    yy = _bcd_to_int(int.from_bytes(f.read(1), "little"))
    year = 2000 + yy
    return datetime(year, mon, day, hour, mins, secs)


def _read_int24_le_signed(buf: bytes) -> np.ndarray:
    if len(buf) % 3 != 0:
        raise ValueError("Buffer length is not a multiple of 3 bytes for int24.")
    b = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 3)
    val = (
        b[:, 0].astype(np.int32)
        | (b[:, 1].astype(np.int32) << 8)
        | (b[:, 2].astype(np.int32) << 16)
    )
    sign_bit = 1 << 23
    val = (val ^ sign_bit) - sign_bit
    return val.astype(np.int32)


@dataclass
class XMRHeader:
    nsamp: int
    filetype: int
    fs: float
    nchan: int
    dum: int
    trigger_time: datetime
    firm: int
    bit: int
    LSBx: float
    LSBy: float
    LSBz: float
    EUx: str
    EUy: str
    EUz: str
    namex: str
    namey: str
    namez: str
    trigX: float
    trigY: float
    trigZ: float
    nst: Optional[datetime]
    comment: str


def read_xmr(path: Path, aantalbits: int = 24) -> Tuple[pd.DataFrame, XMRHeader]:
    """
    Read XMR binary directly.

    If your device stores 20-bit values packed into 24-bit words, set aantalbits=20.
    """
    if not (1 <= aantalbits <= 24):
        raise ValueError("aantalbits must be in [1, 24]")
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("rb") as f:
        f.seek(8, 0)
        nsamp = int.from_bytes(f.read(4), "little", signed=False)

        f.seek(4, 0)
        filetype = int.from_bytes(f.read(1), "little", signed=False)

        f.seek(54, 0)
        fs = float(int.from_bytes(f.read(2), "little", signed=True))
        nchan = int.from_bytes(f.read(1), "little", signed=False)

        f.seek(33, 0)
        dum = int.from_bytes(f.read(2), "little", signed=True)

        dt0 = _read_bcd_datetime(f)  # reads 6 bytes after dum
        trigger_time = dt0 + timedelta(seconds=(dum / fs))

        f.seek(58, 0)
        firm = int.from_bytes(f.read(1), "little", signed=False)

        f.seek(63, 0)
        bit = int.from_bytes(f.read(1), "little", signed=False)

        f.seek(83, 0)

        def read_lsb() -> float:
            base = int.from_bytes(f.read(2), "little", signed=True)
            exp = int.from_bytes(f.read(1), "little", signed=True)
            return float(base) * (10.0 ** float(exp))

        LSBx = read_lsb()
        LSBy = read_lsb()
        LSBz = read_lsb()

        f.seek(92, 0)
        EUx = f.read(5).decode(errors="ignore").strip("\x00").strip()
        EUy = f.read(5).decode(errors="ignore").strip("\x00").strip()
        EUz = f.read(5).decode(errors="ignore").strip("\x00").strip()

        f.seek(107, 0)
        namex = f.read(3).decode(errors="ignore").strip("\x00").strip()
        namey = f.read(3).decode(errors="ignore").strip("\x00").strip()
        namez = f.read(3).decode(errors="ignore").strip("\x00").strip()

        f.seek(116, 0)
        trigX = float(_read_int24_le_signed(f.read(3))[0]) * LSBx
        trigY = float(_read_int24_le_signed(f.read(3))[0]) * LSBy
        trigZ = float(_read_int24_le_signed(f.read(3))[0]) * LSBz

        f.seek(143, 0)
        nst_offset_sec = int.from_bytes(f.read(1), "little", signed=False)
        nst = trigger_time - timedelta(seconds=nst_offset_sec)

        f.seek(145, 0)
        comment = f.read(30).decode(errors="ignore").strip("\x00").strip()

        f.seek(256, 0)
        raw_bytes = f.read(nsamp * nchan * 3)
        if len(raw_bytes) != nsamp * nchan * 3:
            raise ValueError("File ended unexpectedly while reading trace data")

        vals = _read_int24_le_signed(raw_bytes)
        if aantalbits < 24:
            shift = 24 - aantalbits
            vals = (vals >> shift).astype(np.int32)

        trace = vals.reshape(nsamp, nchan).astype(np.float64)

        scales = np.ones(nchan, dtype=float)
        if nchan >= 1:
            scales[0] = LSBx
        if nchan >= 2:
            scales[1] = LSBy
        if nchan >= 3:
            scales[2] = LSBz
        trace *= scales

        t_s = np.arange(nsamp, dtype=float) / fs
        df = pd.DataFrame({"t_s": t_s})
        df["datetime"] = trigger_time + pd.to_timedelta(df["t_s"], unit="s")

        if nchan >= 3:
            df["x"] = trace[:, 0]
            df["y"] = trace[:, 1]
            df["z"] = trace[:, 2]
        else:
            for i in range(nchan):
                df[f"ch{i+1}"] = trace[:, i]

        hdr = XMRHeader(
            nsamp=nsamp,
            filetype=filetype,
            fs=fs,
            nchan=nchan,
            dum=dum,
            trigger_time=trigger_time,
            firm=firm,
            bit=bit,
            LSBx=LSBx,
            LSBy=LSBy,
            LSBz=LSBz,
            EUx=EUx,
            EUy=EUy,
            EUz=EUz,
            namex=namex,
            namey=namey,
            namez=namez,
            trigX=trigX,
            trigY=trigY,
            trigZ=trigZ,
            nst=nst,
            comment=comment,
        )
        return df, hdr


# ============================================================
# BMR reader via converter -> TXT
# ============================================================


def convert_bmr_to_txt(bmr_path: Path, converter_exe: Path) -> Path:
    """
    Run: <converter_exe> <folder> txt
    (converter typically converts all BMRs in folder)

    We isolate each BMR in its own temp folder -> safe.
    """
    cmd = [str(converter_exe), str(bmr_path.parent), "txt"]
    cp = subprocess.run(cmd, capture_output=True, text=True)

    if cp.returncode != 0:
        raise RuntimeError(
            "BMR conversion failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{cp.stdout}\n"
            f"STDERR:\n{cp.stderr}\n"
        )

    candidate = bmr_path.with_suffix(".txt")
    if candidate.exists():
        return candidate

    txts = sorted(
        bmr_path.parent.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not txts:
        raise FileNotFoundError("Converter reported success but no .txt found.")
    return txts[0]


def read_bmr_txt(txt_path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    header: Dict[str, str] = {}

    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = re.match(r"^\#\s*([^=]+)\s*=\s*(.*)\s*$", line.strip())
            if m:
                header[m.group(1).strip()] = m.group(2).strip()

    df = pd.read_csv(txt_path, comment="#", sep=r"\s+", header=None, engine="python")
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.shape[1] < 4:
        raise ValueError(
            f"Expected >=4 numeric columns in {txt_path}, got {df.shape[1]}"
        )

    df = df.rename(columns={0: "t_s", 1: "x", 2: "y", 3: "z"})

    # Optional absolute datetime if StartDate/StartTime exist
    start_date = header.get("StartDate")
    start_time = header.get("StartTime")
    if start_date and start_time:
        parsed_date = None
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%Y.%m.%d"):
            try:
                parsed_date = datetime.strptime(start_date.strip(), fmt).date()
                break
            except ValueError:
                pass

        parsed_time = None
        for tfmt in ("%H:%M:%S", "%H:%M"):
            try:
                parsed_time = datetime.strptime(start_time.strip(), tfmt).time()
                break
            except ValueError:
                pass

        if parsed_date and parsed_time:
            t0 = datetime.combine(parsed_date, parsed_time)
            df["datetime"] = t0 + pd.to_timedelta(df["t_s"], unit="s")

    return df, header


def read_bmr_from_isolated_file(
    bmr_path: Path, converter_exe: Path
) -> Tuple[pd.DataFrame, Dict[str, str], Path]:
    txt_path = convert_bmr_to_txt(bmr_path, converter_exe)
    df, header = read_bmr_txt(txt_path)
    return df, header, txt_path


# ============================================================
# Plotting
# ============================================================


def plot_xyz(
    df: pd.DataFrame, out_png: Path, title: str, use_datetime: bool = False
) -> None:
    if use_datetime and "datetime" in df.columns:
        x = df["datetime"]
        xlabel = "Time"
    else:
        x = df["t_s"]
        xlabel = "Time [s]"

    if all(c in df.columns for c in ["x", "y", "z"]):
        cols = ["x", "y", "z"]
    elif all(c in df.columns for c in ["ch1", "ch2", "ch3"]):
        cols = ["ch1", "ch2", "ch3"]
    else:
        raise ValueError("Expected x/y/z or ch1/ch2/ch3 columns.")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    for ax, c in zip(axes, cols):
        ax.plot(x, df[c], linewidth=0.8)
        ax.set_ylabel(c)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title)
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def signal_stats(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in ["x", "y", "z"]:
        if c in df.columns:
            arr = df[c].to_numpy(dtype=float)
            out[f"{c}_min"] = float(np.nanmin(arr))
            out[f"{c}_max"] = float(np.nanmax(arr))
            out[f"{c}_rms"] = float(np.sqrt(np.nanmean(arr**2)))
    return out


# ============================================================
# ZIP helpers
# ============================================================


@dataclass(frozen=True)
class ZipMember:
    zip_path: Path
    internal_path: str  # path inside zip


def list_zip_members(zip_path: Path, suffix: str) -> List[ZipMember]:
    suffix = suffix.lower()
    out: List[ZipMember] = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.lower().endswith(suffix):
                out.append(ZipMember(zip_path=zip_path, internal_path=name))
    return out


def extract_single_member(member: ZipMember, target_dir: Path) -> Path:
    """
    Extract exactly one member into target_dir, preserving internal folders.
    Returns full extracted path.
    """
    with zipfile.ZipFile(member.zip_path, "r") as z:
        z.extract(member.internal_path, path=target_dir)
    return target_dir / member.internal_path


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


# ============================================================
# Consistency checks + final summary helpers
# ============================================================


def consistency_report(rows: List[Dict[str, object]], fields: List[str]) -> str:
    lines = []
    for f in fields:
        vals = [r.get(f) for r in rows if r.get("status") == "OK"]
        if not vals:
            lines.append(f"- {f}: (no OK records)")
            continue
        norm = [str(v) for v in vals]
        counts: Dict[str, int] = {}
        for v in norm:
            counts[v] = counts.get(v, 0) + 1
        items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top = ", ".join([f"{k} ({n})" for k, n in items[:10]])
        extra = "" if len(items) <= 10 else f" … +{len(items)-10} more"
        lines.append(f"- {f}: {top}{extra}")
    return "\n".join(lines)


def find_inconsistencies(
    rows: List[Dict[str, object]], fields: List[str]
) -> Dict[str, Dict[str, int]]:
    """
    For each field, count unique values among OK rows.
    Returns only fields that have >1 unique value.
    """
    inconsistencies: Dict[str, Dict[str, int]] = {}
    ok_rows = [r for r in rows if r.get("status") == "OK"]
    if not ok_rows:
        return inconsistencies

    for f in fields:
        counts: Dict[str, int] = {}
        for r in ok_rows:
            v = r.get(f)
            key = "<MISSING>" if (v is None or v == "") else str(v)
            counts[key] = counts.get(key, 0) + 1
        if len(counts) > 1:
            inconsistencies[f] = counts

    return inconsistencies


def format_inconsistencies(inc: Dict[str, Dict[str, int]]) -> str:
    if not inc:
        return "  (none)"
    lines = []
    for field, counts in inc.items():
        items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        parts = ", ".join([f"{k} ({n})" for k, n in items])
        lines.append(f"  - {field}: {parts}")
    return "\n".join(lines)


# ============================================================
# Main review logic
# ============================================================

MONTH_PREFIX = "metingen Caubergh "

ZIP_SPECS = [
    ("MT33_background.zip", ".bmr", "MT33_background"),
    ("MT33events.zip", ".xmr", "MT33events"),
    ("MT40_background.zip", ".bmr", "MT40_background"),
    ("MT40events.zip", ".xmr", "MT40events"),
]


def find_month_folders(root: Path) -> List[Path]:
    months = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith(MONTH_PREFIX):
            months.append(p)
    return sorted(months, key=lambda x: x.name.lower())


def write_rows_csv(csv_path: Path, rows: List[Dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = set()
    for r in rows:
        keys.update(r.keys())
    fieldnames = sorted(keys)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def review_one_zip(
    month_folder: Path,
    zip_path: Path,
    expected_suffix: str,
    out_dir: Path,
    logger: logging.Logger,
    n_samples: int,
    rng: random.Random,
    converter_exe: Optional[Path],
    xmr_bits: int,
    use_datetime_plot: bool,
) -> List[Dict[str, object]]:
    """
    Process one zip: list files, sample, read, plot, log metadata.
    Returns list of row dicts for metadata_sampled.csv.
    """
    rows: List[Dict[str, object]] = []

    if not zip_path.exists():
        logger.warning(f"Missing zip: {zip_path}")
        return rows

    try:
        members = list_zip_members(zip_path, expected_suffix)
    except zipfile.BadZipFile:
        logger.error(f"Bad/corrupt zip: {zip_path}")
        return rows

    if not members:
        logger.warning(f"No {expected_suffix} files inside: {zip_path}")
        return rows

    chosen = members if len(members) <= n_samples else rng.sample(members, n_samples)
    logger.info(f"{zip_path.name}: found {len(members)} files, sampling {len(chosen)}")

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for i, m in enumerate(chosen, start=1):
        tmpdir = Path(tempfile.mkdtemp(prefix="mr3000_review_"))
        row: Dict[str, object] = {
            "month": month_folder.name,
            "zip": zip_path.name,
            "internal_path": m.internal_path,
            "suffix": expected_suffix,
            "status": "FAILED",
            "error": "",
        }

        try:
            extracted = extract_single_member(m, tmpdir)

            # Isolate file at temp root for converter simplicity
            isolated_file = tmpdir / Path(m.internal_path).name
            shutil.copy2(extracted, isolated_file)

            base_title = (
                f"{month_folder.name} | {zip_path.name} | {Path(m.internal_path).name}"
            )

            if expected_suffix.lower() == ".xmr":
                df, hdr = read_xmr(isolated_file, aantalbits=xmr_bits)
                stats = signal_stats(df)

                row.update(
                    {
                        "status": "OK",
                        "filetype": "XMR",
                        "nsamp": hdr.nsamp,
                        "fs": hdr.fs,
                        "nchan": hdr.nchan,
                        "bit": hdr.bit,
                        "EUx": hdr.EUx,
                        "EUy": hdr.EUy,
                        "EUz": hdr.EUz,
                        "LSBx": hdr.LSBx,
                        "LSBy": hdr.LSBy,
                        "LSBz": hdr.LSBz,
                        "comment": hdr.comment,
                        **stats,
                    }
                )

                plot_name = safe_filename(f"{i:02d}_{Path(m.internal_path).name}.png")
                plot_path = plots_dir / plot_name
                plot_xyz(
                    df,
                    plot_path,
                    title=f"XMR | {base_title} | fs={hdr.fs}Hz",
                    use_datetime=use_datetime_plot,
                )
                row["plot_png"] = str(plot_path)

            elif expected_suffix.lower() == ".bmr":
                if converter_exe is None or not converter_exe.exists():
                    raise FileNotFoundError(
                        "converter_exe not set/found (needed for BMR)."
                    )

                df, hdr, txt_used = read_bmr_from_isolated_file(
                    isolated_file, converter_exe
                )
                stats = signal_stats(df)

                row.update(
                    {
                        "status": "OK",
                        "filetype": "BMR",
                        "Rate": hdr.get("Rate"),
                        "Channels": hdr.get("Channels"),
                        "Unit": hdr.get("Unit"),
                        "Filter": hdr.get("Filter"),
                        "Range": hdr.get("Range"),
                        "Scale": hdr.get("Scale"),
                        "Mode": hdr.get("Mode"),
                        "StartDate": hdr.get("StartDate"),
                        "StartTime": hdr.get("StartTime"),
                        "Samples": hdr.get("Samples"),
                        "Length": hdr.get("Length"),
                        "txt_used": str(txt_used),
                        **stats,
                    }
                )

                plot_name = safe_filename(f"{i:02d}_{Path(m.internal_path).name}.png")
                plot_path = plots_dir / plot_name
                plot_xyz(
                    df,
                    plot_path,
                    title=f"BMR | {base_title}",
                    use_datetime=use_datetime_plot,
                )
                row["plot_png"] = str(plot_path)

            else:
                raise ValueError(f"Unexpected suffix rule: {expected_suffix}")

        except Exception as e:
            row["error"] = str(e)
            logger.error(
                f"FAILED: {month_folder.name} | {zip_path.name} | {m.internal_path} | {e}"
            )

        finally:
            rows.append(row)
            shutil.rmtree(tmpdir, ignore_errors=True)

    # Per-zip outputs
    meta_csv = out_dir / "metadata_sampled.csv"
    write_rows_csv(meta_csv, rows)

    if expected_suffix.lower() == ".xmr":
        fields = ["fs", "nchan", "bit", "EUx", "EUy", "EUz", "LSBx", "LSBy", "LSBz"]
    else:
        fields = ["Rate", "Channels", "Unit", "Filter", "Range", "Scale", "Mode"]

    rep = consistency_report(rows, fields)
    (out_dir / "consistency_report.txt").write_text(rep, encoding="utf-8")

    return rows


def run_full_review(
    root: Path,
    out_root: Path,
    converter_exe: Path,
    n_samples_per_zip: int = 25,
    seed: int = 42,
    xmr_bits: int = 24,
    use_datetime_plot: bool = False,
) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_root / f"run_log_{ts}.txt")
    logger.info(f"ROOT: {root}")
    logger.info(f"OUT:  {out_root}")
    logger.info(f"SAMPLES per ZIP: {n_samples_per_zip}")
    logger.info(f"SEED: {seed}")
    logger.info(f"XMR bits: {xmr_bits}")
    logger.info(f"Use datetime in plots: {use_datetime_plot}")
    logger.info(f"Converter EXE: {converter_exe}")

    if not root.exists():
        raise FileNotFoundError(root)

    if not converter_exe.exists():
        raise FileNotFoundError(f"Converter EXE not found: {converter_exe}")

    months = find_month_folders(root)
    logger.info(f"Found {len(months)} month folders.")

    rng = random.Random(seed)

    aggregate_rows: List[Dict[str, object]] = []
    zip_summaries: List[Dict[str, object]] = []

    for month in months:
        logger.info(f"=== Month: {month.name} ===")
        month_out = out_root / month.name
        month_out.mkdir(parents=True, exist_ok=True)

        for zip_name, suffix, label in ZIP_SPECS:
            zip_path = month / zip_name
            zip_out = month_out / label
            zip_out.mkdir(parents=True, exist_ok=True)

            logger.info(f"-- Processing: {month.name} | {zip_name}")

            rows = review_one_zip(
                month_folder=month,
                zip_path=zip_path,
                expected_suffix=suffix,
                out_dir=zip_out,
                logger=logger,
                n_samples=n_samples_per_zip,
                rng=rng,
                converter_exe=converter_exe,
                xmr_bits=xmr_bits,
                use_datetime_plot=use_datetime_plot,
            )
            aggregate_rows.extend(rows)

            ok_rows = [r for r in rows if r.get("status") == "OK"]
            fail_rows = [r for r in rows if r.get("status") != "OK"]

            if suffix.lower() == ".xmr":
                fields = [
                    "fs",
                    "nchan",
                    "bit",
                    "EUx",
                    "EUy",
                    "EUz",
                    "LSBx",
                    "LSBy",
                    "LSBz",
                ]
            else:
                fields = [
                    "Rate",
                    "Channels",
                    "Unit",
                    "Filter",
                    "Range",
                    "Scale",
                    "Mode",
                ]

            inc = find_inconsistencies(rows, fields)

            zip_summaries.append(
                {
                    "month": month.name,
                    "zip": zip_name,
                    "label": label,
                    "n_sampled": len(rows),
                    "n_ok": len(ok_rows),
                    "n_failed": len(fail_rows),
                    "has_inconsistency": len(inc) > 0,
                    "inconsistencies": inc,
                }
            )

    # Write global aggregate CSV
    agg_csv = out_root / f"aggregate_metadata_{ts}.csv"
    write_rows_csv(agg_csv, aggregate_rows)

    # =====================================================
    # Final summary written to log + file
    # (counts only ZIPs that actually had sampled files)
    # =====================================================
    total_sampled = len(aggregate_rows)
    total_ok = sum(1 for r in aggregate_rows if r.get("status") == "OK")
    total_failed = total_sampled - total_ok

    zips_sampled = sum(1 for z in zip_summaries if z["n_sampled"] > 0)
    zips_with_failures = sum(
        1 for z in zip_summaries if z["n_sampled"] > 0 and z["n_failed"] > 0
    )
    zips_with_inconsistencies = sum(
        1 for z in zip_summaries if z["n_sampled"] > 0 and z["has_inconsistency"]
    )

    summary_lines: List[str] = []
    summary_lines.append("========================================")
    summary_lines.append("FINAL SUMMARY")
    summary_lines.append("========================================")
    summary_lines.append(f"Months processed: {len(months)}")
    summary_lines.append(f"ZIP groups sampled: {zips_sampled}")
    summary_lines.append(f"Total files sampled: {total_sampled}")
    summary_lines.append(f"OK: {total_ok}")
    summary_lines.append(f"FAILED: {total_failed}")
    summary_lines.append(f"ZIP groups with any FAILED files: {zips_with_failures}")
    summary_lines.append(
        f"ZIP groups with metadata inconsistencies: {zips_with_inconsistencies}"
    )
    summary_lines.append("")

    if zips_with_inconsistencies == 0 and total_failed == 0:
        summary_lines.append(
            "RESULT: PASS — No failures and no inconsistencies detected in the sampled files."
        )
    elif zips_with_inconsistencies == 0 and total_failed > 0:
        summary_lines.append(
            "RESULT: WARN — Some sampled files FAILED to read/plot, but metadata was consistent among OK files."
        )
    elif zips_with_inconsistencies > 0 and total_failed == 0:
        summary_lines.append(
            "RESULT: WARN — All sampled files read/plot OK, but metadata inconsistencies were detected."
        )
    else:
        summary_lines.append(
            "RESULT: FAIL — Failures occurred and metadata inconsistencies were detected."
        )
    summary_lines.append("")

    if zips_with_inconsistencies > 0:
        summary_lines.append("INCONSISTENCIES (per month / zip):")
        for z in zip_summaries:
            if z["n_sampled"] > 0 and z["has_inconsistency"]:
                summary_lines.append(
                    f"- {z['month']} | {z['zip']} | sampled={z['n_sampled']} OK={z['n_ok']} FAILED={z['n_failed']}"
                )
                summary_lines.append(format_inconsistencies(z["inconsistencies"]))
        summary_lines.append("")

    if zips_with_failures > 0:
        summary_lines.append("ZIP GROUPS WITH FAILURES:")
        for z in zip_summaries:
            if z["n_sampled"] > 0 and z["n_failed"] > 0:
                summary_lines.append(
                    f"- {z['month']} | {z['zip']} | OK={z['n_ok']} FAILED={z['n_failed']}"
                )
        summary_lines.append("")

    final_summary = "\n".join(summary_lines)

    logger.info("\n" + final_summary)

    summary_path = out_root / f"final_summary_{ts}.txt"
    summary_path.write_text(final_summary, encoding="utf-8")
    logger.info(f"Final summary written to: {summary_path}")

    logger.info(f"Done. Aggregate metadata: {agg_csv}")
    logger.info("Review outputs saved under month/zip subfolders.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    # --------- EDIT THESE PATHS ----------
    ROOT = Path(r"P:\11207352-stem\Measurement_Culemborg")
    OUT_ROOT = Path(r"P:\11207352-stem\Measurement_Culemborg\data_review")

    # MUST be the EXE, not the .bat
    CONVERTER_EXE = Path(
        r"P:\11207352-stem\Measurement_Culemborg\mr3000-convert-arch32.exe"
    )
    # ------------------------------------

    N_SAMPLES_PER_ZIP = 5
    SEED = 123

    # If XMR amplitudes look wrong/shifted, try 20
    XMR_BITS = 24

    # Use seconds (False) or absolute datetime if present (True)
    USE_DATETIME_PLOT = False

    run_full_review(
        root=ROOT,
        out_root=OUT_ROOT,
        converter_exe=CONVERTER_EXE,
        n_samples_per_zip=N_SAMPLES_PER_ZIP,
        seed=SEED,
        xmr_bits=XMR_BITS,
        use_datetime_plot=USE_DATETIME_PLOT,
    )
