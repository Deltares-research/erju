#!/usr/bin/env python3
"""
MR3000 / Caubergh Huygens red-box readers

- XMR: read binary directly (based on colleague's readxmr.m)
- BMR: convert .BMR -> .txt using mr3000-convert-arch32.exe, then parse ASCII (based on readBMR.m)

Requirements:
  pip install pandas numpy

Notes:
- BMR conversion needs the converter EXE path (absolute path!)
- XMR reading does NOT require the converter
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ============================================================
# Helpers
# ============================================================


def _bcd_to_int(byte_val: int) -> int:
    """Convert one BCD-encoded byte to int, e.g. 0x25 -> 25."""
    hi = (byte_val >> 4) & 0xF
    lo = byte_val & 0xF
    return hi * 10 + lo


def _read_bcd_datetime(f) -> datetime:
    """
    readxmr.m reads 6 bytes: secs, mins, hour, day, month, year (two-digit year)
    and formats "dd-mm-20yy HH:MM:SS"
    """
    secs = _bcd_to_int(int.from_bytes(f.read(1), "little"))
    mins = _bcd_to_int(int.from_bytes(f.read(1), "little"))
    hour = _bcd_to_int(int.from_bytes(f.read(1), "little"))
    day = _bcd_to_int(int.from_bytes(f.read(1), "little"))
    mon = _bcd_to_int(int.from_bytes(f.read(1), "little"))
    yy = _bcd_to_int(int.from_bytes(f.read(1), "little"))
    year = 2000 + yy
    return datetime(year, mon, day, hour, mins, secs)


def _read_int24_le_signed(buf: bytes) -> np.ndarray:
    """
    Interpret a buffer as little-endian signed 24-bit integers.
    buf length must be multiple of 3.
    Returns int32 numpy array.
    """
    if len(buf) % 3 != 0:
        raise ValueError("Buffer length is not a multiple of 3 bytes for int24.")

    b = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 3)
    # little-endian: value = b0 + b1<<8 + b2<<16
    val = (
        b[:, 0].astype(np.int32)
        | (b[:, 1].astype(np.int32) << 8)
        | (b[:, 2].astype(np.int32) << 16)
    )

    # sign-extend from 24 bits
    sign_bit = 1 << 23
    val = (val ^ sign_bit) - sign_bit
    return val.astype(np.int32)


# ============================================================
# XMR (binary) reader
# ============================================================


@dataclass
class XMRHeader:
    nsamp: int
    filetype: int
    fs: float
    nchan: int
    dum: int
    ntr: datetime
    dateTime_str: str
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


def read_xmr(
    path: Union[str, Path], aantalbits: int = 24
) -> Tuple[pd.DataFrame, XMRHeader]:
    """
    Read XMR binary file (Firmware 620.70 style) similar to readxmr.m

    Parameters
    ----------
    path : str | Path
    aantalbits : int
        Number of significant bits in each 24-bit word (default 24).
        Older scripts used 20 bits. If your device stores 20-bit values in 24-bit words,
        set aantalbits=20.

    Returns
    -------
    df : pandas.DataFrame
        Columns: t_s, datetime, ch1..chN (or x,y,z if nchan==3)
    header : XMRHeader
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if not (1 <= aantalbits <= 24):
        raise ValueError("aantalbits must be between 1 and 24")

    with path.open("rb") as f:
        # Offsets exactly as in readxmr.m
        f.seek(8, 0)
        nsamp = int.from_bytes(f.read(4), "little", signed=False)

        f.seek(4, 0)
        filetype = int.from_bytes(f.read(1), "little", signed=False)

        f.seek(54, 0)
        fs = int.from_bytes(f.read(2), "little", signed=True)
        nchan = int.from_bytes(f.read(1), "little", signed=False)

        f.seek(33, 0)
        dum = int.from_bytes(f.read(2), "little", signed=True)

        # After reading dum, read BCD datetime (6 bytes)
        dt0 = _read_bcd_datetime(f)

        # readxmr.m: H.ntr = readdate + (dum/fs)/60/60/24
        # That means dt0 is "start of trace minus dum/fs", and dt0 + dum/fs gives trigger time.
        # We'll mirror their logic:
        trigger_time = dt0 + timedelta(seconds=(dum / fs))
        dateTime_str = trigger_time.strftime("%Y-%m-%d_%H-%M-%S")

        f.seek(58, 0)
        firm = int.from_bytes(f.read(1), "little", signed=False)

        f.seek(63, 0)
        bit = int.from_bytes(f.read(1), "little", signed=False)

        # LSB scaling (short * 10^(int8))
        f.seek(83, 0)

        def read_lsb():
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

        # Triggers are read as bit24 and scaled by LSB
        f.seek(116, 0)
        trig_raw = _read_int24_le_signed(f.read(3))
        trigX = float(trig_raw[0]) * LSBx
        trig_raw = _read_int24_le_signed(f.read(3))
        trigY = float(trig_raw[0]) * LSBy
        trig_raw = _read_int24_le_signed(f.read(3))
        trigZ = float(trig_raw[0]) * LSBz

        # nst = ntr - uint8/60/60/24
        f.seek(143, 0)
        nst_offset_sec = int.from_bytes(f.read(1), "little", signed=False)
        nst = trigger_time - timedelta(seconds=nst_offset_sec)

        f.seek(145, 0)
        comment = f.read(30).decode(errors="ignore").strip("\x00").strip()

        # Trace block starts at 256
        f.seek(256, 0)

        # Each stored word is 24 bits; if aantalbits<24, value is stored in top/bottom?
        # MATLAB uses fread(..., bitN, skipbits=(24-aantalbits))
        # Here we implement:
        # - Read 24-bit signed values
        # - If aantalbits < 24: shift-right the "skipped" bits (common when padding LSBs)
        # This matches many 20-in-24 schemes, but if your device pads differently,
        # we can adjust after you validate with one file.
        raw_bytes = f.read(nsamp * nchan * 3)
        if len(raw_bytes) != nsamp * nchan * 3:
            raise ValueError("File ended unexpectedly while reading trace data.")

        vals = _read_int24_le_signed(raw_bytes)

        if aantalbits < 24:
            shift = 24 - aantalbits
            vals = (vals >> shift).astype(np.int32)

        trace = vals.reshape(nsamp, nchan)

        # Scale channels (assume first 3 correspond to x,y,z if available)
        scales = np.ones(nchan, dtype=float)
        if nchan >= 1:
            scales[0] = LSBx
        if nchan >= 2:
            scales[1] = LSBy
        if nchan >= 3:
            scales[2] = LSBz
        trace_scaled = trace.astype(np.float64) * scales

        # Build dataframe
        t_s = np.arange(nsamp, dtype=float) / float(fs)
        df = pd.DataFrame({"t_s": t_s})
        df["datetime"] = trigger_time + pd.to_timedelta(df["t_s"], unit="s")

        if nchan == 3:
            df["x"] = trace_scaled[:, 0]
            df["y"] = trace_scaled[:, 1]
            df["z"] = trace_scaled[:, 2]
        else:
            for i in range(nchan):
                df[f"ch{i+1}"] = trace_scaled[:, i]

        header = XMRHeader(
            nsamp=nsamp,
            filetype=filetype,
            fs=fs,
            nchan=nchan,
            dum=dum,
            ntr=trigger_time,
            dateTime_str=dateTime_str,
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

        return df, header


# ============================================================
# BMR reader (via TXT conversion)
# ============================================================


def convert_bmr_to_txt(
    bmr_path: Union[str, Path], converter_exe: Union[str, Path]
) -> Path:
    """
    Run: <converter_exe> <folder> txt
    (same idea as BMR_naar_txt.m)

    Returns the most likely .txt output path.
    """
    bmr_path = Path(bmr_path)
    converter_exe = Path(converter_exe)

    if not bmr_path.exists():
        raise FileNotFoundError(bmr_path)
    if not converter_exe.exists():
        raise FileNotFoundError(converter_exe)

    folder = bmr_path.parent
    cmd = [str(converter_exe), str(folder), "txt"]

    print("[convert]", " ".join(cmd))
    cp = subprocess.run(cmd, capture_output=True, text=True)

    if cp.returncode != 0:
        raise RuntimeError(
            "Conversion failed.\n"
            f"Return code: {cp.returncode}\n"
            f"STDOUT:\n{cp.stdout}\n"
            f"STDERR:\n{cp.stderr}\n"
        )

    candidate = bmr_path.with_suffix(".txt")
    if candidate.exists():
        return candidate

    # fallback: newest txt in folder
    txts = sorted(folder.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not txts:
        raise FileNotFoundError(
            "Converter returned success but no .txt files found in folder."
        )
    return txts[0]


def read_bmr_txt(txt_path: Union[str, Path]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Parse MR3000 ASCII file created from BMR conversion.
    Mirrors readBMR.m behavior: reads StartDate/StartTime and converts seconds to datetime.
    """
    txt_path = Path(txt_path)
    header: Dict[str, str] = {}

    # parse all "# key=value" lines
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = re.match(r"^\#\s*([^=]+)\s*=\s*(.*)\s*$", line.strip())
            if m:
                header[m.group(1).strip()] = m.group(2).strip()

    # numeric block: 4 columns => t, x, y, z (as in readBMR.m)
    df = pd.read_csv(
        txt_path,
        comment="#",
        sep=r"\s+",
        header=None,
        engine="python",
    )
    if df.shape[1] < 4:
        raise ValueError(
            f"Expected at least 4 numeric columns in {txt_path}, got {df.shape[1]}"
        )

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.rename(columns={0: "t_s", 1: "x", 2: "y", 3: "z"})

    # Build datetime like readBMR.m:
    # tstart = filedate + filetime ; t = t/(3600*24) + tstart
    # We'll parse StartDate + StartTime into python datetime.
    start_date = header.get("StartDate")
    start_time = header.get("StartTime")

    if start_date and start_time:
        # StartDate in their MATLAB was datenum(tline(13:end)) so often YYYY-MM-DD or similar
        # We'll try a few formats:
        parsed_date = None
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%Y.%m.%d"):
            try:
                parsed_date = datetime.strptime(start_date.strip(), fmt).date()
                break
            except ValueError:
                continue

        parsed_time = None
        for tfmt in ("%H:%M:%S", "%H:%M"):
            try:
                parsed_time = datetime.strptime(start_time.strip(), tfmt).time()
                break
            except ValueError:
                continue

        if parsed_date and parsed_time:
            t0 = datetime.combine(parsed_date, parsed_time)
            df["datetime"] = t0 + pd.to_timedelta(df["t_s"], unit="s")

    return df, header


def read_bmr(
    bmr_path: Union[str, Path], converter_exe: Union[str, Path]
) -> Tuple[pd.DataFrame, Dict[str, str], Path]:
    """
    Convert BMR -> TXT, parse TXT.
    """
    txt_path = convert_bmr_to_txt(bmr_path, converter_exe=converter_exe)
    df, header = read_bmr_txt(txt_path)
    return df, header, txt_path


# ============================================================
# Unified entry point
# ============================================================


def read_mr3000(
    path: Union[str, Path],
    converter_exe: Optional[Union[str, Path]] = None,
    aantalbits_xmr: int = 24,
):
    """
    Read .XMR or .BMR given a single path.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".xmr":
        return read_xmr(path, aantalbits=aantalbits_xmr)

    if ext == ".bmr":
        if converter_exe is None:
            raise ValueError("converter_exe is required to read BMR files")
        return read_bmr(path, converter_exe=converter_exe)

    if ext == ".txt":
        # assume it's a BMR-style export
        df, header = read_bmr_txt(path)
        return df, header

    raise ValueError(f"Unsupported extension: {ext}")


import matplotlib.pyplot as plt


def plot_xyz(df, title=None, use_datetime=False):
    """
    Plot 3-channel vibration data in 3 stacked subplots.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain t_s or datetime, and x/y/z (or ch1/ch2/ch3).
    title : str, optional
        Figure title.
    use_datetime : bool
        If True and df['datetime'] exists, use datetime on X-axis.
    """

    # --- choose time axis ---
    if use_datetime and "datetime" in df.columns:
        x = df["datetime"]
        xlabel = "Time"
    else:
        x = df["t_s"]
        xlabel = "Time [s]"

    # --- choose channels ---
    if all(c in df.columns for c in ["x", "y", "z"]):
        channels = ["x", "y", "z"]
    elif all(c in df.columns for c in ["ch1", "ch2", "ch3"]):
        channels = ["ch1", "ch2", "ch3"]
    else:
        raise ValueError("DataFrame must contain x/y/z or ch1/ch2/ch3")

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        figsize=(12, 8),
    )

    for ax, ch in zip(axes, channels):
        ax.plot(x, df[ch], linewidth=0.8)
        ax.set_ylabel(ch)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(xlabel)

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    # ---------------- EDIT THESE PATHS ----------------
    # Option A: Read an XMR directly (NO converter needed)
    xmr_file = r"C:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\25002003.XMR"

    # Option B: Read a BMR (converter needed)
    bmr_file = r"C:\Users\camposmo\OneDrive - Stichting Deltares\Desktop\25122002.BMR"
    converter_exe = r"P:\11207352-stem\Measurement_Culemborg\mr3000-convert-arch32.exe"
    # --------------------------------------------------

    # ---- XMR example ----
    if Path(xmr_file).exists():
        df_xmr, hdr_xmr = read_xmr(xmr_file, aantalbits=24)  # try 20 if needed
        print("\nXMR header:", hdr_xmr)
        print(df_xmr.head())
        df_xmr.to_csv("xmr_out.csv", index=False)
        print("Wrote xmr_out.csv")

    # ---- BMR example ----
    if Path(bmr_file).exists():
        df_bmr, hdr_bmr, txt_used = read_bmr(bmr_file, converter_exe=converter_exe)
        print("\nBMR txt used:", txt_used)
        print("BMR header keys:", list(hdr_bmr.keys()))
        print(df_bmr.head())
        df_bmr.to_csv("bmr_out.csv", index=False)
        print("Wrote bmr_out.csv")

    df_xmr, hdr_xmr = read_xmr(xmr_file, aantalbits=24)

    plot_xyz(
        df_xmr,
        title=f"XMR vibration – fs={hdr_xmr.fs} Hz",
        use_datetime=True,  # switch to False if you prefer seconds
        )

    df_bmr, hdr_bmr, _ = read_bmr(bmr_file, converter_exe=converter_exe)

    plot_xyz(
        df_bmr,
        title="BMR vibration",
        use_datetime=True,
    )
