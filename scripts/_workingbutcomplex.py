# Code to be able to observe all the process. From the raw 2D complete file, to the filtered 2D file.
# and the PSD of a single channel both in semilogy and linear scale. This works for both .h5 and .tdms files.

import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import windows
from scipy.integrate import cumulative_trapezoid

# If your TdmsReader lives elsewhere, adjust this import.
from src.utils.TDMS_Read import TdmsReader

from SignalProcessingTools.time_signal import TimeSignalProcessing, Windows, FilterDesign


# --------------------- simple plotting helpers ---------------------
def save_heatmap(data2d, t, dx, out_path, title, cbar_label):
    eps = 1e-12
    z = np.log10(np.abs(data2d) + eps)
    x_end = (data2d.shape[1] - 1) * dx

    plt.figure(figsize=(12, 6))
    plt.imshow(z, aspect='auto', origin='lower', cmap='jet',
               extent=[0, x_end, 0, t[-1]])
    plt.colorbar(label=cbar_label)
    plt.xlabel("Position along fibre (m)")
    plt.ylabel("Time (s)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_channel(ts2d, t, ch_idx, dx, out_path, title, ylabel):
    plt.figure(figsize=(12, 5))
    plt.plot(t, ts2d[:, ch_idx])
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(f"{title} — Channel {ch_idx} (≈ {ch_idx * dx:.2f} m)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------- basic transforms ---------------------
def remove_mean(data: np.ndarray) -> np.ndarray:
    """Remove per-channel mean along time (axis=0)."""
    return data - np.mean(data, axis=0)


def from_opticalphase_to_strain(raw_data: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Convert OptaSense optical phase counts to strain (ε).
    Uses wavelength=1550.12 nm and factor 0.78 consistent with your previous work.
    """
    # counts -> radians
    raw_rad = raw_data * (2 * np.pi / 2 ** 16)
    # radians -> strain
    n = float(metadata["fibre_refractive_index"])
    L = float(metadata["gauge_length"])
    return raw_rad * ((1550.12e-9) / (0.78 * 4 * np.pi * n * L))


def from_strain_rate_to_strain(rate_2d: np.ndarray, fs: float) -> np.ndarray:
    """Channel-wise time integration: strain-rate (ε̇) → strain (ε)."""
    rate_2d = np.asarray(rate_2d, dtype=np.float64)
    dt = 1.0 / fs
    return cumulative_trapezoid(rate_2d, dx=dt, axis=0, initial=0.0)


def apply_scaling(strain: np.ndarray, fs: float, nm_per_m_per_sec: float = 11600.0) -> np.ndarray:
    """
    Scaling used in Edwin's workflow:

      • After integrating (ε̇ → ε), scale by (11,600 nm/m per second)/fs, then convert nm→m (×1e-9),
        which reproduces the “11.6 nm/m @ 1 kHz” convention.
      • Set nm_per_m_per_sec=0.0 to bypass while keeping the same call site.

    Returns strain in SI (dimensionless).
    """
    return strain * (nm_per_m_per_sec / fs) * 1e-9


# --------------------- filtering (Bruno's TSP per-channel) ---------------------
def apply_TSP_filter_2d(
        data_2d: np.ndarray,
        t: np.ndarray,
        fs: float,
        Fpass,
        N: int,
        type_filter: str = "bandpass",
        filter_design: FilterDesign = FilterDesign.BUTTERWORTH,
        window_size: int = 1024,
        window: Windows = Windows.HAMMING,
        rp: float = 0.01,
        rs: float = 60.0,
) -> np.ndarray:
    """
    Apply Bruno's TimeSignalProcessing.filter per channel and return the filtered 2D array.
    - data_2d shape: (n_time, n_channels)
    - Filtering runs along time for each channel independently.
    """
    x = np.asarray(data_2d, dtype=np.float64)
    n_t, n_ch = x.shape
    if len(t) != n_t:
        raise ValueError("t must have length equal to the number of time samples")

    out = np.empty_like(x)
    for ch in range(n_ch):
        tsp = TimeSignalProcessing(time=t, signal=x[:, ch], Fs=fs, window_size=window_size, window=window)
        tsp.filter(Fpass=Fpass, N=N, filter_design=filter_design, type_filter=type_filter, rp=rp, rs=rs)
        sig_f = np.asarray(tsp.signal)
        # Keep original length
        if sig_f.shape[0] >= n_t:
            out[:, ch] = sig_f[:n_t]
        else:
            out[:, ch] = np.pad(sig_f, (0, n_t - sig_f.shape[0]), mode="edge")
    return out


# --------------------- PSD for a single channel via TSP ---------------------
def save_channel_psd_doubleplot(time, trace, fs, out_path, title, Fpass, xlim=(0, 100)):
    """
    Filter (again) then PSD via TSP to match your established workflow, with a
    2-row plot: semilogy (top) + linear (bottom).
    """
    tsp = TimeSignalProcessing(time=time, signal=trace, Fs=fs, window_size=1024, window=Windows.HAMMING)
    # You can change these to match your main Fpass/N if desired:
    tsp.filter(Fpass=Fpass, N=5, type_filter="bandpass")
    tsp.psd()

    f = tsp.frequency_Pxx
    Pxx = tsp.Pxx
    eps = 1e-20

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.semilogy(f, np.maximum(Pxx, eps))
    ax1.set_ylabel("PSD (1/Hz)")
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.set_title(title)

    ax2.plot(f, Pxx)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("PSD (1/Hz)")
    ax2.grid(True, ls="--", alpha=0.4)
    if xlim is not None:
        ax2.set_xlim(*xlim)
        ax2.set_xticks(np.arange(xlim[0], xlim[1] + 1, 5))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------- file loaders ---------------------
def load_h5(path):
    """
    Load OptaSense .h5. Returns (raw_data, metadata_dict).
      raw_data: optical-phase counts (int)
      metadata: fs, dx, gauge_length, fibre_refractive_index
    """
    with h5py.File(path, "r") as f:
        ds = f["Acquisition"]["Raw[0]"]["RawData"]
        raw = ds[:]
        meta = {
            "fs": float(f["Acquisition"]["Raw[0]"].attrs["OutputDataRate"]),
            "dx": float(f["Acquisition"].attrs["SpatialSamplingInterval"]),
            "gauge_length": float(f["Acquisition"].attrs["GaugeLength"]),
            "fibre_refractive_index": float(f["Acquisition"]["Custom"].attrs["Fibre Refractive Index"]),
        }
    return raw, meta


def load_tdms(path):
    """
    Load Silixa iDAS .tdms via TdmsReader. Returns (raw_data, metadata_dict).
      raw_data: strain-rate (float/int)
      metadata: fs, dx, gauge_length, fibre_refractive_index (if available)
    """
    tdms = TdmsReader(path)
    props = tdms.get_properties()
    raw = tdms.get_data()
    meta = {
        "fs": float(props["SamplingFrequency[Hz]"]),
        "dx": float(props["SpatialResolution[m]"]),
        "gauge_length": float(props.get("GaugeLength", np.nan)),
        "fibre_refractive_index": float(props.get("FibreIndex", np.nan)),
    }
    return raw, meta


# ===================== MAIN CONFIG =====================
# Point this to EITHER a .h5 or a .tdms file.
# FILE = r"D:\culemborg\culemborg_2020\20112020\subset\iDAS_continous_measurements_30s_UTC_20201120_095157.969.tdms"
FILE = r"E:\recording_2024-08-29T08_01_16Z_5kHzping_1kHzlog_1mCS_10mGL_3000channels\sensor_2024-08-29T080549Z.h5"

out_dir = Path(r"D:\fo_test\unified")
out_dir.mkdir(parents=True, exist_ok=True)
stem = Path(FILE).stem

# Plot toggles
PLOTS = {
    "raw2d": True,
    "raw_ch": True,
    "strain_2d": True,
    "strain_ch": True,
    "filtered_2d": True,
    "filtered_ch": True,
    "psd_ch_double": True,
}

# Processing params
channel_idx = 2000  # change as needed
Fpass = [1, 100]  # Hz
N = 5  # order
apply_taper = True
taper_alpha = 0.04  # Tukey alpha

# ===================== LOAD =====================
suffix = Path(FILE).suffix.lower()
if suffix == ".h5":
    raw_fo, meta = load_h5(FILE)
    source = "h5_opta"  # optical phase counts
elif suffix == ".tdms":
    raw_fo, meta = load_tdms(FILE)
    source = "tdms_idas"  # strain-rate
else:
    raise ValueError("Unsupported file type. Use .h5 or .tdms")

n_t, n_ch = raw_fo.shape
t = np.arange(n_t) / meta["fs"]
dx = meta["dx"]
pos_m = channel_idx * dx

# ===================== WORKFLOW =====================
# 1) mean removal
demeaned = remove_mean(raw_fo.astype(np.float64))

# 2) taper (same for both)
w = windows.tukey(M=n_t, alpha=0.04)
tapered = demeaned * w[:, None]

# 3) convert to strain domain (unified)
if source == "h5_opta":
    # optical phase -> strain (constant scaling)
    strain = from_opticalphase_to_strain(tapered, meta)
elif source == "tdms_idas":
    # strain-rate -> integrate -> strain
    strain = from_strain_rate_to_strain(tapered, fs=meta["fs"])
    strain = apply_scaling(strain, fs=meta["fs"])  # or comment out
else:
    raise RuntimeError("Unknown source type")

# 4) band-pass on strain (same for both)
filtered = apply_TSP_filter_2d(
    data_2d=strain, t=t, fs=meta["fs"],
    Fpass=Fpass, N=5,
    type_filter="bandpass",
    filter_design=FilterDesign.BUTTERWORTH,
    window_size=1024,
    window=Windows.HAMMING
)

# ===================== PLOTS =====================
# RAW plots (what you loaded)
if PLOTS["raw2d"]:
    save_heatmap(raw_fo, t, dx, out_dir / f"{stem}__2d_raw.png",
                 "RAW: time vs position", "log10(|raw| + eps)")
if PLOTS["raw_ch"]:
    save_channel(raw_fo, t, channel_idx, dx,
                 out_dir / f"{stem}__ch{channel_idx}_raw.png",
                 "RAW Channel", "Raw magnitude")

# Strain 2D / channel
if PLOTS["strain_2d"]:
    save_heatmap(
        strain, t, dx, out_dir / f"{stem}__2d_strain.png",
        "STRAIN: time vs position", "log10(|strain| + eps)"
    )
if PLOTS["strain_ch"]:
    save_channel(
        strain, t, channel_idx, dx, out_dir / f"{stem}__ch{channel_idx}_strain.png",
        "STRAIN Channel", "Strain (ε)"
    )

# Filtered 2D / channel
if PLOTS["filtered_2d"]:
    save_heatmap(
        filtered, t, dx, out_dir / f"{stem}__2d_filtered.png",
        "FILTERED: time vs position", "log10(|filtered| + eps)"
    )
if PLOTS["filtered_ch"]:
    save_channel(
        filtered, t, channel_idx, dx, out_dir / f"{stem}__ch{channel_idx}_filtered.png",
        "FILTERED Channel", "Filtered (ε)"
    )

# PSD of a single channel (2-row figure)
if PLOTS["psd_ch_double"]:
    if source == "h5_opta":
        psd_trace = strain[:, channel_idx]  # then TSP.filter + PSD inside helper
    else:
        psd_trace = strain[:, channel_idx]
    save_channel_psd_doubleplot(
        time=t,
        trace=psd_trace,
        fs=meta["fs"],
        Fpass=Fpass,
        out_path=out_dir / f"{stem}__ch{channel_idx}_psd.png",
        title=f"Channel {channel_idx} PSD (log vs. linear)"
    )

print("Saved outputs to:", out_dir)
