# Batch DAS processing (.h5 OptaSense / .tdms Silixa)
# - mean removal
# - optional Tukey taper
# - convert to strain (H5: optical-phase->strain; TDMS: strain-rate->integrate->strain, scaling)
# - band-pass filter (vectorized SOS, zero-phase)
# - plots: 2D heatmaps + single-channel + PSD (Welch) with semilogy+linear

import time
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import windows
from scipy.integrate import cumulative_trapezoid

from src.utils.TDMS_Read import TdmsReader

# =============== PLOTTING ===============
PLOT_DPI = 150  # lower = faster disk writes


def save_heatmap(data2d, t, dx, out_path, title, cbar_label, stride_t=4, stride_x=2):
    """Fast heatmap: downsample only for plotting."""
    eps = 1e-12
    z = np.abs(np.asarray(data2d[::stride_t, ::stride_x], dtype=np.float32))
    z = np.log10(z + eps)
    x_end = (data2d.shape[1] - 1) * dx
    # we keep full end time on the axis for clarity
    plt.figure(figsize=(12, 6))
    plt.imshow(z, aspect='auto', origin='lower', cmap='jet',
               extent=[0, x_end, 0, t[-1]])
    plt.colorbar(label=cbar_label)
    plt.xlabel("Position along fibre (m)")
    plt.ylabel("Time (s)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def save_channel(ts2d, t, ch_idx, dx, out_path, title, ylabel):
    plt.figure(figsize=(12, 5))
    plt.plot(t, ts2d[:, ch_idx])
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(f"{title} — Channel {ch_idx} (≈ {ch_idx * dx:.2f} m)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def save_channel_psd_doubleplot(trace, fs, out_path, title, xlim=(0, 100)):
    """PSD (Welch) double-plot: semilogy (top) + linear (bottom)."""
    f, Pxx = signal.welch(trace, fs=fs, nperseg=1024, nfft=1024,
                          window="hamming", scaling="density", detrend="linear")
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
    plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


# =============== BASIC TRANSFORMS ===============

def remove_mean(data: np.ndarray) -> np.ndarray:
    """Remove per-channel mean along time (axis=0)."""
    return data - data.mean(axis=0, keepdims=True)


def from_opticalphase_to_strain(raw_data: np.ndarray, metadata: dict) -> np.ndarray:
    """OptaSense optical phase counts -> strain (ε)."""
    raw_data = np.asarray(raw_data, dtype=np.float32)
    raw_rad = raw_data * (2 * np.pi / 2 ** 16)  # counts -> radians
    n = float(metadata["fibre_refractive_index"])
    L = float(metadata["gauge_length"])
    return (raw_rad * ((1550.12e-9) / (0.78 * 4 * np.pi * n * L))).astype(np.float32)


def from_strain_rate_to_strain(rate_2d: np.ndarray, fs: float) -> np.ndarray:
    """Channel-wise time integration: strain-rate (ε̇) → strain (ε)."""
    rate_2d = np.asarray(rate_2d, dtype=np.float32)
    dt = 1.0 / fs
    return cumulative_trapezoid(rate_2d, dx=dt, axis=0, initial=0.0).astype(np.float32)


def apply_scaling(strain: np.ndarray, fs: float, nm_per_m_per_sec: float = 11600.0) -> np.ndarray:
    """
    Legacy scaling (Edwin):
      After integrating (ε̇ → ε), scale by (11,600 nm/m per second)/fs, then nm→m (×1e-9).
      Reproduces “11.6 nm/m @ 1 kHz”.
    """
    return (strain * (nm_per_m_per_sec / fs) * 1e-9).astype(np.float32)


# =============== FAST VECTORIZED FILTERING ===============

def design_sos(Fpass, N, fs, design="butter", rp=0.01, rs=60.0, btype="bandpass"):
    if design == "butter":
        sos = signal.butter(N, Fpass, btype=btype, fs=fs, output="sos")
    elif design == "elliptic":
        sos = signal.ellip(N, rp, rs, Fpass, btype=btype, fs=fs, output="sos")
    elif design == "cheby1":
        sos = signal.cheby1(N, rp, Fpass, btype=btype, fs=fs, output="sos")
    else:
        raise ValueError("design must be 'butter', 'elliptic', or 'cheby1'")
    return sos


def apply_SOS_filter_2d(data_2d, sos):
    """Zero-phase filtering along time (axis=0)."""
    x = np.asarray(data_2d, dtype=np.float32, order="C")
    return signal.sosfiltfilt(sos, x, axis=0).astype(np.float32)


# =============== FILE LOADERS ===============

def load_h5(path):
    """Load OptaSense .h5 -> (raw_data, meta). raw_data = optical-phase counts."""
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
    """Load Silixa iDAS .tdms -> (raw_data, meta). raw_data = strain-rate."""
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


# =============== USER CONFIG ===============

INPUT_DIR = Path(r"D:\culemborg\culemborg_2020\20112020\subset")  # folder with .h5 / .tdms
OUT_DIR = Path(r"D:\fo_test\batch_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
PLOT_STRIDE_T = 4  # << increase for much faster heatmaps
PLOT_STRIDE_X = 2  # << increase for much faster heatmaps

# Processing params
channel_idx = 2000
Fpass = [1, 100]  # Hz
N = 5  # IIR order
design = "butter"  # 'butter' | 'elliptic' | 'cheby1'
taper_alpha = 0.04  # Tukey alpha

# =============== SIMPLE LOOP OVER FILES ===============

files = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in (".h5", ".tdms")]

for FILE in files:
    stem = FILE.stem
    suffix = FILE.suffix.lower()

    # ---- Load
    if suffix == ".h5":
        raw_fo, meta = load_h5(FILE)
        source = "h5_opta"  # optical phase counts
    else:
        raw_fo, meta = load_tdms(FILE)
        source = "tdms_idas"  # strain-rate

    n_t, n_ch = raw_fo.shape
    ch = min(channel_idx, n_ch - 1)
    fs = meta["fs"]
    dx = meta["dx"]
    t = np.arange(n_t, dtype=np.float32) / fs

    print(f"\nProcessing {FILE.name}  (samples={n_t}, channels={n_ch}, fs={fs:g} Hz)")

    # 1) mean removal
    demeaned = remove_mean(raw_fo.astype(np.float32))
    # 2) taper
    w = windows.tukey(M=n_t, alpha=taper_alpha).astype(np.float32)
    tapered = demeaned * w[:, None]

    # 3) convert to strain
    if source == "h5_opta":
        strain = from_opticalphase_to_strain(tapered, meta)
    elif source == "tdms_idas":
        strain = from_strain_rate_to_strain(tapered, fs=fs)
        strain = apply_scaling(strain, fs=fs)
    else:
        raise ValueError(f"Unknown source: {source}")

    # 4) band-pass on strain (vectorized)
    sos = design_sos(Fpass=Fpass, N=N, fs=fs, design=design, btype="bandpass")
    filtered = apply_SOS_filter_2d(strain, sos)

    # ---- PLOTS (downsampled heatmaps)
    if PLOTS["raw2d"]:
        save_heatmap(raw_fo, t, dx, OUT_DIR / f"{stem}__2d_raw.png",
                     "RAW: time vs position", "log10(|raw| + eps)",
                     stride_t=PLOT_STRIDE_T, stride_x=PLOT_STRIDE_X)
    if PLOTS["raw_ch"]:
        save_channel(raw_fo, t, ch, dx, OUT_DIR / f"{stem}__ch{ch}_raw.png",
                     "RAW Channel", "Raw magnitude")

    if PLOTS["strain_2d"]:
        save_heatmap(strain, t, dx, OUT_DIR / f"{stem}__2d_strain.png",
                     "STRAIN: time vs position", "log10(|strain| + eps)",
                     stride_t=PLOT_STRIDE_T, stride_x=PLOT_STRIDE_X)
    if PLOTS["strain_ch"]:
        save_channel(strain, t, ch, dx, OUT_DIR / f"{stem}__ch{ch}_strain.png",
                     "STRAIN Channel", "Strain (ε)")

    if PLOTS["filtered_2d"]:
        save_heatmap(filtered, t, dx, OUT_DIR / f"{stem}__2d_filtered.png",
                     "FILTERED: time vs position", "log10(|filtered| + eps)",
                     stride_t=PLOT_STRIDE_T, stride_x=PLOT_STRIDE_X)
    if PLOTS["filtered_ch"]:
        save_channel(filtered, t, ch, dx, OUT_DIR / f"{stem}__ch{ch}_filtered.png",
                     "FILTERED Channel", "Filtered (ε)")

    if PLOTS["psd_ch_double"]:
        save_channel_psd_doubleplot(
            time_arr=t,
            trace=filtered[:, ch],  # already filtered -> no extra filter here
            fs=fs,
            out_path=OUT_DIR / f"{stem}__ch{ch}_psd.png",
            title=f"{stem} — Channel {ch} PSD (log vs. linear)",
            xlim=(0, 100)
        )
    t_plot = time.perf_counter()

print("\nAll files processed.")
print("Outputs:", OUT_DIR)
