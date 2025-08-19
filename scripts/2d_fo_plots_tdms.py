import os
import h5py
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from scipy import signal
from scipy.signal import windows
from scipy.integrate import cumulative_trapezoid
import numpy as np

# from scripts.playboxdascore import integrate_strain_rate_to_strain
from src.utils.TDMS_Read import TdmsReader
from SignalProcessingTools.time_signal import TimeSignalProcessing, Windows, FilterDesign


# --------------------- simple plotting helpers ---------------------
def save_heatmap(data2d, t, dx, out_path, title, cbar_label):
    eps = 1e-12
    z = np.log10(np.abs(data2d) + eps)  # good default for visibility
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


def from_opticalphase_to_strain(raw_data: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Take the raw OptaSense data and convert it to units of strain.
    """
    # Convert into units of radians
    raw_data = raw_data * (2 * np.pi / 2 ** 16)
    # Convert into units of strain
    n = metadata["fibre_refractive_index"]
    L = metadata["gauge_length"]
    data = raw_data * ((1550.12e-9) / (0.78 * 4 * np.pi * n * L))
    return data


def remove_mean(data: np.ndarray) -> np.ndarray:
    """
    Remove the mean from the data along the time axis (axis=0).
    """
    np.mean(data)
    print('The mean of the data is:', np.mean(data))
    return data - np.mean(data, axis=0)


def from_strain_rate_to_strain(rate_2d: np.ndarray, fs: float) -> np.ndarray:
    """
    Convert strain-rate (ε̇) to strain (ε) by time integration, channel-wise.

    Parameters
    ----------
    rate_2d : np.ndarray
        Array of shape (n_time, n_channels) with strain-rate.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Strain array of shape (n_time, n_channels), dimensionless.
    """
    rate_2d = np.asarray(rate_2d, dtype=np.float64)
    dt = 1.0 / fs
    # 'initial=0.0' makes the output length match input (n_time)
    strain = cumulative_trapezoid(rate_2d, dx=dt, axis=0, initial=0.0)
    return strain


def apply_scaling(strain: np.ndarray, fs: float, nm_per_m_per_sec: float = 11600.0) -> np.ndarray:
    """    What this function does:
      - Scales the integrated strain by (nm_per_m_per_sec / fs) to reproduce the
        “11.6 nm/m at 1 kHz” convention, then converts nanostrain → SI strain (×1e-9).
      - Default nm_per_m_per_sec = 11600.0 matches the 11.6 nm/m @ 1 kHz convention.

    Args:
        strain: 2D array (n_time, n_channels), AFTER time-integration.
        fs:     Sampling frequency [Hz].
        nm_per_m_per_sec: Legacy constant in nanostrain per second (default 11600.0).

    Returns:
        Scaled strain in SI units (dimensionless).

    Notes:
        - Set nm_per_m_per_sec=0.0 to bypass the legacy scaling while keeping the same call site.
        - The velocity step (strain × wave speed) and PSD are downstream, not part of this function."""
    return strain * (nm_per_m_per_sec / fs) * 1e-9  # -> strain


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
        # Keep original length n_t (simple & robust)
        if sig_f.shape[0] >= n_t:
            out[:, ch] = sig_f[:n_t]
        else:
            # pad with last value if TSP returned fewer samples (rare)
            out[:, ch] = np.pad(sig_f, (0, n_t - sig_f.shape[0]), mode="edge")

    return out


# ---- Paths ----
fo_tdms_file = r"D:\culemborg\culemborg_2020\20112020\subset\iDAS_continous_measurements_30s_UTC_20201120_095157.969.tdms"
out_dir = Path(r"D:\fo_test")
out_dir.mkdir(parents=True, exist_ok=True)
stem = Path(fo_tdms_file).stem  # This takes the name -> 'sensor_2024-08-29T080549Z'

# Create TDMS instance
tdms = TdmsReader(fo_tdms_file)
tdms_properties = tdms.get_properties()
raw_fo = tdms.get_data()
n_t, n_ch = raw_fo.shape

# Metadata for operations
metadata = {
    "fs": tdms_properties['SamplingFrequency[Hz]'],  # Hz
    "dx": tdms_properties['SpatialResolution[m]'],  # meters
    "gauge_length": tdms_properties['GaugeLength'],
    "fibre_refractive_index": tdms_properties['FibreIndex'],
}

# Common axes / settings
time_array = np.arange(n_t) / metadata["fs"]
x_end = (n_ch - 1) * metadata["dx"]
channel_idx = 3200
pos_m = channel_idx * metadata["dx"]
eps = 1e-12
Fpass = [0.1, 100]  # Hz
N = 5  # Filter order
type_filter = "bandpass"
design = "elliptic"

# 1. Remove mean (per channel) then convert to strain
demeaned_fo_data = remove_mean(raw_fo.astype(np.float64))

# 2. Tapper with the tukey window
w = windows.tukey(M=n_t, alpha=0.04)  # shape (n_t,)
tapered_fo_data = demeaned_fo_data * w[:, None]  # broadcast over channels

# 3. Apply the bandpass filter (per channel) to the demeaned data
filtered_fo_data = apply_TSP_filter_2d(data_2d=tapered_fo_data, t=time_array, fs=metadata["fs"], Fpass=Fpass, N=N,
                                       window_size=1024)

# 4) integrate strain-rate -> strain
strain_fo_data = from_strain_rate_to_strain(filtered_fo_data, fs=metadata["fs"])

# 5) Scaling to nano-strain from Edwin (not really clear what this does)
strain_fo_data = apply_scaling(strain_fo_data, fs=metadata["fs"])

# NOW LETS FOCUS ON THE PSD ############################################################################################
# get the timesignal for channel 2000
trace_ch_2000 = strain_fo_data[:, channel_idx]
# Create the TSP object for channel 2000 and apply the filter and the psd
tsp_ch_2000 = TimeSignalProcessing(time=time_array, signal=trace_ch_2000, Fs=metadata["fs"],
                                   window_size=1024, window=Windows.HAMMING)
tsp_ch_2000.filter(Fpass=Fpass, N=N, type_filter=type_filter)
# Now compute the PSD
tsp_ch_2000.psd()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

freq = tsp_ch_2000.frequency_Pxx
psd = tsp_ch_2000.Pxx
eps = 1e-20  # avoid log(0)

# Top: log-Y (semilogy)
ax1.semilogy(freq, np.maximum(psd, eps))
ax1.set_ylabel("PSD (1/Hz)")
ax1.grid(True, which="both", ls="--", alpha=0.4)
ax1.set_title("Channel 2000 PSD (log vs. linear)")

# Bottom: linear Y
ax2.plot(freq, psd)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("PSD (1/Hz)")
ax2.grid(True, ls="--", alpha=0.4)

# X formatting shared
ax2.set_xlim(0, 100)
ax2.set_xticks(np.arange(0, 101, 5))

plt.tight_layout()
out_path = out_dir / f"{stem}__ch{channel_idx}_psd.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")

# --------------------- pick what to plot (flip True/False) ---------------------
PLOTS = {
    "raw2d": True,
    "raw_ch": True,
    "strain_2d": True,
    "strain_ch": True,
    "filtered_2d": True,
    "filtered_ch": True,
}

# --------------------- make the plots you asked for ---------------------
if PLOTS["raw2d"]:
    save_heatmap(
        data2d=raw_fo, t=time_array, dx=metadata["dx"],
        out_path=out_dir / f"{stem}__2d_raw.png",
        title="RAW: time vs position", cbar_label="log10(|raw| + eps)"
    )

if PLOTS["raw_ch"]:
    save_channel(
        ts2d=raw_fo, t=time_array, ch_idx=channel_idx, dx=metadata["dx"],
        out_path=out_dir / f"{stem}__ch{channel_idx}_raw.png",
        title="RAW Channel", ylabel="Optical phase magnitude"
    )

if PLOTS["strain_2d"]:
    save_heatmap(
        data2d=strain_fo_data, t=time_array, dx=metadata["dx"],
        out_path=out_dir / f"{stem}__2d_strain.png",
        title="STRAIN: time vs position", cbar_label="log10(|strain| + eps)"
    )

if PLOTS["strain_ch"]:
    save_channel(
        ts2d=strain_fo_data, t=time_array, ch_idx=channel_idx, dx=metadata["dx"],
        out_path=out_dir / f"{stem}__ch{channel_idx}_strain.png",
        title="STRAIN Channel", ylabel="Strain (ε)"
    )

if PLOTS["filtered_2d"]:
    save_heatmap(
        data2d=filtered_fo_data, t=time_array, dx=metadata["dx"],
        out_path=out_dir / f"{stem}__2d_filtered.png",
        title="FILTERED STRAIN: time vs position", cbar_label="log10(|filtered| + eps)"
    )

if PLOTS["filtered_ch"]:
    save_channel(
        ts2d=filtered_fo_data, t=time_array, ch_idx=channel_idx, dx=metadata["dx"],
        out_path=out_dir / f"{stem}__ch{channel_idx}_filtered.png",
        title="FILTERED STRAIN Channel", ylabel="Filtered Strain (ε)"
    )

print("Saved plots to:", out_dir)
