import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- File paths ---
file_106 = Path(
    r"N:\Projects\11210000\11210064\B. Measurements and calculations\holten\10m GL\res-20240829_20240831-SPRA-ch_1194-dir_1_Fpass_2-100Hz\processed_data_event_20240830_172137.mat.pickle")
file_134 = Path(
    r"N:\Projects\11210000\11210064\B. Measurements and calculations\holten\10m GL\res-20240829_20240831-SPRA-ch_1194-dir_1_Fpass_2-100Hz\processed_data_event_20240829_152801.mat.pickle")

label_106 = "106 km/h"
label_134 = "135 km/h"


def last_psd(psd_field):
    """
    Return a 1D numpy array for the PSD of the *current* event.
    Handles both old (list-of-lists cumulative) and future (1D) formats.
    """
    arr = np.array(psd_field, dtype=float)
    if arr.ndim == 1:  # already a single spectrum
        return arr
    if arr.ndim == 2:  # cumulative: (Nevents_so_far, Nfreq)
        return arr[-1]
    # Fallback: flatten (shouldn't happen)
    return arr.reshape(-1)


def load_event_current_only(pickle_path):
    with open(pickle_path, "rb") as f:
        d = pickle.load(f)

    # Time → seconds since start (keeps raw order, no resampling)
    time_dt = np.array(d["time"])
    t0 = time_dt[0]
    tsec = np.array([(t - t0).total_seconds() for t in time_dt], float)

    # Traces as stored
    trace_x = np.array(d["trace_x"][0], float)  # wrapped as [list]
    trace_y = np.array(d["trace_y"][0], float)
    trace_z = np.array(d["trace_z"][0], float)
    trace_fo = np.array(d["trace_fo"], float)  # flat list

    # Frequency vector (1D)
    freq = np.array(d["freq"], float)

    # Get only the last PSD (current event)
    PSD_x = last_psd(d["PSD_x"])
    PSD_y = last_psd(d["PSD_y"])
    PSD_z = last_psd(d["PSD_z"])
    PSD_fo = last_psd(d["PSD_fo"])

    # Sanity: lengths match
    if freq.size != PSD_x.size:
        raise ValueError(f"freq length {freq.size} != PSD_x length {PSD_x.size} in {pickle_path.name}")

    return {
        "t": tsec,
        "x": trace_x, "y": trace_y, "z": trace_z, "fo": trace_fo,
        "f": freq,
        "PSD_x": PSD_x, "PSD_y": PSD_y, "PSD_z": PSD_z, "PSD_fo": PSD_fo,
    }


ev106 = load_event_current_only(file_106)
ev134 = load_event_current_only(file_134)

# --- Plot 4×2: left = time signals, right = *last* PSD only (no averaging, no interpolation) ---
fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex='col')

c106 = "#1f77b4"  # blue
c134 = "#d62728"  # red


def plot_row(r, trace_key, psd_key, ylab_trace, ylab_psd):
    # Left: time traces
    axes[r, 0].plot(ev106["t"], ev106[trace_key], lw=0.9, label=label_106, color=c106, alpha=0.9)
    axes[r, 0].plot(ev134["t"], ev134[trace_key], lw=0.9, label=label_134, color=c134, alpha=0.85)
    axes[r, 0].set_ylabel(ylab_trace)
    axes[r, 0].grid(True, alpha=0.3)
    if r == 0:
        axes[r, 0].set_title("Time-domain signals (t since event start)")
    if r == 3:
        axes[r, 0].set_xlabel("Time (s)")

    # Right: only the *last* PSD from each pickle
    axes[r, 1].plot(ev106["f"], ev106[psd_key], lw=1.0, label=label_106, color=c106, alpha=0.95)
    axes[r, 1].plot(ev134["f"], ev134[psd_key], lw=1.0, label=label_134, color=c134, alpha=0.95)
    axes[r, 1].set_ylabel(ylab_psd)
    axes[r, 1].grid(True, alpha=0.3)
    if r == 0:
        axes[r, 1].set_title("Power Spectral Density (current event only)")
    if r == 3:
        axes[r, 1].set_xlabel("Frequency (Hz)")
    axes[r, 1].set_xlim(0, 100)  # your bandpass region


plot_row(0, "x", "PSD_x", "Acc X (m/s²)", "PSD X (units²/Hz)")
plot_row(1, "y", "PSD_y", "Acc Y (m/s²)", "PSD Y (units²/Hz)")
plot_row(2, "z", "PSD_z", "Acc Z (m/s²)", "PSD Z (units²/Hz)")
plot_row(3, "fo", "PSD_fo", "FO strain", "FO PSD (units²/Hz)")

axes[0, 0].legend(loc="upper right");
axes[0, 1].legend(loc="upper right")
plt.tight_layout()
plt.show()
