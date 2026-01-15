# === Compare two records (from the same accelerometer file) on VERTICAL (V1,V5) and HORIZONTAL (V2,V6) ===
# Layout: 2 rows (Loc1, Loc5) × 2 columns (Vertical, Horizontal)
# - Figure 1: Time-domain traces (optional twin y-axes per subplot)
# - Figure 2: PSDs via TimeSignalProcessing (optional twin y-axes per subplot)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SignalProcessingTools.time_signal import TimeSignalProcessing, Windows

# -------------------- USER INPUTS --------------------
ACCEL_CSV_PATH = r"D:\culemborg\culemborg_2020\processed_Edwin\003_Data_Processing\csv_files\Culemborg_11112020\cul_024.csv"
TIME_SEGMENTS_CSV = r"D:\culemborg\culemborg_2020\processed_Edwin\003_Data_Processing\csv_files\Culemborg_11112020\Time_segments\cul_024_time_segments.csv"

REC_A = "Record3"  # pick any column that exists in the time-segments CSV
REC_B = "Record4"  # pick any column that exists in the time-segments CSV

FS = 1000.0  # Hz
PSD_FMAX = 100  # Hz x-limit for PSD plots
USE_TWINY = True  # True = separate y-scales per record (twin axes); False = same axis

# Column name in accel CSV that holds the sample index:
INDEX_COL = "Unnamed: 0"
# Channels
VERT_COLS = ["V1", "V5"]  # vertical
HORIZ_COLS = ["V2", "V6"]  # horizontal
ALL_COLS = VERT_COLS + HORIZ_COLS


# -------------------- HELPERS --------------------
def tsp_psd(time_vec, signal_1d, fs, window_size=1024, window=Windows.HAMMING):
    """Compute PSD using Bruno's TimeSignalProcessing.psd()."""
    tsp = TimeSignalProcessing(time=time_vec, signal=signal_1d,
                               Fs=fs, window_size=window_size, window=window)
    tsp.psd()
    return np.asarray(tsp.frequency_Pxx), np.asarray(tsp.Pxx)
    # tsp.fft(half_representation=True)
    # return np.asarray(tsp.frequency), np.asarray(tsp.amplitude)


def extract_selection(db_accel, idx_series, fs, index_col=INDEX_COL, cols=ALL_COLS):
    """
    Keep only rows whose 'index_col' is exactly in idx_series (order preserved),
    and return (t, {col: ndarray} for requested cols).
    """
    # numeric copies (drop non-numeric/NaN)
    acc_idx = pd.to_numeric(db_accel[index_col], errors='coerce').dropna().astype(int)
    idx = pd.to_numeric(idx_series, errors='coerce').dropna().astype(int)

    # make sure requested columns exist
    for c in cols:
        if c not in db_accel.columns:
            raise KeyError(f"Column '{c}' not found in accelerometer CSV.")

    # subset accel rows and ensure index column numeric
    cols_needed = [index_col] + cols
    acc_subset = db_accel.loc[acc_idx.index, cols_needed].copy()
    acc_subset[index_col] = acc_idx.values

    # preserve the order given by idx_series
    order_df = pd.DataFrame({index_col: idx.values, 'order': np.arange(len(idx))})
    sel = (order_df.merge(acc_subset, on=index_col, how='inner')
           .sort_values('order')
           .drop(columns='order'))

    if sel.empty:
        return None

    samples = sel[index_col].to_numpy()
    t = (samples - samples[0]) / fs  # start at 0 s

    out = {c: sel[c].to_numpy() for c in cols}
    return t, out


# -------------------- LOAD --------------------
db_accel = pd.read_csv(ACCEL_CSV_PATH)
db_times = pd.read_csv(TIME_SEGMENTS_CSV)

if REC_A not in db_times.columns or REC_B not in db_times.columns:
    raise KeyError(f"Record columns '{REC_A}' and/or '{REC_B}' not found in {TIME_SEGMENTS_CSV}.")

idx_A = db_times[REC_A]
idx_B = db_times[REC_B]

selA = extract_selection(db_accel, idx_A, FS, index_col=INDEX_COL, cols=ALL_COLS)
selB = extract_selection(db_accel, idx_B, FS, index_col=INDEX_COL, cols=ALL_COLS)

if selA is None or selB is None:
    raise ValueError("No matching rows for one/both records. Check the time-segments indices and accel index column.")

tA, sigA = selA  # dict with keys in ALL_COLS
tB, sigB = selB

# -------------------- PLOT 1: Time-domain (2x2 grid) --------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=False)
# Map subplots: rows = locations; columns = vertical / horizontal
layout = [
    # (title, left_col_series_name, right_col_series_name, row_idx)
    ("Loc1", "V1", "V2", 0),
    ("Loc5", "V5", "V6", 1),
]

for title, vert_col, horiz_col, r in layout:
    # ----- Left column: Vertical -----
    ax = axes[r, 0]
    yA = sigA[vert_col];
    yB = sigB[vert_col]
    if USE_TWINY:
        l1, = ax.plot(tA, yA, lw=1.2, label=REC_A, color="tab:blue")
        ax.set_ylabel(f"{REC_A} amp")
        ax.set_title(f"{title} vertical ({vert_col})")
        ax.grid(True)
        ax2 = ax.twinx()
        l2, = ax2.plot(tB, yB, lw=1.2, label=REC_B, color="tab:orange", alpha=0.9)
        ax2.set_ylabel(f"{REC_B} amp", color="tab:orange")
        ax2.tick_params(axis='y', labelcolor="tab:orange")
        ax.legend([l1, l2], [REC_A, REC_B], loc="upper right")
    else:
        ax.plot(tA, yA, lw=1.2, label=REC_A, color="tab:blue")
        ax.plot(tB, yB, lw=1.2, label=REC_B, color="tab:orange", alpha=0.9)
        ax.set_title(f"{title} vertical ({vert_col})")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        ax.legend(loc="upper right")

    # ----- Right column: Horizontal -----
    ax = axes[r, 1]
    yA = sigA[horiz_col];
    yB = sigB[horiz_col]
    if USE_TWINY:
        l1, = ax.plot(tA, yA, lw=1.2, label=REC_A, color="tab:blue")
        ax.set_ylabel(f"{REC_A} amp")
        ax.set_title(f"{title} horizontal ({horiz_col})")
        ax.grid(True)
        ax2 = ax.twinx()
        l2, = ax2.plot(tB, yB, lw=1.2, label=REC_B, color="tab:orange", alpha=0.9)
        ax2.set_ylabel(f"{REC_B} amp", color="tab:orange")
        ax2.tick_params(axis='y', labelcolor="tab:orange")
        ax.legend([l1, l2], [REC_A, REC_B], loc="upper right")
    else:
        ax.plot(tA, yA, lw=1.2, label=REC_A, color="tab:blue")
        ax.plot(tB, yB, lw=1.2, label=REC_B, color="tab:orange", alpha=0.9)
        ax.set_title(f"{title} horizontal ({horiz_col})")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        ax.legend(loc="upper right")

# bottom x labels
axes[1, 0].set_xlabel("Time (s)")
axes[1, 1].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()

# -------------------- PLOT 2: PSDs (2x2 grid) --------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)

for title, vert_col, horiz_col, r in layout:
    # ----- Left column: Vertical PSD -----
    ax = axes[r, 0]
    fA, PA = tsp_psd(tA, sigA[vert_col], FS)
    fB, PB = tsp_psd(tB, sigB[vert_col], FS)

    if USE_TWINY:
        l1, = ax.plot(fA, PA, label=REC_A, color="tab:blue")
        ax.set_ylabel(f"PSD {REC_A}\n(unit²/Hz)")
        ax.set_title(f"{title} vertical ({vert_col})")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax2 = ax.twinx()
        l2, = ax2.plot(fB, PB, label=REC_B, color="tab:orange", alpha=0.9)
        ax2.set_ylabel(f"PSD {REC_B}\n(unit²/Hz)", color="tab:orange")
        ax2.tick_params(axis='y', labelcolor="tab:orange")
        ax.set_xlim(0, PSD_FMAX)
        ax.legend([l1, l2], [REC_A, REC_B], loc="upper right")
    else:
        ax.plot(fA, PA, label=REC_A, color="tab:blue")
        ax.plot(fB, PB, label=REC_B, color="tab:orange", alpha=0.9)
        ax.set_xlim(0, PSD_FMAX)
        ax.set_ylabel("PSD (unit²/Hz)")
        ax.set_title(f"{title} vertical ({vert_col})")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(loc="upper right")

    # ----- Right column: Horizontal PSD -----
    ax = axes[r, 1]
    fA, PA = tsp_psd(tA, sigA[horiz_col], FS)
    fB, PB = tsp_psd(tB, sigB[horiz_col], FS)

    if USE_TWINY:
        l1, = ax.plot(fA, PA, label=REC_A, color="tab:blue")
        ax.set_ylabel(f"PSD {REC_A}\n(unit²/Hz)")
        ax.set_title(f"{title} horizontal ({horiz_col})")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax2 = ax.twinx()
        l2, = ax2.plot(fB, PB, label=REC_B, color="tab:orange", alpha=0.9)
        ax2.set_ylabel(f"PSD {REC_B}\n(unit²/Hz)", color="tab:orange")
        ax2.tick_params(axis='y', labelcolor="tab:orange")
        ax.set_xlim(0, PSD_FMAX)
        ax.legend([l1, l2], [REC_A, REC_B], loc="upper right")
    else:
        ax.plot(fA, PA, label=REC_A, color="tab:blue")
        ax.plot(fB, PB, label=REC_B, color="tab:orange", alpha=0.9)
        ax.set_xlim(0, PSD_FMAX)
        ax.set_ylabel("PSD (unit²/Hz)")
        ax.set_title(f"{title} horizontal ({horiz_col})")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(loc="upper right")

axes[1, 0].set_xlabel("Frequency (Hz)")
axes[1, 1].set_xlabel("Frequency (Hz)")
plt.tight_layout()
plt.show()
