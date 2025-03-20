import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from datetime import datetime, timedelta, UTC
from DatabaseUtils import get_commands
from utils.file_utils import from_window_get_fo_file, calculate_sampling_frequency, highpass, bandpass

# ---------------------- CONFIGURATION ----------------------

# Paths
path_database = r"P:/11207352-stem/database/Wielrondheid_132887.db"
path_fo_files = r'E:\recording_2024-09-06T11_58_54Z_5kHzping_1kHzlog_1mCS_10mGL_6000channels'
save_dir = r"N:/Projects/11210000/11210064/B. Measurements and calculations/holten/accel_fo_psd/"

# Ensure save directory exists
os.makedirs(save_dir, exist_ok=True)

# Time range for extracting events
start_date = '2024-09-07 00:00:00'
end_date = '2024-09-08 00:00:00'

# Parameters for querying the database
locations = ['Meetjournal_MP8_Holten_zuid_4m_C']
campaigns = None
traintype = "SPR(A)"
track = "2"

# Sensor channel configuration
center_channel = 1194
channel_range = 5
start_channel = max(0, center_channel - channel_range)  # Ensure start channel is not negative
end_channel = center_channel + channel_range
relative_center_channel = channel_range * 2 // 2  # Calculate the center in the new channel selection

# Bandpass filter parameters
freq_min = 0.1  # Minimum frequency for bandpass filter
freq_max = 100  # Maximum frequency for bandpass filter
filter_order = 4  # Filter order

# PSD computation parameters
length_w = 128  # Window length for PSD computation


# ---------------------- FUNCTION TO COMPUTE PSD ----------------------

def compute_psd(signal_data, fs, length_w=128):
    """Compute PSD using Welch's method."""
    freq, psd = signal.welch(signal_data, fs=fs, nperseg=length_w, window='hamming', scaling='density')
    return freq, psd


# ---------------------- GET ACCELEROMETER DATA ----------------------

# Connect to the database and fetch event data
conn = get_commands.connect_to_db(path_database)
events, tim, mis = get_commands.get_events_between_dates(
    conn, start_date, end_date, locations, campaigns, get_timeseries=True, traintype=traintype, track=track
)
get_commands.close_connection(conn)

# ---------------------- PROCESS EACH EVENT ----------------------

# Loop through each event and its associated accelerometer time series
for event_index, (event, (inner_key, data)) in enumerate(zip(events, list(tim.values())[0].items()), start=1):

    # Extract absolute event start time from database
    start_time_str = event[2]  # Assuming event[2] is a string
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")

    # Extract accelerometer time series data
    relative_time = data["TIME"]
    trace_x = data["TRACE_X"]
    trace_y = data["TRACE_Y"]
    trace_z = data["TRACE_Z"]

    # Convert relative times to absolute times
    absolute_time = [start_time + timedelta(seconds=t) for t in relative_time]
    absolute_time_unix = np.array([t.timestamp() for t in absolute_time])  # Convert to UNIX timestamps

    # Define the time window for FO data extraction
    time_window = [absolute_time[0], absolute_time[-1]]

    # ---------------------- GET FO DATA ----------------------

    # Find the relevant FO files that match the time window
    fo_files_in_event = from_window_get_fo_file(path_fo_files, time_window)

    # Initialize FO data variables
    timestamps = []
    fo_resampled = None
    sampling_frequency = None  # To store FO sampling frequency

    if fo_files_in_event:  # Only process FO data if files are found
        fo_data = []  # List to collect raw FO data

        for fo_file in fo_files_in_event:
            with h5py.File(fo_file, 'r') as file:
                file_raw_data = file['Acquisition']['Raw[0]']['RawData'][:, start_channel:end_channel + 1]
                fo_data.append(file_raw_data)

                # Extract metadata from the first FO file
                if fo_file == fo_files_in_event[0]:
                    raw_data_time = file['Acquisition']['Raw[0]']['RawDataTime']
                    file_start_time = datetime.fromtimestamp(raw_data_time[0] * 1e-6, UTC)  # ✅ FIXED
                    sampling_frequency = calculate_sampling_frequency(file)

        # Concatenate FO data
        raw_data = np.concatenate(fo_data, axis=0)

        # Compute timestamps for FO data
        timestamps = [file_start_time + timedelta(seconds=i / sampling_frequency) for i in range(raw_data.shape[0])]
        timestamps_unix = np.array([t.timestamp() for t in timestamps])

        raw_data = raw_data.astype(np.float64)
        # **Fix FO Scaling**: Normalize before resampling
        raw_data[:, relative_center_channel] -= np.mean(raw_data[:, relative_center_channel])
        raw_data[:, relative_center_channel] /= np.std(raw_data[:, relative_center_channel])

        # Resample FO data
        fo_resampled = np.interp(absolute_time_unix, timestamps_unix, raw_data[:, relative_center_channel])

    # ---------------------- PLOT AND SAVE RAW SIGNALS WITH PSD ----------------------

    fig, axes = plt.subplots(3, 2, figsize=(16, 8), sharex='col', gridspec_kw={'width_ratios': [2, 1]})

    for i, (trace, label) in enumerate(zip([trace_x, trace_y, trace_z], ["X", "Y", "Z"])):
        # **Time-Domain Plot (Left Column)**
        ax1 = axes[i, 0]
        ax2 = ax1.twinx()  # Dual Y-axis for FO data
        ax1.plot(absolute_time, trace, label=f"Accelerometer {label}", color='b', alpha=0.7)

        if fo_resampled is not None:
            ax2.plot(absolute_time, fo_resampled, label="FO Data", color='r', alpha=0.7)

        ax1.set_ylabel(f"{label}")
        ax1.legend(loc="upper left")
        ax2.set_ylabel("FO Signal")
        ax2.legend(loc="upper right")
        ax1.grid(True, linestyle='--', alpha=0.5)

        # **PSD Plot (Right Column)**
        ax3 = axes[i, 1]
        ax4 = ax3.twinx()  # Dual Y-axis for FO PSD

        freq, psd_trace = compute_psd(trace, fs=1000, length_w=length_w)
        ax3.semilogx(freq, psd_trace, label=f"Accel {label} PSD", color='b', alpha=0.7)

        if fo_resampled is not None:
            freq, psd_fo = compute_psd(fo_resampled, fs=sampling_frequency, length_w=length_w)
            ax4.semilogx(freq, psd_fo, label="FO PSD", color='r', alpha=0.7)

        ax3.set_ylabel(f"PSD {label}")
        ax3.legend(loc="upper left")
        ax4.set_ylabel("FO PSD")
        ax4.legend(loc="upper right")
        ax3.grid(True, linestyle='--', alpha=0.5)

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Frequency [Hz]")

    plt.suptitle(f"Accelerometer & FO Signals with PSD: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Save the plot
    filename = f"Event_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}_fo_accel_PSD.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

    print(f"Saved signal & PSD plot for event {event_index} as {filename}")