"""
Old script to compare channels in the FO data. You give a center channel and a step size, and it will plot the
channels around the center channel. This is now improved and implemented in the function
called 'plot_fo_window_and_psd_grid' inside the 'compare_data.py' script.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from datetime import datetime, timedelta
from DatabaseUtils import get_commands
from src.utils.file_utils import from_window_get_fo_file, calculate_sampling_frequency, highpass, bandpass

# ---------------------- CONFIGURATION ----------------------

# Paths
path_database = r"P:/11207352-stem/database/Wielrondheid_132887.db"
path_fo_files = r'E:\recording_2024-09-06T11_58_54Z_5kHzping_1kHzlog_1mCS_10mGL_6000channels'

# Channel configuration
center_channel = 1194  # Center channel for plotting
step = 20  # Channel spacing: 1=continuous, >1=spaced
num_instances = 10  # Number of channels above and below center

# Save directory (dynamic naming)
base_save_dir = r"N:/Projects/11210000/11210064/B. Measurements and calculations/holten"
folder_suffix = f"ch_compare_ctr{center_channel}_step{step}_n{num_instances}"
save_dir = os.path.join(base_save_dir, folder_suffix)
os.makedirs(save_dir, exist_ok=True)

# Time range for extracting events
start_date = '2024-09-07 00:00:00'
end_date = '2024-09-08 00:00:00'

# Parameters for querying the database
locations = ['Meetjournal_MP8_Holten_zuid_4m_C']
campaigns = None
traintype = "SPR(A)"
track = "2"

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


# ---------------------- GET EVENT DATA ----------------------

# Connect to the database and fetch event data
conn = get_commands.connect_to_db(path_database)
events, tim, mis = get_commands.get_events_between_dates(
    conn, start_date, end_date, locations, campaigns, get_timeseries=True, traintype=traintype, track=track
)
get_commands.close_connection(conn)

# ---------------------- PROCESS EACH EVENT ----------------------

# Select channels around center with specified step
channels_to_plot = sorted([
    center_channel + i * step
    for i in range(-num_instances, num_instances + 1)
    if center_channel + i * step >= 0  # Avoid negative channel index
])

# Loop through each event and its associated accelerometer time series
for event_index, (event, (inner_key, data)) in enumerate(zip(events, list(tim.values())[0].items()), start=1):

    # Extract absolute event start time from database
    start_time_str = event[2]  # Position 2 contains the start time
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")

    # Extract accelerometer time series data
    relative_time = data["TIME"]
    # Convert relative times to absolute times
    absolute_time = [start_time + timedelta(seconds=t) for t in relative_time]
    # Define the time window for FO data extraction
    time_window = [absolute_time[0], absolute_time[-1]]

    # ---------------------- GET FO DATA ----------------------

    # Find FO files in the given time window
    fo_files_in_event = from_window_get_fo_file(path_fo_files, time_window)

    if not fo_files_in_event:
        print(f"No FO files found for event {event_index}. Skipping...")
        continue

    # Initialize variables
    fo_data = []
    timestamps = []

    for fo_file in fo_files_in_event:
        with h5py.File(fo_file, 'r') as file:
            # Extract relevant data for selected channels
            file_raw_data = file['Acquisition']['Raw[0]']['RawData'][:, channels_to_plot]
            fo_data.append(file_raw_data)

            # If this is the first file, extract metadata (start time & sampling frequency)
            if fo_file == fo_files_in_event[0]:
                raw_data_time = file['Acquisition']['Raw[0]']['RawDataTime']
                file_start_time = datetime.utcfromtimestamp(raw_data_time[0] * 1e-6)
                sampling_frequency = calculate_sampling_frequency(file)

    # Concatenate all FO data
    raw_fo_data = np.concatenate(fo_data, axis=0)

    # Compute timestamps for FO data
    timestamps = [file_start_time + timedelta(seconds=i / sampling_frequency) for i in range(raw_fo_data.shape[0])]

    # Convert timestamps to NumPy datetime64 for indexing
    timestamps_array = np.array(timestamps, dtype='datetime64[ns]')

    # Find closest start and end indices within the FO timestamps
    start_time_np = np.datetime64(time_window[0])
    end_time_np = np.datetime64(time_window[1])
    start_index = np.argmin(np.abs(timestamps_array - start_time_np))
    end_index = np.argmin(np.abs(timestamps_array - end_time_np))

    # Crop FO data to the time window
    timestamps = timestamps[start_index:end_index + 1]
    raw_fo_data = raw_fo_data[start_index:end_index + 1, :]

    # ---------------------- APPLY FILTERS ----------------------

    # Create arrays for filtered FO data
    raw_data_highpass = np.empty_like(raw_fo_data)
    raw_data_bandpass = np.empty_like(raw_fo_data)

    for channel in range(raw_fo_data.shape[1]):
        # Apply high-pass filter
        raw_data_highpass[:, channel] = highpass(data=raw_fo_data[:, channel], cutoff=0.1)

        # Apply bandpass filter
        raw_data_bandpass[:, channel] = bandpass(
            data=raw_data_highpass[:, channel], freqmin=freq_min, freqmax=freq_max, fs=sampling_frequency,
            corners=filter_order
        )

    # ---------------------- PLOT FO CHANNELS & PSD ----------------------

    num_channels = len(channels_to_plot)  # Total number of plotted channels

    fig, axes = plt.subplots(num_channels, 2, figsize=(14, num_channels * 2), sharex='col',
                             gridspec_kw={'width_ratios': [2, 1]})

    for i, channel_idx in enumerate(channels_to_plot):
        color = 'r' if channel_idx == center_channel else 'k'  # Center channel is red

        # Plot time-domain signal (Left Column)
        axes[i, 0].plot(timestamps, raw_data_bandpass[:, i], color=color, alpha=0.7)
        axes[i, 0].set_ylabel(f"Ch {channel_idx}")
        axes[i, 0].grid(True, linestyle='--', alpha=0.5)

        # Compute and plot PSD (Right Column)
        freq, psd = compute_psd(raw_data_bandpass[:, i], fs=sampling_frequency, length_w=length_w)
        axes[i, 1].semilogx(freq, psd, color=color, alpha=0.7)  # Log scale for frequency
        axes[i, 1].set_ylabel(f"PSD Ch {channel_idx}")
        axes[i, 1].grid(True, linestyle='--', alpha=0.5)

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Frequency [Hz]")

    plt.suptitle(f"FO Channel Comparison & PSD: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Save the plot
    filename = f"ch_comparison_PSD_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

    print(f"Saved channel comparison plot for event {event_index} as {filename}")
