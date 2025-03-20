import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from DatabaseUtils import get_commands
from utils.file_utils import from_window_get_fo_file, calculate_sampling_frequency, highpass, bandpass


# ---------------------- CONFIGURATION ----------------------

# Paths
path_database = r"P:/11207352-stem/database/Wielrondheid_132887.db"
path_fo_files = r'E:\recording_2024-09-06T11_58_54Z_5kHzping_1kHzlog_1mCS_10mGL_6000channels'
save_dir = r"N:/Projects/11210000/11210064/B. Measurements and calculations/holten/accel_vs_fo/"

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

    # Define the time window for FO data extraction
    time_window = [absolute_time[0], absolute_time[-1]]

    # ---------------------- GET FO DATA ----------------------

    # Find the relevant FO files that match the time window
    fo_files_in_event = from_window_get_fo_file(path_fo_files, time_window)

    # Initialize FO data variables (to avoid NameError if no FO data is found)
    timestamps = []
    raw_data_bandpass = None  # Will store the final processed FO signal

    if fo_files_in_event:  # Only process FO data if files are found
        fo_data = []  # List to collect raw FO data

        for fo_file in fo_files_in_event:
            with h5py.File(fo_file, 'r') as file:
                # Extract relevant data
                file_raw_data = file['Acquisition']['Raw[0]']['RawData'][:, start_channel:end_channel + 1]
                fo_data.append(file_raw_data)

                # If this is the first file, extract metadata (start time & sampling frequency)
                if fo_file == fo_files_in_event[0]:
                    raw_data_time = file['Acquisition']['Raw[0]']['RawDataTime']
                    file_start_time = datetime.utcfromtimestamp(raw_data_time[0] * 1e-6)
                    sampling_frequency = calculate_sampling_frequency(file)

        # Concatenate FO data from multiple files
        raw_data = np.concatenate(fo_data, axis=0)

        # Compute timestamps for FO data
        timestamps = [file_start_time + timedelta(seconds=i / sampling_frequency) for i in range(raw_data.shape[0])]

        # Convert timestamps to NumPy datetime64 for indexing
        timestamps_array = np.array(timestamps, dtype='datetime64[ns]')

        # Find closest start and end indices within the FO timestamps
        start_time_np = np.datetime64(time_window[0])
        end_time_np = np.datetime64(time_window[1])
        start_index = np.argmin(np.abs(timestamps_array - start_time_np))
        end_index = np.argmin(np.abs(timestamps_array - end_time_np))

        # Crop FO data to the time window
        timestamps = timestamps[start_index:end_index + 1]
        raw_data = raw_data[start_index:end_index + 1, :]

        # ---------------------- APPLY FILTERS ----------------------

        # Create arrays for filtered FO data
        raw_data_highpass = np.empty_like(raw_data)
        raw_data_bandpass = np.empty_like(raw_data)

        for channel in range(raw_data.shape[1]):
            # Apply high-pass filter
            raw_data_highpass[:, channel] = highpass(data=raw_data[:, channel], cutoff=0.1)

            # Apply bandpass filter
            raw_data_bandpass[:, channel] = bandpass(
                data=raw_data_highpass[:, channel], freqmin=0.1, freqmax=100, fs=1000, corners=4
            )


# ---------------------- RESAMPLE FO DATA TO ACCELEROMETER TIMEBASE ----------------------

    if timestamps and raw_data_bandpass is not None:
        timestamps_unix = np.array([t.timestamp() for t in timestamps])
        absolute_time_unix = np.array([t.timestamp() for t in absolute_time])

        fo_resampled = np.interp(absolute_time_unix, timestamps_unix, raw_data_bandpass[:, relative_center_channel])


    # ---------------------- PLOT AND SAVE RAW SIGNALS (WITH DUAL Y-AXES) ----------------------
    fig, ax_raw = plt.subplots(3, 1, figsize=(15, 8), sharex=True)

    # Create secondary Y-axes for FO data
    ax_raw_fo = [ax.twinx() for ax in ax_raw]

    # Plot accelerometer data (raw) with transparency
    ax_raw[0].plot(absolute_time, trace_x, label="Accelerometer X", color='k', alpha=0.7)
    ax_raw[1].plot(absolute_time, trace_y, label="Accelerometer Y", color='k', alpha=0.7)
    ax_raw[2].plot(absolute_time, trace_z, label="Accelerometer Z", color='k', alpha=0.7)

    # Plot FO data if available (raw) with transparency on secondary Y-axes
    if fo_resampled is not None:
        ax_raw_fo[0].plot(absolute_time, fo_resampled, label="FO Data", color='r', alpha=0.7)
        ax_raw_fo[1].plot(absolute_time, fo_resampled, label="FO Data", color='r', alpha=0.7)
        ax_raw_fo[2].plot(absolute_time, fo_resampled, label="FO Data", color='r', alpha=0.7)

    # Labels, legend, and grid
    for ax, ax_fo in zip(ax_raw, ax_raw_fo):
        ax.legend(loc="upper left")
        ax.set_ylabel("Accelerometer Signal")
        ax.grid(True, linestyle='--', alpha=0.5)  # Dashed gridlines with transparency
        ax_fo.set_ylabel("FO Signal")  # Label for FO secondary Y-axis
        ax_fo.legend(loc="upper right")

    ax_raw[2].set_xlabel("Time [s]")
    plt.suptitle(f"Raw Signals: {start_time}")

    # Save the raw signal plot
    filename_raw = f"Event_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}_fo_accel.png"
    plt.savefig(os.path.join(save_dir, filename_raw))
    plt.close()

