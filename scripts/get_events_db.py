import os
import numpy as np
from scipy import signal
from datetime import datetime, timedelta
from DatabaseUtils import get_commands, plot_signal


def compute_psd(signal_data, fs, length_w=128):
    """Compute PSD using Welch's method."""
    freq, psd = signal.welch(signal_data, fs=fs, nperseg=length_w, window='hamming', scaling='density')
    return freq, psd


def estimate_sampling_frequency(time_series):
    """
    Estimate the sampling frequency of a time series.

    :param time_series: list or numpy array of datetime objects
    :return: estimated sampling frequency in Hz
    """
    time_diffs = np.diff(time_series)  # Compute time intervals (returns timedelta)
    median_dt = np.median([td.total_seconds() for td in time_diffs])  # Convert to seconds
    fs = 1 / median_dt if median_dt > 0 else None  # Avoid division by zero
    return fs


directory = r"P:/11207352-stem/database/Wielrondheid_132887.db"

conn = get_commands.connect_to_db(directory)
start_date = '2024-09-07 00:00:00'
end_date = '2024-09-08 00:00:00'
locations = ['Meetjournal_MP8_Holten_zuid_4m_C']
campaigns = None
traintype = "SPR(A)"
track = "2"
events, tim, mis = get_commands.get_events_between_dates(conn,
                                                         start_date,
                                                         end_date,
                                                         locations,
                                                         campaigns,
                                                         get_timeseries=True,
                                                         traintype=traintype,
                                                         track=track)
get_commands.close_connection(conn)

counter = 0  # Initialize the counter for plot file names
save_dir = r"N:/Projects/11210000/11210064/B. Measurements and calculations/holten/accel_PSD/"

# Ensure the save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Ensure event metadata and time series data are correctly paired
for event, (inner_key, data) in zip(events, list(tim.values())[0].items()):
    # Extract absolute event start time
    start_time_str = event[2]  # Assuming event[2] contains the absolute start time as string
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")

    print(f"Event Start Time: {start_time}")
    print(f"  Matfile: {inner_key}")

    # Extract time series data
    relative_time = data["TIME"]
    trace_x = data["TRACE_X"]
    trace_y = data["TRACE_Y"]
    trace_z = data["TRACE_Z"]

    # Compute absolute time series
    absolute_time = [start_time + timedelta(seconds=t) for t in relative_time]

    # Calculate sampling frequency
    fs = estimate_sampling_frequency(absolute_time)
    if fs is None:
        print(f"Skipping event {counter} due to invalid sampling frequency.")
        continue

    # Compute PSD using your function
    freq_x, psd_x = compute_psd(trace_x, fs)
    freq_y, psd_y = compute_psd(trace_y, fs)
    freq_z, psd_z = compute_psd(trace_z, fs)

    print(f"  Time Series Length: {len(relative_time)}")
    print(f"  Sample Absolute Time: {absolute_time[:5]}")
    print(f"  Sample X Signal: {trace_x[:5]}")  # Print first 5 values
    print(f"  Sample Y Signal: {trace_y[:5]}")
    print(f"  Sample Z Signal: {trace_z[:5]}")
    print("")

    title = (f"Campaign/Location: {locations}\n"
             f"Matfile: {inner_key}\n"
             f"Traintype: {traintype} and Track: {track}")

    save_path = os.path.join(save_dir, f"Event_{counter}_in_{inner_key}.png")

    plot_signal.plot_simple_signal(trace_x, trace_y, trace_z, absolute_time,
                                   title, save=False, save_path=save_path)
