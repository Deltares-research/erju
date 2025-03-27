import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from DatabaseUtils import get_commands

from src.erju.process_FO_base import BaseFOdata
from src.utils.file_utils import from_window_get_fo_file, compute_psd, bandpass, align_signals, \
    compute_cosine_similarity_windows, compute_psd_fixed
from src.utils.plot_utils import plot_accel_vs_fo, plot_fo_before_after, \
    plot_accel_signals_and_psd, plot_accel_fo_with_psd, plot_accel_filtered_comparison, \
    plot_signals_with_alignment_overlay, plot_cosine_similarity_boxplot, save_aggregated_psd_subplot


# What data we want to compare? Lets tackle it one by one:
# 1. There is the accelerometer data
# 2. There is the FO data


# 1. Accelerometer data ##########################################

# Function to fetch the data from the database based on some given conditions
def fetch_accel_data(db_path: str,
                     start_date: str,
                     end_date: str,
                     locations: list,
                     campaigns: list = None,
                     get_timeseries: bool = False,
                     traintype: str = None,
                     track: str = None):
    """
    Connects to SQLite STEM database and fetches the accelerometer data

    Args:
        db_path (str): Path to the SQLite database
        start_date (str): Start date in the format 'YYYY-MM-DD HH:MM:SS'
        end_date (str): End date in the format 'YYYY-MM-DD HH:MM:SS'
        locations (list): List of locations i.e. ['Meetjournal_MP8_Holten_zuid_4m_C', 'location2']
        campaigns (list): List of campaigns
        get_timeseries (bool): Whether to get the timeseries of the events, default is False
        traintype (str): Filter by traintype (optional)
        track (str): Filter by track (optional)

    Returns:
        events (list): List of events
        tim (dict): Dictionary of timeseries
        mis (list): List of missing files

    """
    # Connect to the database and fetch event data
    conn = get_commands.connect_to_db(db_path)
    # Fetch the events between the given dates and other conditions
    events, tim, mis = get_commands.get_events_between_dates(conn=conn,
                                                             start_date=start_date,
                                                             end_date=end_date,
                                                             locations=locations,
                                                             campaigns=campaigns,
                                                             get_timeseries=get_timeseries,
                                                             traintype=traintype,
                                                             track=track)
    # Close the connection
    get_commands.close_connection(conn)

    # Print the number of events fetched in a given time range
    print(f"Number of events fetched: {len(events)}")

    return events, tim, mis


# Function to extract the time series data from the fetched data
def unpack_timeseries(event: list, data: dict):
    """
    Unpacks accelerometer time series data and returns absolute time and XYZ traces.

    Args:
        event (list/tuple): Event record from the database (event[2] is the start time string)
        data (dict): Dictionary containing 'TIME', 'TRACE_X', 'TRACE_Y', 'TRACE_Z'

    Returns:
        start_time (datetime): Parsed start time of the event
        absolute_time (list of datetime): Absolute timestamps for each data point
        trace_x (np.array): Accelerometer trace in X
        trace_y (np.array): Accelerometer trace in Y
        trace_z (np.array): Accelerometer trace in Z
        time_window (list of datetime): [start, end] of the absolute time range
    """
    # Extract absolute start time of the event
    start_time_str = event[2]
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")

    # Extract time series traces
    relative_time = data["TIME"]
    trace_x = data["TRACE_X"]
    trace_y = data["TRACE_Y"]
    trace_z = data["TRACE_Z"]

    # Convert to absolute timestamps
    absolute_time = [start_time + timedelta(seconds=t) for t in relative_time]

    # Define time window
    time_window = [absolute_time[0], absolute_time[-1]]

    return absolute_time, trace_x, trace_y, trace_z, time_window


def estimate_sampling_frequency(time_vector):
    """
    Estimate the sampling frequency [Hz] from a list of datetime timestamps.

    Args:
        time_vector (list of datetime): List of datetime objects representing time axis.

    Returns:
        float: Estimated sampling frequency in Hz.
    """
    if len(time_vector) < 2:
        raise ValueError("Need at least two timestamps to compute sampling frequency.")

    # Calculate time differences in seconds
    time_deltas = np.diff([t.timestamp() for t in time_vector])
    avg_delta = np.mean(time_deltas)

    return 1.0 / avg_delta


if __name__ == "__main__":
    # Define the paths
    path_db = r"P:/11207352-stem/database/Wielrondheid_132887.db"
    path_fo = r'E:\recording_2024-09-06T11_58_54Z_5kHzping_1kHzlog_1mCS_10mGL_6000channels'

    # Time range for extracting events
    start_date = '2024-09-07 00:00:00'
    end_date = '2024-09-08 00:00:00'
    # Parameters for querying the database
    locations = ['Meetjournal_MP8_Holten_zuid_4m_C']
    campaigns = None
    traintype = "SPR(A)"
    track = "2"
    # fo channels
    first_channel = 1189
    last_channel = 1199

    # Fetch the accelerometer data
    events, tim, mis = fetch_accel_data(db_path=path_db,
                                        start_date=start_date,
                                        end_date=end_date,
                                        locations=locations,
                                        campaigns=campaigns,
                                        traintype=traintype,
                                        track=track,
                                        get_timeseries=True)

    # Get the event time series dictionary (assuming one location)
    event_series = list(tim.values())[0]

    # Initialize storage for similarities
    similarity_scores_x = []
    similarity_scores_y = []
    similarity_scores_z = []

    # Initialize storage for PSDs
    psd_x_all = []
    psd_y_all = []
    psd_z_all = []
    psd_fo_all = []
    freqs_shared = None

    # Loop through each event and its corresponding time series
    for event, (event_id, data) in zip(events, event_series.items()):
        # Unpack the time series data
        absolute_time, trace_x, trace_y, trace_z, time_window = unpack_timeseries(event, data)

        # The frequency of the accelerometer data is 1000 Hz.
        freq_accel = estimate_sampling_frequency(absolute_time)
        print(f"Estimated sampling frequency of accelerometer data: {freq_accel} Hz")

        # Bandpass filter
        trace_x_filt = bandpass(trace_x, 1, 100, 1000, 4)
        trace_y_filt = bandpass(trace_y, 1, 100, 1000, 4)
        trace_z_filt = bandpass(trace_z, 1, 100, 1000, 4)

        # 2 Lets look at the FO data ##########################################

        # Lets create an instance of the BaseFOdata class

        fo = BaseFOdata.create_instance(dir_path=path_fo,
                                        first_channel=first_channel,
                                        last_channel=last_channel,
                                        reader='optasense')
        # For each event, find the files in the time window
        fo_files_in_event = from_window_get_fo_file(path_fo, time_window)

        # Now we loop through the fo files one by one and extract the data
        # First lets create a container to store the fo data
        fo_data = []
        super_raw_data = []
        # Now the loop
        for fo_file in fo_files_in_event:
            # Get properties from the first file
            if fo_file == fo_files_in_event[0]:
                fo.extract_properties_per_file(fo_file)
                file_start_time = fo.properties['FileStartTime']
                sampling_frequency = int(fo.properties['SamplingFrequency[Hz]'])

            # Try using the extract_data from the BaseFOdata class
            # This function already has a bandpass filter implemented, as well as a
            # conversion form optical phase to strain
            processed_data, raw_signal_data = fo.extract_data(file_name=fo_file,
                                                              first_channel=first_channel,
                                                              last_channel=last_channel)

            # In the original code, the data is transposed, so we will un-transpose it
            # Append the data to the list
            fo_data.append(processed_data.T)
            super_raw_data.append(raw_signal_data)

        # Concatenate FO data from multiple files
        fo_data = np.concatenate(fo_data, axis=0)
        super_raw_data = np.concatenate(super_raw_data, axis=0)

        # Compute timestamps for FO data
        timestamps = [file_start_time + timedelta(seconds=i / sampling_frequency) for i in range(fo_data.shape[0])]

        # Convert timestamps to NumPy datetime64 for indexing
        timestamps_array = np.array(timestamps, dtype='datetime64[ns]')

        # Find closest start and end indices within the FO timestamps
        start_time_np = np.datetime64(time_window[0])
        end_time_np = np.datetime64(time_window[1])
        start_index = np.argmin(np.abs(timestamps_array - start_time_np))
        end_index = np.argmin(np.abs(timestamps_array - end_time_np))

        # Crop FO data to the time window
        timestamps = timestamps[start_index:end_index + 1]
        fo_data = fo_data[start_index:end_index + 1, :]
        super_raw_data = super_raw_data[start_index:end_index + 1, :]

        # Compute PSDs
        fx, psd_x = compute_psd(trace_x, fs=1000)
        fy, psd_y = compute_psd(trace_y, fs=1000)
        fz, psd_z = compute_psd(trace_z, fs=1000)
        ff, psd_fo = compute_psd(fo_data[:, 1194 - first_channel], fs=sampling_frequency)

        # Compute windowed cosine similarity
        window_size = 256  # samples
        overlap = 128  # 50% overlap

        ch_index = 1194 - first_channel
        fo_trace = fo_data[:, ch_index]
        aligned_fo, lag = align_signals(trace_x, fo_trace)

        scores_x = compute_cosine_similarity_windows(trace_x, aligned_fo, window_size, overlap)
        scores_y = compute_cosine_similarity_windows(trace_y, aligned_fo, window_size, overlap)
        scores_z = compute_cosine_similarity_windows(trace_z, aligned_fo, window_size, overlap)

        similarity_scores_x.extend(scores_x)
        similarity_scores_y.extend(scores_y)
        similarity_scores_z.extend(scores_z)

        # Compute PSDs with fixed frequency bins (shared across events)
        fx, psd_x = compute_psd_fixed(trace_x_filt, fs=1000)
        fy, psd_y = compute_psd_fixed(trace_y_filt, fs=1000)
        fz, psd_z = compute_psd_fixed(trace_z_filt, fs=1000)
        ff, psd_fo = compute_psd_fixed(aligned_fo, fs=sampling_frequency)

        # Save frequencies once
        if freqs_shared is None:
            freqs_shared = fx

        psd_x_all.append(psd_x)
        psd_y_all.append(psd_y)
        psd_z_all.append(psd_z)
        psd_fo_all.append(psd_fo)

        # Plot the accelerometer vs FO data
        # plot_accel_vs_fo(
        #     save_dir=r"N:\Projects\11210000\11210064\B. Measurements and calculations\holten\new_analysis\accel_vs_fo",
        #     event_id=event_id,
        #     accel_time=absolute_time,
        #     trace_x=trace_x,
        #     trace_y=trace_y,
        #     trace_z=trace_z,
        #     fo_time=timestamps,
        #     fo_data=fo_data,
        #     fo_channel=1194,
        #     first_channel=first_channel,
        #     save_interactive=True
        # )

        # Plot FO data before and after filtering
        # plot_fo_before_after(
        #     save_dir=r"N:\Projects\11210000\11210064\B. Measurements and calculations\holten\new_analysis\before_after_filters",
        #     event_id=event_id,
        #     timestamps=timestamps,
        #     raw_signal_data=super_raw_data,
        #     processed_data=fo_data,
        #     fo_channel=1194,
        #     first_channel=first_channel,
        #     save_interactive=True  # set False if you just want PNG
        # )

        # plot_accel_signals_and_psd(event_id,
        #                            absolute_time,
        #                            trace_x,
        #                            trace_y,
        #                            trace_z,
        #                            fs=1000,
        #                            save_dir=r"N:\Projects\11210000\11210064\B. Measurements and calculations\holten\new_analysis\accel_with_psd")

        # Accelerometer data and fo with PSD's for 128/256/512
        # plot_accel_fo_with_psd(event_id,
        #                        absolute_time,
        #                        trace_x,
        #                        trace_y,
        #                        trace_z,
        #                        timestamps,
        #                        fo_data,
        #                        fo_channel=1194,
        #                        first_channel=first_channel,
        #                        fs_accel=1000,
        #                        fs_fo=sampling_frequency,
        #                        save_dir=r"N:\Projects\11210000\11210064\B. Measurements and calculations\holten\new_analysis\accel_fo_with_psd_512",
        #                        freq_range=(0, 100),
        #                        save_interactive=True
        #                        )
        #
        # Accelerometer and fo data WITHOUT FILTER OR STRAIN CONVERSION
        # plot_accel_fo_with_psd(event_id,
        #                        absolute_time,
        #                        trace_x,
        #                        trace_y,
        #                        trace_z,
        #                        timestamps,
        #                        super_raw_data,
        #                        fo_channel=1194,
        #                        first_channel=first_channel,
        #                        fs_accel=1000,
        #                        fs_fo=sampling_frequency,
        #                        save_dir=r"N:\Projects\11210000\11210064\B. Measurements and calculations\holten\new_analysis\accel_foRAW_with_psd",
        #                        freq_range=(0, 100),
        #                        save_interactive=True)

        # Accelerometer FILTERED DATA and fo data
        # plot_accel_fo_with_psd(event_id,
        #                        absolute_time,
        #                        trace_x_filt,
        #                        trace_y_filt,
        #                        trace_z_filt,
        #                        timestamps,
        #                        fo_data,
        #                        fo_channel=1194,
        #                        first_channel=first_channel,
        #                        fs_accel=1000,
        #                        fs_fo=sampling_frequency,
        #                        save_dir=r"N:\Projects\11210000\11210064\B. Measurements and calculations\holten\new_analysis\accelF_foF_with_psd",
        #                        freq_range=(0, 100),
        #                        save_interactive=True)

        # plot_accel_filtered_comparison(
        #     event_id,
        #     absolute_time,
        #     trace_x,
        #     trace_y,
        #     trace_z,
        #     trace_x_filt,
        #     trace_y_filt,
        #     trace_z_filt,
        #     save_dir=r"N:\Projects\11210000\11210064\B. Measurements and calculations\holten\new_analysis\accel_filt_nofilt_compare",
        # )

        # IN ORDER TO CHECK ALLIGNMENT BETWEEN FO AND ACCELEROMETER DATA
        # plot_signals_with_alignment_overlay(event_id, timestamps, trace_x, trace_y, trace_z, aligned_fo)

    # # PLOT THE COSINE SIMILARITY BOXPLOT
    # plot_cosine_similarity_boxplot(
    #     sim_x=similarity_scores_x,
    #     sim_y=similarity_scores_y,
    #     sim_z=similarity_scores_z,
    #     save_dir=r"N:\Projects\11210000\11210064\B. Measurements and calculations\holten\new_analysis\cosine",
    #     filename="cosine_similarity_boxplot.png"
    # )

    save_aggregated_psd_subplot(freqs_shared,
                                psds_x=psd_x_all,
                                psds_y=psd_y_all,
                                psds_z=psd_z_all,
                                psds_fo=psd_fo_all,
                                save_dir=r"N:\Projects\11210000\11210064\B. Measurements and calculations\holten\new_analysis\psd_summary",
                                filename="aggregated_psd.png")
