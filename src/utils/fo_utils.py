import os
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from src.utils.file_utils import get_files_in_dir, extract_timestamp_from_name
from src.erju.process_FO_base import BaseFOdata


def from_window_get_fo_file(fo_data_path: str, time_window: list):
    """
    Get the FO signal data files that fall within the given time window and extend the selection
    by including one file before and one file after each match to ensure no data is lost at the edges.

    IMPORTANT NOTES:
      - The .h5 file naming format is: sensor_2024-09-07T060828Z.h5, where the timestamp
        (2024-09-07T060828) indicates the time of the first measurement in that file.
      - Each file covers data from its start time until the start time of the next file.
        For the last file, we assume its duration is the same as the previous interval (or a default if only one file exists).

    Args:
        fo_data_path (str): Path to the folder containing FO signal files.
        time_window (list): A list containing the start and end time [start_time, end_time] as datetime objects.

    Returns:
        list: A list of full file paths that cover the event time window, extended by one file before and after.
    """
    # Ensure time_window is a list of two datetime objects
    if not (isinstance(time_window, list) and len(time_window) == 2 and
            all(isinstance(t, datetime) for t in time_window)):
        raise ValueError("time_window must be a list of two datetime objects: [start_time, end_time]")

    event_start, event_end = time_window

    # Get a list of all the .h5 file names in the folder (assumed sorted alphabetically, which also sorts them timewise)
    file_names = get_files_in_dir(folder_path=fo_data_path, file_format='.h5')

    # Extract timestamps from the file names.
    # This function should return a list of timestamps in a format that can be converted to pd.Timestamp.
    fo_timestamps = extract_timestamp_from_name(file_names)

    # Convert each timestamp to a pandas Timestamp object (this includes the full date and time)
    fo_timestamps = [pd.Timestamp(ts) for ts in fo_timestamps]

    # Build time intervals for each file.
    # Each file is assumed to cover the time interval [its timestamp, next file's timestamp)
    intervals = []
    for i in range(len(fo_timestamps)):
        start = fo_timestamps[i]
        if i < len(fo_timestamps) - 1:
            # End of this file is the start of the next file
            end = fo_timestamps[i + 1]
        else:
            # For the last file, assume the duration is the same as the previous file's duration (if available)
            if i > 0:
                duration = fo_timestamps[i] - fo_timestamps[i - 1]
                end = fo_timestamps[i] + duration
            else:
                # If there is only one file, assume a default duration (e.g., 1 minute)
                end = fo_timestamps[i] + pd.Timedelta(minutes=1)
                logger.info(f"Only one file. Assuming a default duration of 1 minute.")
        intervals.append((start, end))

    # Find file indices where the file's time interval intersects the event time window.
    # Two intervals intersect if the file interval starts before the event ends
    # and the file interval ends after the event starts.
    matching_indices = []
    for i, (int_start, int_end) in enumerate(intervals):
        if int_start < event_end and int_end > event_start:
            matching_indices.append(i)

    # Extend the selection by including one file before and one file after each matching file.
    extended_indices = set(matching_indices)
    for idx in matching_indices:
        if idx > 0:  # Add the previous file if it exists
            extended_indices.add(idx - 1)
        if idx < len(file_names) - 1:  # Add the next file if it exists
            extended_indices.add(idx + 1)

    # Convert the indices back to a sorted list of full file paths
    matching_file_paths = [os.path.join(fo_data_path, file_names[i]) for i in sorted(extended_indices)]

    return matching_file_paths


def get_fo_data_from_window(fo_data_path: str, first_channel: int, last_channel: int, center_channel: int,
                            time_window: list):
    """
    Given an event (and thus, it's time window), this function will find all the FO files that have data
    in that time window, extract it, append it together and return it. Note that it return two variables:
    One is the data just transformed to strain, the second is the data with the bandpass applied internally.

    """
    # Create an instance of the BaseFOdata class
    fo = BaseFOdata(path_fo=fo_data_path, first_channel=first_channel,
                    last_channel=last_channel, center_channel=center_channel)

    fo_strain_data = []
    fo_strain_and_bandpass_data = []

    # Find the FO files that are in the time window
    fo_files_in_event = from_window_get_fo_file(fo_data_path=fo_data_path, time_window=time_window)

    # Loop through each file and extract the data
    for i, fo_file in enumerate(fo_files_in_event):
        if i == 0:
            # If it is the first file:
            fo.extract_properties_per_file(fo_file)
            file_start_time = fo.properties['FileStartTime']
            sampling_frequency = fo.properties['SamplingFrequency[Hz]']

        # Extract the data from the file
        fo_processed_data, fo_strain_data = fo.extract_data(file_name=fo_file,
                                                            first_channel=first_channel,
                                                            last_channel=last_channel)

        # Transpose the data to match the expected format
        fo_processed_data = fo_processed_data.T
        fo_strain_data = fo_strain_data.T

        # Append the data to the lists
        fo_strain_data.append(fo_strain_data)
        fo_strain_and_bandpass_data.append(fo_processed_data)

    # Concatenate the data from all files
    fo_strain_data = np.concatenate(fo_strain_data, axis=0)
    fo_strain_and_bandpass_data = np.concatenate(fo_strain_and_bandpass_data, axis=0)

    # Compute the timestamps for the FO data
    timestamps = [file_start_time + pd.Timedelta(seconds=i / sampling_frequency) for i in
                  range(fo_strain_data.shape[0])]
    # Convert timestamps to datetime objects
    timestamps = [pd.Timestamp(ts) for ts in timestamps]

    return fo_strain_data, fo_strain_and_bandpass_data, timestamps
