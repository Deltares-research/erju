import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

from src.erju.process_FO_base import BaseFOdata
from src.utils.file_utils import extract_timestamp_from_name, get_files_in_dir


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
    if not (
        isinstance(time_window, list)
        and len(time_window) == 2
        and all(isinstance(t, datetime) for t in time_window)
    ):
        raise ValueError(
            "time_window must be a list of two datetime objects: [start_time, end_time]"
        )

    event_start, event_end = time_window

    # Get a list of all .h5 file names sorted by name (timestamp is embedded in name).
    file_names = sorted(get_files_in_dir(folder_path=fo_data_path, file_format=".h5"))
    if len(file_names) == 0:
        return []

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
    matching_file_paths = [
        os.path.join(fo_data_path, file_names[i]) for i in sorted(extended_indices)
    ]

    return matching_file_paths


def extract_fo_event_data(
    fo_data_path: str,
    time_window: list,
    center_channel: int,
    channel_half_window: int,
    reader: str = "optasense",
):
    """Extract FO strain for an event window using the same logic as compare_data.py.

    Returns a dictionary with:
      - found (bool)
      - reason (str)
      - file_paths (list[str])
      - timestamps (list[datetime])
      - fs_hz (float)
      - strain (np.ndarray: [fo_time, fo_channel])
      - channel_ids (np.ndarray: absolute channel numbers)
      - center_channel / channel_half_window
      - optional metadata (gauge_length_m, channel_spacing_m)
    """
    first_channel = int(center_channel) - int(channel_half_window)
    last_channel = int(center_channel) + int(channel_half_window)
    if first_channel < 0:
        return {
            "found": False,
            "reason": "invalid_channel_window",
            "file_paths": [],
        }

    fo_files_in_event = from_window_get_fo_file(fo_data_path, time_window)
    if len(fo_files_in_event) == 0:
        return {
            "found": False,
            "reason": "no_matching_fo_files",
            "file_paths": [],
        }

    fo = BaseFOdata.create_instance(
        dir_path=fo_data_path,
        first_channel=first_channel,
        last_channel=last_channel,
        reader=reader,
    )

    strain_chunks = []
    sampling_frequency = None
    file_start_time = None
    gauge_length_m = None
    channel_spacing_m = None

    for i, fo_file in enumerate(fo_files_in_event):
        if i == 0:
            fo.extract_properties_per_file(fo_file)
            file_start_time = fo.properties.get("FileStartTime")
            sampling_frequency = float(fo.properties.get("SamplingFrequency[Hz]"))
            gauge_length_m = fo.properties.get("GaugeLength")
            channel_spacing_m = fo.properties.get("SpatialSamplingInterval")

        # extract_data returns (filtered_to_strain, strain). We store raw strain only.
        _, raw_strain = fo.extract_data(
            file_name=fo_file,
            first_channel=first_channel,
            last_channel=last_channel,
        )
        strain_chunks.append(raw_strain.T)

    if len(strain_chunks) == 0:
        return {
            "found": False,
            "reason": "no_fo_data_extracted",
            "file_paths": fo_files_in_event,
        }

    strain_all = np.concatenate(strain_chunks, axis=0)

    # Build timestamps exactly as compare_data.py does.
    timestamps = [
        file_start_time + timedelta(seconds=i / sampling_frequency)
        for i in range(strain_all.shape[0])
    ]

    timestamps_array = np.array(timestamps, dtype="datetime64[ns]")
    start_time_np = np.datetime64(time_window[0])
    end_time_np = np.datetime64(time_window[1])
    start_index = int(np.argmin(np.abs(timestamps_array - start_time_np)))
    end_index = int(np.argmin(np.abs(timestamps_array - end_time_np)))

    if end_index < start_index:
        start_index, end_index = end_index, start_index

    timestamps_cropped = timestamps[start_index : end_index + 1]
    strain_cropped = strain_all[start_index : end_index + 1, :]

    if len(timestamps_cropped) == 0 or strain_cropped.size == 0:
        return {
            "found": False,
            "reason": "empty_fo_after_crop",
            "file_paths": fo_files_in_event,
        }

    channel_ids = np.arange(first_channel, last_channel + 1, dtype=np.int32)

    return {
        "found": True,
        "reason": "ok",
        "fo_reader": reader,
        "file_paths": fo_files_in_event,
        "timestamps": timestamps_cropped,
        "fs_hz": np.float32(sampling_frequency),
        "strain": np.asarray(strain_cropped, dtype=np.float32),
        "channel_ids": channel_ids,
        "center_channel": int(center_channel),
        "channel_half_window": int(channel_half_window),
        "gauge_length_m": gauge_length_m,
        "channel_spacing_m": channel_spacing_m,
    }
