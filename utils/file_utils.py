import os
import re
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from loguru import logger

from obspy.core.trace import Trace
from obspy.signal.trigger import plot_trigger, recursive_sta_lta, trigger_onset
from scipy.signal import butter, filtfilt, iirfilter, sosfilt, zpk2sos

# Old script to get the files in a directory
def get_files_in_dir(folder_path: str, file_format: str, keep_extension: bool = True):
    """
    Get a list of unique file names inside the given folder path that match the given file format.

    Args:
        folder_path (str): The path to the folder containing the files.
        file_format (str): The file format to filter by (e.g., ".tdms").
        keep_extension (bool): Whether to keep the file extension in the returned file names. Default is True.

    Returns:
        file_list (list): List of file names in the folder that match the given format.
    """
    # Validate the inputs
    if not os.path.isdir(folder_path):
        raise ValueError("The provided path is not a valid directory.")
    if not file_format.startswith('.'):
        raise ValueError("The file format should start with a dot (e.g., '.txt').")

    # Get the list of files in the folder with the specified extension
    file_list = [f for f in os.listdir(folder_path) if f.endswith(file_format)]

    # Remove the file extension if keep_extension is False
    if not keep_extension:
        file_list = [os.path.splitext(f)[0] for f in file_list]

    return file_list


def extract_timestamp_from_name(file_names: list):
    """
    Extract timestamps from the given list of file names. Works for:

    - .tdms files with format: iDAS_continous_measurements_30s_UTC_20201111_121152.869.tdms
      (Extracts date [20201111] and time [121152.869])

    - .h5 files with format: sensor_2024-09-07T060128Z.h5
      (Extracts timestamp from ISO format: [2024-09-07T060128Z])

    Args:
        file_names (list): List of file names.

    Returns:
        timestamps (list): List of extracted timestamps as datetime objects.
    """
    timestamps = []

    for name in file_names:
        # Match .tdms files
        tdms_match = re.search(r'UTC_(\d{8})_(\d{6}\.\d+)\.tdms$', name)
        if tdms_match:
            date_str = tdms_match.group(1)
            time_str = tdms_match.group(2)
            timestamp_str = f"{date_str}_{time_str}"
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S.%f')
            timestamps.append(timestamp)
            continue

        # Match .h5 files
        h5_match = re.search(r'sensor_(\d{4}-\d{2}-\d{2}T\d{6})Z\.h5$', name)
        if h5_match:
            timestamp_str = h5_match.group(1)
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H%M%S')
            timestamps.append(timestamp)

    return timestamps


def get_files_list(folder_path: str, file_extension: str = 'h5'):
    """
    Get a list of files in a directory with a specific extension.

    Args:
        folder_path (str): The path to the directory containing the files.
        file_extension (str): The file extension to search for in the directory (default is 'h5').

    Returns:
        files: A list of Paths to the files detected in the folder.
    """

    # Convert the folder path to a Path object
    folder_path = Path(folder_path)

    # Check if the provided path is a directory, if not return an empty and raise an error
    if not folder_path.is_dir():
        logger.error(f"Provided path is not a directory: {folder_path}")
        return []

    # Automatically add '*' if the user provides only the extension
    if not file_extension.startswith("*."):
        file_extension = f"*.{file_extension}"

    # Get a list of only the file names in the directory
    file_names = [file.name for file in folder_path.glob(file_extension)]
    # Get a list of the complete file paths
    file_paths = list(folder_path.glob(file_extension))


    # If no files are found, log a warning and return an empty list
    if not file_paths:
        logger.warning(f"No files found with extension '{file_extension}' in {folder_path}")
        return []

    # Log the number of files detected
    logger.info(f"Detected {len(file_names)} files in {folder_path}")

    return file_paths


def get_file_extensions(folder_path: str):
    """
    Get a list of all unique file extensions in the specified folder.

    Args:
        folder_path (str): The path to the directory to scan for file extensions.

    Returns:
        List[str]: A sorted list of unique file extensions (e.g., ['.txt', '.jpeg', '.pdf']).
    """

    folder_path = Path(folder_path)

    if not folder_path.is_dir():
        print(f"Provided path is not a directory: {folder_path}")
        return []

    # Collect all file extensions in the folder (excluding subdirectories)
    extensions = {file.suffix for file in folder_path.iterdir() if file.is_file()}

    # Convert set to a sorted list for readability
    return sorted(extensions)



def find_trains_STALTA(
    data: np.ndarray,
    inspect_channel: int,
    sf: int,
    batch: int,
    batch_length: int,
    lower_seconds: int = 1,
    upper_seconds: int = 10,
    upper_thres: float = 6,
    lower_thres: float = 0.5,
    minimum_trigger_period: float = 3.0,
) -> pd.DataFrame:
    """
    Detect trains in a single channel using the STA-LTA algorithm.

    Args:
        data (np.ndarray): FOAS data for a single channel
        inspect_channel(int): Single channel number
        sf (int): Sampling frequency
        batch (int): Batch number
        batch_length (int): Length of a batch
        lower_seconds (int, optional): Determines the size of the STA window. Defaults to 1.
        upper_seconds (int, optional): Determines the size of the LTA window. Defaults to 10.
        upper_thres (float, optional): Threshold for switching trigger on. Defaults to 4.5.
        lower_thres (float, optional): Threshold for switching trigger off. Defaults to 1.5.
        minimum_trigger_period (float, optional): The minimum period (in seconds) a trigger has to be included in the output. Defaults to 3.0

    Returns:
        df_trains (pd.DataFrame): DataFrame with the detected trains
    """

    # Preprocessing to clean up the signal.
    # Steps validated by Edwin from Deltares
    # Convert to trace
    # trace = Trace(data=data, header={"sampling_rate": sf})
    # trace.detrend("demean")  # Removing the mean
    # trace.taper(max_percentage=0.1, type="cosine")  # Applying a taper
    # trace.filter("bandpass", freqmin=1.0, freqmax=50, corners=4, zerophase=True)  # Bandpass filtering
    # singlechanneldata = trace.data  # Storing the data in a numpy array

    # High-pass filter the data
    singlechanneldata = highpass(data)[:-sf]

    # Run STA-LTA on the signal
    values = do_stalta(
        data=singlechanneldata,
        freq=sf,
        plots=False,  # Only True for local dev
        lower=lower_seconds,
        upper=upper_seconds,
        lower_thres_plot=lower_thres,
        upper_thres_plot=upper_thres,
    )

    # Find the triggers in the signal based on the thresholds provided
    triggers = trigger_onset(values, upper_thres, lower_thres)

    # If no triggers are found, return an empty DataFrame
    if len(triggers) == 0:
        return pd.DataFrame(columns=["start", "end", "channel"])
    # Calculate the offset for the batch
    offset = batch * batch_length  # TODO: We should not want to do this for very large runs
    df_trains = pd.DataFrame(triggers, columns=["start", "end"]).assign(batch=batch)
    df_trains = df_trains.loc[lambda d: d.end - d.start > minimum_trigger_period * sf]
    df_trains["start"] = df_trains["start"] + offset
    df_trains["end"] = df_trains["end"] + offset
    df_trains["channel"] = inspect_channel

    return df_trains



def do_stalta(
    data: Trace | np.ndarray,
    freq: float,
    upper_thres_plot: float = 4.5,
    lower_thres_plot: float = 1.5,
    plots: bool = True,
    lower: int = 1,
    upper: int = 10,
) -> np.ndarray:
    """
    Wrapper around the recursive STA-LTA algorithm to a trace or numpy array, and optionally plot the results.

    The size of the STA and LTA windows are determined by the lower and upper parameters, respectively.

    Args:
        data (Union[Trace, np.ndarray]): Input signal
        freq (float): _description_
        upper_thres (float, optional): Threshold for switching trigger on, only relevant for plotting. Defaults to 4.5.
        lower_thres (float, optional): Threshold for switching trigger off, only relevant for plotting. Defaults to 1.5.
        plots (bool, optional): If True, plot the results. Defaults to True.
        lower (int, optional): Determines the size of the STA window. Defaults to 1.
        upper (int, optional): Determines the size of the LTA window. . Defaults to 10.

    Returns:
        np.ndarray: STA-LTA values
    """
    trace = data if isinstance(data, Trace) else Trace(np.array(data))

    cft = recursive_sta_lta(trace.data, int(lower * freq), int(upper * freq))
    if plots:
        plot_trigger(trace, cft, upper_thres_plot, lower_thres_plot)
    return cft


def from_windows_get_fo_signal(fo_data_path: str, database: pd.DataFrame):
    """
    Get the FO signal data for the time windows.
    """

    # Get a list of all the file names in format .tdms in the folder
    file_names = get_files_in_dir(folder_path=fo_data_path, file_format='.h5')

    # Extract the timestamps from the file names
    fo_timestamps = extract_timestamp_from_name(file_names)

    # Loop through the rows in the dataframe and look for the timestamps with times that fit within each start and end time
    for index, row in database.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']

        # Initialize a list to hold the files that match the time window
        matching_files = []

        # Loop through the timestamps and find the ones that fit within the start and end time
        for i, timestamp in enumerate(fo_timestamps):
            # Ensure the timestamp is a pandas Timestamp object
            if isinstance(timestamp, datetime):
                timestamp = pd.Timestamp(timestamp)
            if start_time <= timestamp <= end_time:
                # Get the file name
                file_name = file_names[i]
                # Append the file name to the list of matching files
                matching_files.append(file_name)

        # Join the list of matching files into a single string separated by commas (or store as a list if preferred)
        database.at[index, 'files'] = ', '.join(matching_files)

    return database


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


def calculate_sampling_frequency(file: h5py.File) -> float:
    """
    Calculate the sampling frequency from an open HDF5 file by measuring the time interval
    between consecutive samples in the 'RawDataTime' dataset. With the time interval, the
    sampling frequency is calculated as the inverse of the time interval.

    Args:
        file (h5py.File): An open HDF5 file object.

    Returns:
        float: The calculated sampling frequency in Hz.
    """
    try:
        # Access the 'RawDataTime' dataset to get timestamps for calculating sampling interval
        raw_data_time = file['Acquisition']['Raw[0]']['RawDataTime']

        # Calculate time interval between the first two samples in seconds
        time_interval = (raw_data_time[1] - raw_data_time[0]) * 1e-6  # Convert from microseconds to seconds

        # Sampling frequency is the inverse of the time interval
        sampling_frequency = 1 / time_interval
        # Make sampling frequency an integer
        sampling_frequency = int(sampling_frequency)

        return sampling_frequency

    except KeyError:
        raise ValueError("The 'RawDataTime' dataset is missing in the file structure.")
    except IndexError:
        raise ValueError("The 'RawDataTime' dataset has insufficient data for frequency calculation.")




def highpass(data: np.ndarray, cutoff: float = 0.1) -> np.ndarray:
    b, a = butter(1, cutoff, btype="high", analog=False)
    return filtfilt(b, a, data)


def bandpass(data, freqmin, freqmax, fs, corners, zerophase=True):
    fe = 0.5 * fs
    low = freqmin / fe
    high = freqmax / fe
    z, p, k = iirfilter(corners, [low, high], btype='band', ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)

    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)

