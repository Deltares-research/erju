import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from loguru import logger

from obspy.core.trace import Trace
from obspy.signal.trigger import plot_trigger, recursive_sta_lta, trigger_onset
from scipy.signal import butter, filtfilt

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
    Extract timestamps from the given list of file names. This works for .tdms files which have
    the following format [iDAS_continous_measurements_30s_UTC_20201111_121152.869.tdms]. Here the
    last part of the file name is the timestamp, as in this case [121152.869], and the date is
    after the UTC part [20201111]. This will need to be adapted if the file names have a different
    format.

    Args:
        file_names (list): List of file names.

    Returns:
        timestamps (list): List of extracted timestamps.
    """
    # Initialize the list to store the timestamps
    timestamps = []
    # Iterate over the file names
    for name in file_names:
        # Use regular expression to extract the date and timestamp from the file name
        # The pattern matches 'UTC_' followed by 8 digits for the date and then the time with 6 digits, a dot, and fractional seconds
        match = re.search(r'UTC_(\d{8})_(\d{6}\.\d+)\.tdms$', name)
        # If a match is found, extract the date and timestamp
        if match:
            # Extract the date string
            date_str = match.group(1)
            # Extract the timestamp string
            time_str = match.group(2)
            # Combine the date and timestamp stringsl
            timestamp_str = f"{date_str}_{time_str}"
            # Convert the combined string to a datetime object
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S.%f')
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


def highpass(data: np.ndarray, cutoff: float = 0.1) -> np.ndarray:
    """
    Apply a high-pass filter to the data. The filter is a first-order Butterworth filter with a cutoff frequency.

    Args:
        data (np.ndarray): The input data to filter.
        cutoff (float): The cutoff frequency for the high-pass filter.

    Returns:
        np.ndarray: The filtered data
    """
    b, a = butter(1, cutoff, btype="high", analog=False)
    processed_data = filtfilt(b, a, data)

    return processed_data


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
        singlechanneldata,
        sf / 2,
        plots=False,  # Only True for local dev
        lower=lower_seconds,
        upper=upper_seconds,
        lower_thres=lower_thres,
        upper_thres=upper_thres,
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
    upper_thres: float = 4.5,
    lower_thres: float = 1.5,
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
        plot_trigger(trace, cft, upper_thres, lower_thres)
    return cft