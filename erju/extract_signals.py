import h5py
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from datetime import datetime
from scipy.signal import butter, filtfilt, iirfilter, sosfilt, zpk2sos
from utils.file_utils import get_files_list
from obspy.core.trace import Trace
from obspy.signal.trigger import plot_trigger, recursive_sta_lta, trigger_onset


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


def data_consistency_check(file_path_list: list) -> bool:
    """
    Check the consistency of the data in the files by doing some simple comparisons and calculations.

    Args:
        file_path_list (list): A list of the paths to the files to be checked.

    Returns:
        bool: True if the data is consistent, False otherwise.
    """
    # Access the metadata of the first and list files
    with h5py.File(file_path_list[0], 'r') as file:
        first_file_sf = calculate_sampling_frequency(file)
        first_file_data_shape = file['Acquisition']['Raw[0]']['RawData'].shape

    with h5py.File(file_path_list[-1], 'r') as file:
        last_file_sf = calculate_sampling_frequency(file)
        last_file_data_shape = file['Acquisition']['Raw[0]']['RawData'].shape


    # Check if the sampling frequency and the shape is the same, if not raise an error for each specific case and if yes, return True and a log message that data is consistent.
    if first_file_sf != last_file_sf:
        raise ValueError(f"Sampling frequency mismatch: {first_file_sf} Hz (first file) vs. {last_file_sf} Hz (last file)")
    elif first_file_data_shape != last_file_data_shape:
        raise ValueError(f"Data shape mismatch: {first_file_data_shape} (first file) vs. {last_file_data_shape} (last file)")
    else:
        logger.info("Data is consistent.")
        return True


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


def do_stalta(data: Trace | np.ndarray, freq: float, upper_thres: float = 4.5, lower_thres: float = 1.5,
              plots: bool = True, lower: int = 1, upper: int = 10,) -> np.ndarray:
    """
    Wrapper around the recursive STA-LTA algorithm to a trace or numpy array, and optionally plot the results.
    The size of the STA and LTA windows are determined by the lower and upper parameters, respectively.

    Args:
        data (Union[Trace, np.ndarray]): Input signal
        freq (float): Sampling frequency
        upper_thres (float, optional): Threshold for switching trigger on, only relevant for plotting. Defaults to 4.5.
        lower_thres (float, optional): Threshold for switching trigger off, only relevant for plotting. Defaults to 1.5.
        plots (bool, optional): If True, plot the results. Defaults to True.
        lower (int, optional): Determines the size of the STA window. Defaults to 1.
        upper (int, optional): Determines the size of the LTA window. . Defaults to 10.

    Returns:
        np.ndarray: STA-LTA values
    """
    # Convert input to Trace if it is not already
    trace = data if isinstance(data, Trace) else Trace(np.array(data))
    # Run the recursive STA-LTA algorithm
    cft = recursive_sta_lta(a=trace.data, nsta=int(lower * freq), nlta=int(upper * freq))
    # Plot the results if requested
    if plots:
        # Plot the STA-LTA values
        plot_trigger(trace, cft, upper_thres, lower_thres)
    return cft


def find_trains_STALTA(
    data: np.ndarray,
    inspect_channel: int,
    sf: int,
    batch: int,
    batch_length: int,
    file_start_time: float,
    window_extension: int = 10,
    lower_seconds: int = 1,
    upper_seconds: int = 10,
    upper_thres: float = 6,
    lower_thres: float = 0.5,
    minimum_trigger_period: float = 3.0,
) -> pd.DataFrame:
    """Detect trains in a single channel using the STA-LTA algorithm.

    Args:
        data (np.ndarray): FOAS data for a single channel
        inspect_channel(int): Single channel number
        sf (int): Sampling frequency
        batch (int): Batch number
        batch_length (int): Length of a batch
        file_start_time (float): Start time of the file
        window_extension (int, optional): The time in seconds to extend the event window. Defaults to 10.
        lower_seconds (int, optional): Determines the size of the STA window. Defaults to 1.
        upper_seconds (int, optional): Determines the size of the LTA window. Defaults to 10.
        upper_thres (float, optional): Threshold for switching trigger on. Defaults to 4.5.
        lower_thres (float, optional): Threshold for switching trigger off. Defaults to 1.5.
        minimum_trigger_period (float, optional):
            The minimum period (in seconds) a trigger has to be to be included in the output. Defaults to 3.0

    Returns:
        pd.DataFrame: DataFrame with start and end times and the channel number of the detected trains
    """
    # Run STA-LTA on the signal
    values = do_stalta(
        data=data,
        freq=sf, #TODO: this used to be sf/2 in the original, why?
        plots=False,  # Only True for local dev
        lower=lower_seconds,
        upper=upper_seconds,
        lower_thres=lower_thres,
        upper_thres=upper_thres,
    )

    # Find the events where the STA-LTA value exceeds the thresholds and return the start and end indices
    events = trigger_onset(values, upper_thres, lower_thres)

    # Create the windows around the detected events
    windows_indices = []
    windows_times = []

    # For each event, create a windows around it
    for event in events:
        # Extend the window by a buffer at the start and the end
        start_index = max(0, event - window_extension * sf)
        end_index = min(len(data) -1, event[1] + window_extension * sf)
        # Compute the corresponding time for the start and end indices
        start_time = file_start_time + start_index / sf

        if len(events) == 0:  # Only continue if there are events
            return pd.DataFrame(columns=["start", "end", "channel"])
        offset = batch * batch_length  # TODO: We should not want to do this for very large runs
        df_trains = pd.DataFrame(events, columns=["start", "end"]).assign(batch=batch)
        df_trains = df_trains.loc[lambda d: d.end - d.start > minimum_trigger_period * sf]
        df_trains["start"] = df_trains["start"] + offset
        df_trains["end"] = df_trains["end"] + offset
        df_trains["channel"] = inspect_channel
        df_trains["start_time"] = file_start_time

        return df_trains



# From a given folder path, get all the files with a given extension
path_to_files = Path(r'C:\Projects\erju\data\holten\recording_2024-08-29T08_01_16Z_5kHzping_1kHzlog_1mCS_10mGL_3000channels')

# Get the list of files
file_paths = get_files_list(folder_path=path_to_files, file_extension='h5')

# Check the consistency of the files.
data_consistency_check(file_path_list=file_paths)

# Generate batches with overlapping logic ([f1, f2], [f2, f3], [f3, f4], ...)
# This is done to avoid missing signals in between files
file_batches = []
batchsize = 2
for i in range(0, len(file_paths), batchsize - 1):  # Ensure overlap of one file
    batch = file_paths[i:i + batchsize]
    file_batches.append(batch)

# The user defines the center channel and the number of channels around it
center_channel = 1200
channel_range = 5

# Calculate the new start channel and end channel
start_channel = max(0, center_channel - channel_range)  # Ensure start channel is not negative
end_channel = center_channel + channel_range

# Iterate over the batches
for batch_number, batch in enumerate(file_batches):
    logger.info(f"Processing batch {batch_number + 1}/{len(file_batches)}")
    # Create an empty list to store the data for the batch
    batch_raw_data = []

    # Read the data from the files in the batch
    for file_path in batch:
        # Open the file
        with h5py.File(file_path, 'r') as file:
            # Get the RawData dataset
            file_raw_data = file['Acquisition']['Raw[0]']['RawData'][:, start_channel:end_channel + 1]
            # If it is the first file in the batch, read the metadata
            if file_path == batch[0]:
                # Get the RawDataTime dataset
                raw_data_time = file['Acquisition']['Raw[0]']['RawDataTime']
                # Get the first timestamp and convert it to a datetime object
                file_start_time = datetime.utcfromtimestamp(raw_data_time[0] * 1e-6)
                num_measurements = file['Acquisition']['Raw[0]']['RawData'].shape[0]

            # Append the data to the list
            batch_raw_data.append(file_raw_data)
            length_batch_raw_data = len(batch_raw_data)

    # Concatenate the data from the batch. Here are all the channels from the combined batch files
    # Concatenate data and calculate time
    raw_data = np.concatenate(batch_raw_data, axis=0)
    time = np.array([
        file_start_time + pd.Timedelta(microseconds=int(raw_data_time[i]))
        for i in range(len(raw_data_time))
    ])
    print('Elements inside raw_data: ', len(raw_data))

    # Create arrays to hold filtered data with the same shape as raw_data
    raw_data_highpass = np.empty_like(raw_data)
    raw_data_bandpass = np.empty_like(raw_data)

    # Iterate over each channel (column)
    for channel in range(raw_data.shape[1]):
        # Apply the high pass filter to the current channel
        raw_data_highpass[:, channel] = highpass(data=raw_data[:, channel], cutoff=0.1)

        # Apply the bandpass filter to the high-pass-filtered data of the current channel
        raw_data_bandpass[:, channel] = bandpass(data=raw_data_highpass[:, channel],
                                                 freqmin=0.1, freqmax=100, fs=1000, corners=4)

    # Apply the STA/LTA algorithm to the filtered data
    try:
        print('Elements inside raw_data_bandpass: ', len(raw_data_bandpass))



    except ValueError as e:
        logger.error(f"An error occurred while processing batch {batch_number + 1}: {e}")






    # Plot one raw and filtered channel for comparison
    channel_to_plot = 5  # Adjust this to visualize different channels
    fig, ax0 = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

    ax0[0].plot(raw_data[:, channel_to_plot], color='black', label='Raw data')
    ax0[0].set_title('Raw data')
    ax0[0].set_ylabel('Amplitude')
    ax0[0].grid(True)

    ax0[1].plot(raw_data_bandpass[:, channel_to_plot], color='blue', label='Filtered data')
    ax0[1].set_title('Filtered data bandpass 0.1 - 100 Hz')
    ax0[1].set_ylabel('Amplitude')
    ax0[1].grid(True)

    ax0[2].plot(raw_data_highpass[:, channel_to_plot], color='red', label='Filtered data')
    ax0[2].set_title('Filtered data highpass 0.1 Hz')
    ax0[2].set_ylabel('Amplitude')
    ax0[2].grid(True)

    plt.show()
    plt.close()



save_path = r'C:\Projects\erju\outputs\holten'