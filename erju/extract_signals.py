import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from loguru import logger
from datetime import datetime
from scipy.signal import butter, filtfilt, iirfilter, sosfilt, zpk2sos
from utils.file_utils import get_files_list


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
    raw_data = np.concatenate(batch_raw_data, axis=0)
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