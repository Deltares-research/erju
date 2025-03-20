import os
import h5py
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from loguru import logger
from datetime import datetime, timedelta
import netCDF4 as nc

from numpy.core.defchararray import upper
from scipy.signal import butter, filtfilt, iirfilter, sosfilt, zpk2sos
from src.utils.file_utils import get_files_list
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


def do_stalta(data: Trace | np.ndarray,
              freq: float,
              upper_thresh: float,
              lower_thresh: float,
              plots: bool = True,
              lower: int = 1,
              upper: int = 10,) -> np.ndarray:
    """
    Wrapper around the recursive STA-LTA algorithm to a trace or numpy array, and optionally plot the results.
    The size of the STA and LTA windows are determined by the lower and upper parameters, respectively.

    Args:
        data (Union[Trace, np.ndarray]): Input signal
        freq (float): Sampling frequency
        upper_thresh (float, optional): Threshold for switching trigger on, only relevant for plotting. Defaults to 4.5.
        lower_thresh (float, optional): Threshold for switching trigger off, only relevant for plotting. Defaults to 1.5.
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
        plot_trigger(trace, cft, upper_thresh, lower_thresh)
    return cft


def find_trains_STALTA(
    data: np.ndarray,
    timestamps: list,
    inspect_channel: int,
    sf: int,
    batch: int,
    batch_length: int,
    file_start_time: float,
    upper_thresh: float,
    lower_thresh: float,
    window_extension: int = 10,
    lower_seconds: int = 1,
    upper_seconds: int = 10,
    output_dir: str = "output",
    minimum_trigger_period: float = 3.0,
) -> pd.DataFrame:
    """Detect trains in a single channel using the STA-LTA algorithm and return the start and end indices of the events.
    It also creates a NetCDF file for each detected event.

    Args:
        data (np.ndarray): FOAS data for a single channel
        timestamps (list): List of timestamps for each data point
        inspect_channel(int): Single channel number
        sf (int): Sampling frequency
        batch (int): Batch number
        batch_length (int): Length of a batch
        file_start_time (float): Start time of the file
        window_extension (int, optional): The time in seconds to extend the event window. Defaults to 10.
        lower_seconds (int, optional): Determines the size of the STA window. Defaults to 1.
        upper_seconds (int, optional): Determines the size of the LTA window. Defaults to 10.
        upper_thresh (float, optional): Threshold for switching trigger on. Defaults to 4.5.
        lower_thresh (float, optional): Threshold for switching trigger off. Defaults to 1.5.
        minimum_trigger_period (float, optional):
            The minimum period (in seconds) a trigger has to be to be included in the output. Defaults to 3.0

    Returns:
        windows_indices (list): List of tuples with the start and end indices of the detected events
        windows_times (list): List of tuples with the start and end times of the detected events
        values (np.ndarray): STA-LTA ratio values
    """
    # Run STA-LTA on the signal
    values = do_stalta(
        data=data,
        freq=sf/2, # It works better at finding events with sf/2 instead of the actual sf that I initially used
        plots=False,  # Only True for local dev
        lower=lower_seconds,
        upper=upper_seconds,
        lower_thresh=lower_thresh,
        upper_thresh=upper_thresh,
    )

    # Find the events where the STA-LTA value exceeds the thresholds and return the start and end indices
    events = trigger_onset(values, upper_thresh, lower_thresh)

    # No events detected
    if len(events) == 0:
        # Return empty lists for windows and events
        return [], [], values

    # Create the windows around the detected events
    windows_indices = []
    windows_times = []

    # For each event, create a windows around it
    for event in events:
        # Extend the window by a buffer at the start and the end
        start_index = max(0, event[0] - window_extension * sf)
        end_index = min(len(data) -1, event[1] + window_extension * sf)
        # Get the start and end time of the window
        start_time = timestamps[start_index]
        end_time = timestamps[end_index]


        create_netcdf_file(data=data,
                           timestamps=timestamps,
                           start_time=start_time,
                           end_time=end_time,
                           sampling_frequency=sf,
                           output_dir=output_dir)


        # Append the window to the lists
        windows_indices.append((start_index, end_index))
        windows_times.append((start_time, end_time))

        # For each event, create a netcdf file with the data

    return windows_indices, windows_times, values



def plot_signals_and_stalta(raw_signal, filtered_signal, timestamps, stalta_ratio, window_times, trigger_on, trigger_off,
                            output_dir="output", save_plot=False):
    """
    Plots the FO signal and STA/LTA ratio with detected events highlighted and saves the plot.

    Parameters:
        raw_signal (array-like): List or array containing the raw signal data.
        filtered_signal (array-like): List or array containing the filtered signal data.
        timestamps (array-like): List or array containing corresponding timestamps for the signal.
        stalta_ratio (array-like): Array or list of STA/LTA ratio values.
        window_times (list): List of tuples indicating the start and end times of detected events.
        trigger_on (float): The value for the 'on' trigger line.
        trigger_off (float): The value for the 'off' trigger line.
        output_dir (str): Directory to save the plot.
        save_plot (bool): If True, save the plot to the output directory.
    """
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))

    # Plot the raw_signal
    ax[0].plot(timestamps, raw_signal, color='black')
    ax[0].set_title(f'Raw batch signal')
    ax[0].set_xlabel('Time [HH:MM:SS]')
    ax[0].set_ylabel('Amplitude')
    ax[0].grid(True)

    # Plot the extended signal
    ax[1].plot(timestamps, filtered_signal, color='blue')
    # Add shaded areas for detected events
    for window in window_times:
        ax[1].axvspan(window[0], window[1], color='gray', alpha=0.5)
    ax[1].set_title('Filtered batch signal data and events')
    ax[1].set_xlabel('Time [HH:MM:SS]')
    ax[1].set_ylabel('Amplitude')
    ax[1].grid(True)

    # Plot the STA/LTA ratio
    ax[2].plot(timestamps, stalta_ratio, color='red')
    # Add horizontal lines for trigger values
    ax[2].axhline(y=trigger_on, color='green', linestyle='--', label='Trigger On')
    ax[2].axhline(y=trigger_off, color='red', linestyle='--', label='Trigger Off')
    ax[2].set_title('STA/LTA Ratio')
    ax[2].set_xlabel('Time [HH:MM:SS]')
    ax[2].set_ylabel('Ratio')
    ax[2].grid(True)

    # Format the x-axis to show the time
    date_form = DateFormatter("%H:%M:%S")
    ax[0].xaxis.set_major_formatter(date_form)
    ax[1].xaxis.set_major_formatter(date_form)
    ax[2].xaxis.set_major_formatter(date_form)

    ax[2].legend()  # Show legend for the trigger lines

    plt.tight_layout()
    #plt.show()

    # If save_plot is True, save the plot to the output directory
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"batch_signal_{timestamps[0].strftime('%Y%m%dT%H%M%S')}.png"
        file_path = os.path.join(output_dir, file_name)
        plt.savefig(file_path)

    plt.close()


def create_netcdf_file(data, timestamps, start_time, end_time, sampling_frequency, output_dir="output"):
    """
    Create and save a NetCDF file for a detected event.

    Args:
        data (np.ndarray): Signal data for the event.
        timestamps (list): Timestamps for the event.
        start_time (datetime): Start time of the event.
        end_time (datetime): End time of the event.
        sampling_frequency (int): Sampling frequency of the signal.
        output_dir (str): Directory to save the NetCDF files.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"event_{start_time.strftime('%Y%m%dT%H%M%S')}.nc"
    file_path = os.path.join(output_dir, file_name)

    # Extract the timestamps for only the event
    start_idx = timestamps.index(start_time)
    end_idx = timestamps.index(end_time)
    event_timestamps = timestamps[start_idx:end_idx + 1]  # Get timestamps for the event

    # Create NetCDF file
    with nc.Dataset(file_path, 'w', format='NETCDF4') as ncfile:
        # Create dimensions
        ncfile.createDimension('time', len(event_timestamps))

        # Create variables
        time_var = ncfile.createVariable('time', 'f8', ('time',))
        data_var = ncfile.createVariable('signal', 'f4', ('time',))

        # Assign data
        time_var[:] = np.array([(ts - start_time).total_seconds() for ts in event_timestamps])
        data_var[:] = data[start_idx:end_idx + 1]  # Extract corresponding data

        # Add metadata
        ncfile.description = f"Event data from {start_time} to {end_time}"
        ncfile.sampling_frequency = sampling_frequency
        ncfile.start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
        ncfile.end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')

    print(f"Saved event data to {file_path}")

########################################################################################################################


# From a given folder path, get all the files with a given extension
path_to_files = Path(r'C:\fo_samples\holten')
output_dir = r'N:\Projects\11210000\11210064\B. Measurements and calculations\holten\fo_plot'

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
center_channel = 1194
channel_range = 5
# The user defines the thresholds and window extension for the STA/LTA algorithm
upper_thresh = 3
lower_thresh = 0.5
window_extension = 10

# Calculate the relative position of the center channel in a new list of a given number of channels according to the channel range
relative_center_channel = channel_range * 2 // 2

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
                # Calculate the sampling frequency
                sampling_frequency = calculate_sampling_frequency(file)

            # Append the data to the list
            batch_raw_data.append(file_raw_data)
            length_batch_raw_data = len(batch_raw_data)

    # Concatenate the data from the batch. Here are all the channels from the combined batch files
    # Concatenate data and calculate time
    raw_data = np.concatenate(batch_raw_data, axis=0)

    # Calculate the time for each data point in the signal starting from the file start time as a datetime object
    # Assuming file_start_time is a datetime object and raw_data.shape[0] is the number of samples
    timestamps = [file_start_time + timedelta(seconds=i / sampling_frequency) for i in range(raw_data.shape[0])]

    # Create arrays to hold filtered data with the same shape as raw_data
    raw_data_highpass = np.empty_like(raw_data)
    raw_data_bandpass = np.empty_like(raw_data)

    # Iterate over each channel (column)
    #TODO: Here you have 2 filters, choose which one you are going to use (discuss w/Bruno)
    for channel in range(raw_data.shape[1]):
        # Apply the high pass filter to the current channel
        raw_data_highpass[:, channel] = highpass(data=raw_data[:, channel], cutoff=0.1)

        # Apply the bandpass filter to the high-pass-filtered data of the current channel
        raw_data_bandpass[:, channel] = bandpass(data=raw_data_highpass[:, channel],
                                                 freqmin=0.1, freqmax=100, fs=1000, corners=4)


    # Apply the STA/LTA algorithm to the filtered data
    # HERE I choose to find the events in just the center channel
    try:
        # First lets calculate the STA and LTA window sizes
        signal_seconds = raw_data_highpass.shape[0] / sampling_frequency
        LTA_window_size = min(signal_seconds / 2, 50)
        LTA_window_size = max(LTA_window_size, 10)
        STA_window_size = LTA_window_size // 10

        # Find the events in the center channel using the STA/LTA method#
        window_indices, window_times, values = find_trains_STALTA(data=raw_data_highpass[:, relative_center_channel],
                                                                  timestamps=timestamps,
                                                                  inspect_channel=relative_center_channel,
                                                                  sf=sampling_frequency,
                                                                  batch=batch_number,
                                                                  batch_length=num_measurements * batchsize,
                                                                  file_start_time=file_start_time,
                                                                  upper_thresh=upper_thresh,
                                                                  lower_thresh=lower_thresh,
                                                                  window_extension=window_extension,
                                                                  lower_seconds=STA_window_size,
                                                                  upper_seconds=LTA_window_size,
                                                                  output_dir=output_dir,)

        # Print the number of events detected in the batch
        if not window_indices:
            logger.info(f"No events detected in batch {batch_number + 1}")
        if len(window_indices) > 0:
            logger.info(f"Detected {len(window_indices)} events in batch {batch_number + 1}")

        # Plot the signal and STA/LTA ratio with detected events using the STA/LTA method
        plot_signals_and_stalta(raw_signal=raw_data[:, relative_center_channel],
                                filtered_signal=raw_data_highpass[:, relative_center_channel],
                                timestamps=timestamps,
                                stalta_ratio=values,
                                window_times=window_times,
                                trigger_on=upper_thresh,
                                trigger_off=lower_thresh,
                                output_dir=output_dir,
                                save_plot=True)

    # Handle exceptions
    except ValueError as e:
        logger.error(f"An error occurred while processing batch {batch_number + 1}: {e}")



