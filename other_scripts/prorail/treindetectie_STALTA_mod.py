from pathlib import Path

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from loguru import logger
from obspy.core.trace import Trace
from obspy.signal.trigger import plot_trigger, recursive_sta_lta, trigger_onset
from scipy.signal import butter, filtfilt


def calculate_sampling_frequency(file: h5py.File) -> float:
    """
    Calculate the sampling frequency from an open HDF5 file by measuring the time interval
    between consecutive samples in the 'RawDataTime' dataset.

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


def do_stalta(
    data: Trace | np.ndarray,
    freq: float,
    upper_thres: float = 4.5,
    lower_thres: float = 1.5,
    plots: bool = True,
    lower: int = 1,
    upper: int = 10,
) -> np.ndarray:
    """Wrapper around the recursive STA-LTA algorithm to a trace or numpy array, and optionally plot the results.

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

    cft = recursive_sta_lta(a=trace.data, nsta=int(lower * freq), nlta=int(upper * freq))
    if plots:
        plot_trigger(trace, cft, upper_thres, lower_thres)
    return cft


def find_trains_STALTA(
    data: np.ndarray,
    inspect_channel: int,
    sf: int,
    batch: int,
    batch_length: int,
    file_start_time: float,
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
        lower_seconds (int, optional): Determines the size of the STA window. Defaults to 1.
        upper_seconds (int, optional): Determines the size of the LTA window. Defaults to 10.
        upper_thres (float, optional): Threshold for switching trigger on. Defaults to 4.5.
        lower_thres (float, optional): Threshold for switching trigger off. Defaults to 1.5.
        minimum_trigger_period (float, optional):
            The minimum period (in seconds) a trigger has to be to be included in the output. Defaults to 3.0

    Returns:
        pd.DataFrame: DataFrame with start and end times and the channel number of the detected trains
    """
    # Preprocessing to clean up the signal.
    # Steps validated by Edwin from Deltares
    # Convert to trace
    # trace = Trace(data=data, header={"sampling_rate": sf})
    # trace.detrend("demean")  # Removing the mean
    # trace.taper(max_percentage=0.1, type="cosine")  # Applying a taper
    # trace.filter("bandpass", freqmin=1.0, freqmax=50, corners=4, zerophase=True)  # Bandpass filtering
    # singlechanneldata = trace.data  # Storing the data in a numpy array

    #singlechanneldata = highpass(data)[:-sf]
    singlechanneldata = data

    # Run STA-LTA on the signal
    values = do_stalta(
        data=singlechanneldata,
        freq=sf / 2, # Why freq is sf/2?
        plots=False,  # Only True for local dev
        lower=lower_seconds,
        upper=upper_seconds,
        lower_thres=lower_thres,
        upper_thres=upper_thres,
    )

    # Find triggers in the signal and the reversed signal
    triggers = trigger_onset(values, upper_thres, lower_thres)

    if len(triggers) == 0:  # Only continue if there are triggers
        return pd.DataFrame(columns=["start", "end", "channel"])
    offset = batch * batch_length  # TODO: We should not want to do this for very large runs
    df_trains = pd.DataFrame(triggers, columns=["start", "end"]).assign(batch=batch)
    df_trains = df_trains.loc[lambda d: d.end - d.start > minimum_trigger_period * sf]
    df_trains["start"] = df_trains["start"] + offset
    df_trains["end"] = df_trains["end"] + offset
    df_trains["channel"] = inspect_channel
    df_trains["start_time"] = file_start_time

    return df_trains

def detect_trainpassages_in_folder(
        filenames: list[Path],
        inspect_channel: int = None,  # Optional parameter for a specific channel
        batchsize: int = 2,
        stalta_lower_thres: float = 0.5,
        stalta_upper_thres: float = 6,
        plot_signal: bool = False,  # Option to plot and save signals
        save_path: str = None       # Folder path to save the plots
) -> pd.DataFrame:
    """Detects all the h5py files with train passages in a folder using the STA-LTA algorithm."""

    # Get metadata from the first file
    with h5py.File(filenames[0], "r") as file_start, h5py.File(filenames[-1], "r") as file_end:
        sf = calculate_sampling_frequency(file_start)
        if sf != calculate_sampling_frequency(file_end):
            raise ValueError("Sampling frequency is not the same in the begin and end files")

        data_shape = file_start["Acquisition"]["Raw[0]"]["RawData"].shape
        if data_shape != file_end["Acquisition"]["Raw[0]"]["RawData"].shape:
            raise ValueError("Data shape is not the same in the begin and end files")

        filelength = data_shape[0]
        n_channels = data_shape[1]

        if inspect_channel is None:
            inspect_channel = n_channels // 2

    # Initialize list to store results
    dfs = []

    # Generate batches with overlapping logic
    file_batches = []
    for i in range(0, len(filenames), batchsize - 1):  # Ensure overlap of one file
        batch = filenames[i:i + batchsize]
        file_batches.append(batch)

    # Iterate over the batches
    file_start_times = []  # Initialize an empty list to collect start times
    for batch_number, batch in enumerate(file_batches):
        logger.info(f"Reading files in batch {batch_number}/{len(file_batches)}; file names: {batch}")
        signal_batch_data = []

        # Read the data for the inspect channel file by file
        for file_path in batch:
            with h5py.File(file_path, "r") as file:
                data = file["Acquisition"]["Raw[0]"]["RawData"][:, inspect_channel]
                # Get metadata just for the first file
                if len(signal_batch_data) == 0:
                    # Get the "RawDataTime" dataset
                    rawDataTime = file['Acquisition']['Raw[0]']['RawDataTime']
                    # Get the first and last entry in "RawDataTime"
                    # Assuming rawDataTime contains timestamps in microseconds
                    file_start_time = datetime.utcfromtimestamp(
                        rawDataTime[0] * 1e-6)  # Convert to seconds, then to datetime
                    file_end_time = datetime.utcfromtimestamp(
                        rawDataTime[-1] * 1e-6)  # Convert to seconds, then to datetime
                    time_interval = (rawDataTime[1] - rawDataTime[0]) * 1e-6  # Convert to seconds

                # Append the data to the list
                signal_batch_data.append(data)


        # Concatenate the data
        signal_batch_data = np.concatenate(signal_batch_data, axis=0)

        # Apply high-pass filter to the concatenated data
        signal_batch_data = highpass(signal_batch_data, cutoff=0.1)

        # Process the data
        try:
            single_signal_concat = signal_batch_data
            signal_seconds = single_signal_concat.shape[0] / sf
            LTA_window_size = min(signal_seconds / 2, 50)
            LTA_window_size = max(LTA_window_size, 10)
            STA_window_size = LTA_window_size // 10
            train_df = find_trains_STALTA(
                data=single_signal_concat,
                inspect_channel=inspect_channel,
                sf=sf,
                batch=batch_number,
                batch_length=batchsize * filelength,
                upper_thres=stalta_upper_thres,
                lower_thres=stalta_lower_thres,
                lower_seconds=STA_window_size,
                upper_seconds=LTA_window_size,
                file_start_time = file_start_time
            )

            if plot_signal:
                for train_start, train_end in zip(train_df['start'], train_df['end']):
                    train_signal = single_signal_concat[train_start:train_end]
                    fig, ax0 = plt.subplots(1, 1, figsize=(12, 6))
                    ax0.plot(single_signal_concat)
                    ax0.set_title(f"Raw Signal (File {os.path.basename(batch[0])})")
                    ax0.set_xlabel('Sample')
                    ax0.set_ylabel('Amplitude')
                    ax0.grid(True)
                    if save_path:
                        plot_filename = os.path.join(save_path, f"train_passage_{os.path.basename(batch[0])}.png")
                        plt.savefig(plot_filename)
                    plt.close()

            dfs.append(train_df)
        except TypeError as e:
            logger.warning(f"Channel {inspect_channel} failed; error message: {e}")

    # Combine results
    df = pd.concat(dfs).reset_index(drop=True)
    if df.empty:
        logger.warning("No trains detected, returning empty DataFrame")
        return pd.DataFrame({})

    # Fix: Ensure indices map correctly to filenames
    def safe_map_index(index):
        return filenames[index].name if 0 <= index < len(filenames) else "Invalid index"

    df = df.assign(startfile_index=lambda x: x.start // filelength)
    df = df.assign(endfile_index=lambda x: x.end // filelength)
    df["startfile_index"] = df["startfile_index"].astype(int)
    df["endfile_index"] = df["endfile_index"].astype(int)
    df["startfile"] = df["startfile_index"].map(safe_map_index)
    df["endfile"] = df["endfile_index"].map(safe_map_index)
    df = df.drop(columns=["startfile_index", "endfile_index", "batch"], errors="ignore")

    return df






########################################################################

# From a given folder path, get all the files with a given extension
path_to_files = Path(r'C:\Projects\erju\data\holten\recording_2024-08-29T08_01_16Z_5kHzping_1kHzlog_1mCS_10mGL_3000channels')
filenames = list(path_to_files.glob("*.h5"))
save_path = r'C:\Projects\erju\outputs\holten'

files_with_trains = detect_trainpassages_in_folder(filenames=filenames,
                                                   batchsize=2,
                                                   stalta_lower_thres=0.5,
                                                   stalta_upper_thres=6,
                                                   inspect_channel=1200,
                                                   plot_signal=True,
                                                   save_path=save_path)  # Specify channel here


print(files_with_trains)


