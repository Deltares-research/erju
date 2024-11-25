from pathlib import Path

import h5py
import numpy as np
import pandas as pd
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

    cft = recursive_sta_lta(trace.data, int(lower * freq), int(upper * freq))
    if plots:
        plot_trigger(trace, cft, upper_thres, lower_thres)
    return cft


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
    """Detect trains in a single channel using the STA-LTA algorithm.

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

    return df_trains


def detect_trainpassages_in_folder(
    filenames: list[Path],
    detection_resolution: int,
    batchsize: int = 2,
    stalta_lower_thres: float = 0.5,
    stalta_upper_thres: float = 6,
) -> pd.DataFrame:
    """Detects all the h5py files with trainpassages in a folder using the STA-LTA algorithm.

    Args:
        filenames (list[Path]): List of file paths to the h5py files, files are assumed to be in the same folder
        detection_resolution (int): Gap between channels used in the method to detect trains
        batchsize (int, optional): Number of files processed per batch. Defaults to 2.
        stalta_lower_thres (float, optional): Threshold for switching trigger off. Defaults to 1.5.
        stalta_upper_thres (float, optional): Threshold for switching trigger on. Defaults to 4.5.

    Returns:
        pd.DataFrame: Table with channel, filename and start, end times of the detected trains
    """
    # Get metadata from the first file
    # Compare with the last file to ensure they are the same
    with h5py.File(filenames[0], "r") as file_start, h5py.File(filenames[-1], "r") as file_end:
        sf = calculate_sampling_frequency(file_start)
        if sf != calculate_sampling_frequency(file_end):
            raise ValueError("Sampling frequency is not the same in the begin and end files")

        data_shape = file_start["Acquisition"]["Raw[0]"]["RawData"].shape
        if data_shape != file_end["Acquisition"]["Raw[0]"]["RawData"].shape:
            raise ValueError("Data shape is not the same in the begin and end files")

        filelength = data_shape[0]
        n_channels = data_shape[1]
        channels_to_inspect = list(range(0, n_channels, detection_resolution))

    # Load local files
    dfs = []
    file_batches = [filenames[i : i + batchsize] for i in range(0, len(filenames), batchsize)]
    batchlength = batchsize * filelength
    for batch_number, batch in enumerate(file_batches):
        logger.info(f"Reading files in batch {batch_number}")
        batch_data = []

        for file_path in batch:
            with h5py.File(file_path, "r") as file:
                data = file["Acquisition"]["Raw[0]"]["RawData"][:, channels_to_inspect]
                batch_data.append(data)

        batch_data = np.concatenate(batch_data, axis=0)

        for channel_index, channel in enumerate(channels_to_inspect):
            try:
                single_signal_concat = batch_data[:, channel_index]  # type: ignore

                signal_seconds = single_signal_concat.shape[0] / sf

                # The LTA window size is determined by the signal length
                LTA_window_size = min(signal_seconds / 2, 50)
                LTA_window_size = max(LTA_window_size, 10)
                STA_window_size = LTA_window_size // 10
                dfs.append(
                    find_trains_STALTA(
                        single_signal_concat,
                        channel,
                        sf,
                        batch_number,
                        batchlength,
                        upper_thres=stalta_upper_thres,
                        lower_thres=stalta_lower_thres,
                        lower_seconds=STA_window_size,
                        upper_seconds=LTA_window_size,
                    )
                )
            except TypeError as e:
                logger.warning(f"Channel {channel} failed; error message: {e}")

    df = pd.concat(dfs).reset_index(drop=True)

    if df.empty:
        logger.warning("No trains detected, return empty DataFrame")
        return pd.DataFrame({})

    df = df.assign(startfile_index=lambda x: x.start // (filelength))
    df = df.assign(endfile_index=lambda x: x.end // (filelength))
    df["startfile_index"] = df["startfile_index"].astype(int)
    df["endfile_index"] = df["endfile_index"].astype(int)
    df["startfile"] = df["startfile_index"].map(lambda x: filenames[x].name)
    df["endfile"] = df["endfile_index"].map(lambda x: filenames[x].name)
    df = df.drop(columns=["startfile_index", "endfile_index", "batch"])
    return df

########################################################################################################################
"""
Based on the code provided by Joost (ProRail) and Edwin (Deltares), this code:
- Looks through all .h5 files in a given folder
- Then open file by file (batch=1; can be changed)
- And looks through a channel each 250 spaces (detection_resolution=250; can be changed)
- Applies the STA-LTA algorithm to each of those channels
- If it detects a train, it saves the start and end times of the train, the channel number, and the file name
- It then saves all the files with trains in a CSV file
"""
#TODO: This now works only with .h5 files. Lets make it general like in the old train_detectionOLDFILE.py

# From a given folder path, get all the files with a given extension
path_to_files = Path(r'C:\Projects\erju\data\holten\recording_2024-08-29T08_01_16Z_5kHzping_1kHzlog_1mCS_10mGL_3000channels')
filenames = list(path_to_files.glob("*.h5"))
save_path = r'C:\Projects\erju\outputs\holten'

files_with_trains = detect_trainpassages_in_folder(filenames=filenames,
                                                   batchsize=1,
                                                   detection_resolution=250,
                                                   stalta_lower_thres=0.5,
                                                   stalta_upper_thres=6,
                                                   )

# Combine "startfile" and "endfile" columns into one Series and get unique values
all_files = pd.concat([files_with_trains['startfile'], files_with_trains['endfile']]).unique()

# Convert the result to a DataFrame for saving as a CSV
pd.DataFrame(all_files, columns=['file_name']).to_csv(save_path + r'\all_files_with_trains.csv', index=False)



