# This is the new (as of 13/11/24) version of train_detection.py, based on the STA_LTA method.
# The idea of this script is to find files with trains.

# I will first start with the files of the Silixa format (.tdms)

# Import the necessary libraries
import h5py
from loguru import logger
import numpy as np
import pandas as pd
from datetime import timedelta
from utils.file_utils import get_files_list, get_file_extensions, find_trains_STALTA, do_stalta, highpass
from utils.h5_utils import calculate_sampling_frequency, convert_microseconds_to_datetime

def detect_trainpassage_single_channel_h5(folder_path: str,
                                          channel: int,
                                          batch_size: int = 2,
                                          stalta_lower_thres: float = 0.5,
                                          stalta_upper_thres: float = 6):
    """
    Detect the train passage in a single channel. This process looks into a specific channel (location), file by file
    for a signal anomaly that indicates the passage of a train.

     Args:
            folder_path (str): The path to the directory containing the files.
            batch_size (int): The number of files to read at once.
            channel (int): The channel number to look for the train passage.
    """
    file_extension = '*.h5'
    file_paths = get_files_list(folder_path=folder_path, file_extension=file_extension)

    # Get metadata from the first file
    # Compare with the last file to ensure they are the same
    with h5py.File(file_paths[0], "r") as file_start, h5py.File(file_paths[-1], "r") as file_end:
        sf = calculate_sampling_frequency(file_start)
        if sf != calculate_sampling_frequency(file_end):
            raise ValueError("Sampling frequency is not the same in the begin and end files")

        # Compare the data shape of the first and last files
        data_shape = file_start["Acquisition"]["Raw[0]"]["RawData"].shape
        if data_shape != file_end["Acquisition"]["Raw[0]"]["RawData"].shape:
            raise ValueError("Data shape is not the same in the begin and end files")

        # Calculate the length of the data
        file_length = data_shape[0]

        # Extract the start time of the first file for reference
        rawDataTime = file_start['Acquisition']['Raw[0]']['RawDataTime']
        file_start_time = convert_microseconds_to_datetime(rawDataTime[0])

        # Create an empty dataframe to store the results
        dfs = []
        # Calculate the number of batches to read the files
        file_batches = [file_paths[i : i + batch_size] for i in range(0, len(file_paths), batch_size)]
        # Calculate the length of the batch
        batch_length = batch_size * file_length

        # Loop over the batches
        for batch_number, batch in enumerate(file_batches):
            logger.info(f"Reading files in batch {batch_number}")
            batch_data = []
            # Loop over the files in the batch
            for file_path in batch:
                with h5py.File(file_path, "r") as file:
                    # Get only the specified channel's data
                    data = file["Acquisition"]["Raw[0]"]["RawData"][:, channel]
                    # Append the data to the batch
                    batch_data.append(data)

            # Concatenate the data in the batch into a single array
            batch_data = np.concatenate(batch_data, axis=0)

            try:
                # Calculate the length of the signal in seconds
                signal_seconds = batch_data.shape[0] / sf

                # The LTA window size is determined by the signal length
                LTA_window_size = min(signal_seconds / 2, 50)
                LTA_window_size = max(LTA_window_size, 10)
                STA_window_size = LTA_window_size // 10
                # Apply the STA/LTA method to the signal
                dfs.append(
                    find_trains_STALTA(
                        batch_data,
                        channel,
                        sf,
                        batch_number,
                        batch_length,
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

        df = df.assign(startfile_index=lambda x: x.start // (file_length))
        df = df.assign(endfile_index=lambda x: x.end // (file_length))
        df["startfile_index"] = df["startfile_index"].astype(int)
        df["endfile_index"] = df["endfile_index"].astype(int)
        df["startfile"] = df["startfile_index"].map(lambda x: file_paths[x].name)
        df["endfile"] = df["endfile_index"].map(lambda x: file_paths[x].name)

        # Add start and end time for each detected event
        df['event_start_time'] = df['start'].apply(
            lambda x: (file_start_time + timedelta(seconds=x / sf)).strftime("%d/%m/%Y %H:%M:%S"))
        df['event_end_time'] = df['end'].apply(
            lambda x: (file_start_time + timedelta(seconds=x / sf)).strftime("%d/%m/%Y %H:%M:%S"))

        # Drop unnecessary columns
        df = df.drop(columns=["startfile_index", "endfile_index", "batch"])

        return df





def detect_trainpassage_single_channel(folder_path: str, channel: int):
    """
    Detect the train passage in a single channel. This process looks into a specific channel (location), file by file
    for a signal anomaly that indicates the passage of a train. It will call a different function depending on the
    file extension.

     Args:
            folder_path (str): The path to the directory containing the files.
            channel (int): The channel number to look for the train passage.
    """

    # Browse the directory and gather a list of all the file extensions.
    file_extensions = get_file_extensions(folder_path=folder_path)

    # Call a different function depending on the file extension.
    if '.h5' in file_extensions:
        logger.info("The files are in the OptaSense .h5 format.")
        df = detect_trainpassage_single_channel_h5(folder_path=folder_path, channel=channel)
        print(df)

    elif '.tdms' in file_extensions:
        logger.info("The files are in the Silixa .tdms format.")

    elif '.bin' in file_extensions:
        logger.info("The files are in the Focus .bin format.")

    else:
        logger.warning("The file extension is not recognized. Please provide files in the Silixa .tdms, OptaSense .h5 or Focus .bin format.")







########################################################################################################################

files_path = r'C:\Projects\erju\data\holten\recording_2024-08-29T08_01_16Z_5kHzping_1kHzlog_1mCS_10mGL_3000channels'
detect_trainpassage_single_channel(folder_path=files_path, channel=1200)

print('Done')