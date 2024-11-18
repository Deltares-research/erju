# This is the new main script for processing the FO data.
# Date: 15/11/2024

# Import the necessary libraries
from loguru import logger

from utils.file_utils import get_file_extensions

# In this script, you will give a folder with data, specify the channel of interest as well as how many channels
# next to the main channel (to both sides) you want to extract and choose if you want
# to plot the complete file and/or the channels. The script will then use the specif channel
# and the STA/LTA method to detect the train passage in the data, and extract said data.

def find_and_extract_train_signals(dir_path: str,
                                   save_path: str,
                                   channel: int,
                                   num_channels: int,
                                   plot_channels: bool = False,
                                   plot_file: bool = False,):
    """
    This function will find and extract train signals from the data in the specified directory.

    Args:
        dir_path (str): The path to the directory containing the data files.
        save_path (str): The path to save the extracted data.
        channel (int): The channel number to use for train signal detection.
        num_channels (int): The number of channels to extract next (both sides) to the main channel.
        plot_channels (bool): Whether to plot the extracted channels.
        plot_file (bool): Whether to plot the complete file.
    """

    # Browse the directory and gather a list of all the file extensions.
    file_extensions = get_file_extensions(folder_path=dir_path)

    # Call a different function depending on the file extension.
    if '.h5' in file_extensions:
        logger.info("The files are in the OptaSense .h5 format.")
        detect_trainpassage_single_channel_h5(folder_path=dir_path, channel=channel)
        print(df)

    elif '.tdms' in file_extensions:
        logger.info("The files are in the Silixa .tdms format.")

    elif '.bin' in file_extensions:
        logger.info("The files are in the Focus .bin format.")

    else:
        logger.warning(
            "The file extension is not recognized. Please provide files in the Silixa .tdms, OptaSense .h5 or Focus .bin format.")

