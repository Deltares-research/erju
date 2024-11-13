# This is the new (as of 13/11/24) version of train_detection.py, based on the STA_LTA method.
# The idea of this script is to find files with trains.

# I will first start with the files of the Silixa format (.tdms)

# Import the necessary libraries
import h5py
from loguru import logger
from utils.file_utils import get_files_list, get_file_extensions

def detect_trainpassage_single_channel_h5(folder_path: str):
    """
    Detect the train passage in a single channel. This process looks into a specific channel (location), file by file
    for a signal anomaly that indicates the passage of a train.
     
     Args:
            folder_path (str): The path to the directory containing the files.
            file_extension (str): The file extension to look for in the directory.
    """
    file_extension = '*.h5'
    file_paths = get_files_list(folder_path=folder_path, file_extension=file_extension)


def detect_trainpassage_single_channel(folder_path: str):
    """
    Detect the train passage in a single channel. This process looks into a specific channel (location), file by file
    for a signal anomaly that indicates the passage of a train. It will call a different function depending on the
    file extension.

     Args:
            folder_path (str): The path to the directory containing the files.
    """

    # Browse the directory and gather a list of all the file extensions.
    file_extensions = get_file_extensions(folder_path=folder_path)

    # Call a different function depending on the file extension.
    if '.h5' in file_extensions:
        logger.info("The files are in the OptaSense .h5 format.")
        detect_trainpassage_single_channel_h5(folder_path=folder_path)


    elif '.tdms' in file_extensions:
        logger.info("The files are in the Silixa .tdms format.")

    elif '.bin' in file_extensions:
        logger.info("The files are in the Focus .bin format.")

    else:
        logger.warning("The file extension is not recognized. Please provide files in the Silixa .tdms, OptaSense .h5 or Focus .bin format.")







########################################################################################################################

files_path = r'C:\Projects\erju\data\holten\recording_2024-08-29T08_01_16Z_5kHzping_1kHzlog_1mCS_10mGL_3000channels'
detect_trainpassage_single_channel(folder_path=files_path)

# I will now test the function with the Silixa files
print("The function has been tested successfully!")