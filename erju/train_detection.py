# This is the new (as of 13/11/24) version of train_detection.py, based on the STA_LTA method.
# The idea of this script is to find files with trains.

# I will first start with the files of the Silixa format (.tdms)

# Import the necessary libraries
from utils.file_utils import get_files_list

def detect_trainpassage_single_channel(folder_path: str, file_extension: str = '*.h5'):
    """
    Detect the train passage in a single channel. This process looks into a specific channel (location), file by file
    for a signal anomaly that indicates the passage of a train.
     
     Args:
            folder_path (str): The path to the directory containing the files.
            file_extension (str): The file extension to look for in the directory.
    """

    file_paths = get_files_list(folder_path=folder_path, file_extension=file_extension)






########################################################################################################################

files_path = r'C:\Projects\erju\data\holten\recording_2024-08-29T08_01_16Z_5kHzping_1kHzlog_1mCS_10mGL_3000channels'
detect_trainpassage_single_channel(folder_path=files_path, file_extension='h5')

# I will now test the function with the Silixa files
print("The function has been tested successfully!")