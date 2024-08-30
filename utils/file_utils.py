import os
import re
from datetime import datetime

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


