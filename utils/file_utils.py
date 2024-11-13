import os
import re
from datetime import datetime
from pathlib import Path
from loguru import logger

# Old script to get the files in a directory
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


def get_files_list(folder_path: str, file_extension: str = 'h5'):
    """
    Get a list of files in a directory with a specific extension.

    Args:
        folder_path (str): The path to the directory containing the files.
        file_extension (str): The file extension to search for in the directory (default is 'h5').

    Returns:
        files: A list of Paths to the files detected in the folder.
    """

    # Convert the folder path to a Path object
    folder_path = Path(folder_path)

    # Check if the provided path is a directory, if not return an empty and raise an error
    if not folder_path.is_dir():
        logger.error(f"Provided path is not a directory: {folder_path}")
        return []

    # Automatically add '*' if the user provides only the extension
    if not file_extension.startswith("*."):
        file_extension = f"*.{file_extension}"

    # Get a list of only the file names in the directory
    file_names = [file.name for file in folder_path.glob(file_extension)]
    # Get a list of the complete file paths
    file_paths = list(folder_path.glob(file_extension))


    # If no files are found, log a warning and return an empty list
    if not file_paths:
        logger.warning(f"No files found with extension '{file_extension}' in {folder_path}")
        return []

    # Log the number of files detected
    logger.info(f"Detected {len(file_names)} files in {folder_path}")

    return file_paths


def get_file_extensions(folder_path: str):
    """
    Get a list of all unique file extensions in the specified folder.

    Args:
        folder_path (str): The path to the directory to scan for file extensions.

    Returns:
        List[str]: A sorted list of unique file extensions (e.g., ['.txt', '.jpeg', '.pdf']).
    """

    folder_path = Path(folder_path)

    if not folder_path.is_dir():
        print(f"Provided path is not a directory: {folder_path}")
        return []

    # Collect all file extensions in the folder (excluding subdirectories)
    extensions = {file.suffix for file in folder_path.iterdir() if file.is_file()}

    # Convert set to a sorted list for readability
    return sorted(extensions)
