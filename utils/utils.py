
import os


def get_files_in_dir(folder_path: str, file_format: str):
    """
    Get a list of unique file names inside the given folder path that match the given file format.

    Args:
        folder_path (str): The path to the folder containing the files.
        file_format (str): The file format to filter by (e.g., ".tdms").

    Returns:
        file_list (list): List of file names in the folder that match the given format with extensions.
    """
    # Validate the inputs
    if not os.path.isdir(folder_path):
        raise ValueError("The provided path is not a valid directory.")
    if not file_format.startswith('.'):
        raise ValueError("The file format should start with a dot (e.g., '.txt').")

    # Get the list of files in the folder with the specified extension
    file_list = [f for f in os.listdir(folder_path) if f.endswith(file_format)]

    return file_list

