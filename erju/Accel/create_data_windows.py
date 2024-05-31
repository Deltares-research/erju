import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class AccelDataTimeWindows():
    """
    This class creates the windows of time in which a train passing by is detected.
    It uses the data from the accelerometers to create the windows. Since we want the same
    time window for all sensors, the windows are created with one location and used to
    extract the data from all other sensors.
    """

    def __init__(self, accel_data_path: str, window_size: int = 10, event_separation_internal: int = 5,
                 threshold: float = 0.02):
        """
        Initialize the AccelDataTimeWindows class

        Args:
            accel_data_path (str): The path to the folder containing the accelerometer data.
            window_size (int): The size of the window in seconds. Default is 10 seconds.
            event_separation_internal (int): The separation between events in seconds. Default is 5 seconds.
            threshold (float): The threshold signal value to detect the train passing by. Default is 0.02.
        """

        self.accel_data_path = accel_data_path
        self.accel_file_names = None  # The list of file names in the folder without extensions
        self.window_size = window_size * 1000  # measurements at 1000 Hz, we need to multiply by 1000 for seconds
        self.event_separation_internal = event_separation_internal * 1000  # measure at 1000 Hz
        self.threshold = threshold  # The threshold signal value to detect the train passing by
        self.settings = {}  # The dictionary containing the settings for each file

    def get_file_names(self):
        """
        Get a list of unique file names inside the dir_path folder. For this
        we look into one specific file format (.asc) and remove the extension.

        Args:
            None

        Returns:
            file_list (list): List of file names in the folder without extensions
        """
        # Get the list of files in the folder with .asc extension
        ascii_file_names = [f for f in os.listdir(self.accel_data_path) if f.endswith('.asc')]
        # From the ascii_file_names list, remove the .asc extension
        self.accel_file_names = [f.split('.')[0] for f in ascii_file_names]

        return self.accel_file_names

    def extract_accel_data_from_file(self, file_name: str, no_cols: int = 3) -> pd.DataFrame:
        """
        Read the data from the .asc file and create a dataframe with the data. The data
        is read from the file and the time is created from the start time and the time
        increment. The date is extracted from the settings file.

        Args:
            file_name (str): The name of the file without extension
            no_cols (int): The number of columns in the file. Default is 3.

        Returns:
            accel_data (pd.DataFrame): The dataframe containing all the signal data
        """
        # Create the file path by adding the file name and the .asc extension
        file_path = os.path.join(self.accel_data_path, file_name + '.asc')

        # Generate column names dynamically
        column_names = ['T(ms)'] + [f'Chan{i}' for i in range(1, no_cols)]

        # Read the file into a dataframe with dynamically generated column names
        accel_data = pd.read_csv(file_path,
                                 sep='\t',
                                 usecols=range(no_cols),
                                 skiprows=1,
                                 dtype={0: int, **{i: float for i in range(1, no_cols)}},
                                 names=column_names)

        # Get the start time from the settings file and format it as a datetime object
        start_time_str = self.settings["set_" + file_name]["Start time"].strip('"')
        start_time = datetime.strptime(start_time_str, "%H:%M:%S")

        # Create a time series that starts from the start time and increments by 1 ms for each row
        time_series = [start_time + timedelta(milliseconds=i) for i in range(len(accel_data))]

        # Replace the "T(ms)" column with the new time series
        accel_data["T(ms)"] = time_series

        # Get the date of the measurement from the "Last Modified" key in the settings
        last_modified_str = self.settings["set_" + file_name]["Last Modified"].strip('"')
        last_modified_date = datetime.strptime(last_modified_str, "%d-%m-%Y %H:%M:%S").date()

        # Replace the date part of the "T(ms)" column with the "Last Modified" date
        accel_data["T(ms)"] = accel_data["T(ms)"].apply(
            lambda dt: datetime.combine(last_modified_date, dt.time()))

        return accel_data

    def extract_settings(self, file_name: str) -> dict:
        """
        From the settings file (.set) with the same file names,
        extract the settings and store them in a dictionary.

        Args:
            file_name (str): The name of the file without extension

        Returns:
            settings (dict): A dictionary containing the settings for the file
        """
        # Create the file path by adding the file name and the .set extension
        file_path = os.path.join(self.accel_data_path, file_name + '.set')

        # Create an empty dictionary to store the settings
        file_settings = {}

        # Check if the .set file exists
        if not os.path.exists(file_path):
            # If the file does not exist, print an error message and return
            print(f"Error: The file {file_path} does not exist.")
            self.settings["set_" + file_name] = file_settings
            return

        try:
            # Try to open the file and extract the settings
            with open(file_path, 'r') as f:
                # Read each line in the file
                for line in f:
                    # If the line contains a '=', it represents a setting
                    if '=' in line:
                        # Split the line into a key and a value
                        key, value = line.strip().split('=')

                        # Store the setting in the dictionary
                        file_settings[key.strip()] = value.strip()

        except Exception as e:
            # If an error occurred while reading the file, print an error message
            print(f"Error: An error occurred while reading the file {file_path}: {e}")

        # Store the settings for this file in the settings dictionary
        self.settings["set_" + file_name] = file_settings

        return self.settings

    def create_windows_indices_and_times(self, accel_data):
        """
        Create the windows of time in which a train passing by is detected.
        To do this, the threshold signal value is used to detect the train.
        Also, as each location has 2(2D) or 3(3D) signals, they are combined
        before creating the windows.

        Args:
            accel_data (pd.DataFrame): The dataframe containing the data from the location_name

        Returns:
            windows_indices (list): A list of tuples containing the start and end indices of each window
            windows_times (list): A list of tuples containing the start and end times of each window
        """
        # Create empty lists to store the windows indices and times
        windows_indices = []
        windows_times = []

        # Combine all the signals available in the extracted df with data, except the time column
        combined_signal = accel_data.iloc[:, 1:].sum(axis=1)

        # Find the indices where the combined signal crosses the threshold
        cross_indices = np.where(np.diff(combined_signal > self.threshold))[0]

        # Split the indices into "events" where the signal is above the threshold
        # For this we look at the difference between the indices and split when the difference
        # is greater than the event_separation_internal value (in this case 5 seconds)
        events = np.split(cross_indices, np.where(np.diff(cross_indices) > self.event_separation_internal)[0] + 1)

        # For each "event", find the start and end points
        for event in events:
            start_point = event[0]
            end_point = event[-1]

            # Create a window around the start and end points by adding the window_size
            # both before and after the start and end points. We also make sure that the
            # window does not go outside the signal data (that's why we use max and min)
            start_index = max(0, start_point - self.window_size)
            end_index = min(len(accel_data) - 1, end_point + self.window_size)

            # Get the corresponding times
            start_time = accel_data["T(ms)"].iloc[start_index]
            end_time = accel_data["T(ms)"].iloc[end_index]

            # Append the window to the lists
            windows_indices.append((start_index, end_index))
            windows_times.append((start_time, end_time))

        return windows_indices, windows_times


# TRY TO RUN THE CODE #################################################################################################

# Define the path to the folder containing the accelerometer data
accel_data_path = r'D:\accel_trans\culemborg_data'

# Create an instance of the AccelDataTimeWindows class
time_windows = AccelDataTimeWindows(accel_data_path=accel_data_path, window_size=10, event_separation_internal=5,
                                    threshold=0.02)

# Get the list of file names in the folder
file_names = time_windows.get_file_names()
# Extract the settings for the first file
settings = time_windows.extract_settings(file_names[0])
# Create a dataframe with the data from the first location, specifying the number of columns
accel_data_df = time_windows.extract_accel_data_from_file(file_names[0], no_cols=3)

# Find the indices and times where the combined signal crosses the threshold
windows_indices, windows_times = time_windows.create_windows_indices_and_times(accel_data_df)
print('number of windows:', len(windows_indices))
print('windows indices:', windows_indices)
print('windows times:', windows_times)
