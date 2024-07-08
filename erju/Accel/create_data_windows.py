import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from obspy.signal.trigger import recursive_sta_lta, trigger_onset




class AccelDataTimeWindows():
    """
    This class creates the windows of time in which a train passing by is detected.
    It uses the data from the accelerometers to create the windows. Since we want the same
    time window for all sensors, the windows are created with one location and used to
    extract the data from all other sensors. Outputs are the indices and times of the windows.
    """

    def __init__(self, accel_data_path: str, logbook_path: str, window_buffer: int = 10,
                 event_separation_internal: int = 5, threshold: float = 0.02):
        """
        Initialize the AccelDataTimeWindows class

        Args:
            accel_data_path (str): The path to the folder containing the accelerometer data.
            logbook_path (str): The path to the logbook file containing the train passing times.
            window_buffer (int): The size of the window extension in seconds. Default is 10 seconds.
            event_separation_internal (int): The separation between events in seconds. Default is 5 seconds.
            threshold (float): The threshold signal value to detect the train passing by. Default is 0.02.
        """

        self.accel_data_path = accel_data_path
        self.logbook_path = logbook_path
        self.accel_file_names = None  # The list of file names in the folder without extensions
        self.window_buffer = window_buffer * 1000  # measurements at 1000 Hz, we need to multiply by 1000 for seconds
        self.event_separation_internal = event_separation_internal * 1000  # measure at 1000 Hz
        self.threshold = threshold  # The threshold signal value to detect the train passing by
        self.settings = {}  # The dictionary containing the settings for each file

    def extract_accel_data_from_file(self, file_name: str, no_cols: int = 3) -> pd.DataFrame:
        """
        Read the data from a (single) .asc file and create a dataframe with the data. The data
        is read from the file and the time is created from the start time and the time
        increment. The date is extracted from the settings file.
        NOTE: this is done PER FILE, you have to choose which file to use. Also, the number of
        columns is set to 3 because we use the first 3 columns of data from the file which
        correspond to location 1. We know this beforehand by looking at the data and this
        will probably will be needed to be adjusted for other locations or file types.

        Args:
            file_name (str): The name of the file without extension
            no_cols (int): The number of columns in the file. Default is 3.

        Returns:
            accel_data (pd.DataFrame): The dataframe containing all the signal data
        """
        # Extract the settings for this file and store them in self.settings
        self.extract_settings(file_name)

        # Debug print to check settings
        print(f"Settings: {self.settings}")

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

        # Debug print to check settings access
        if "set_" + file_name not in self.settings:
            print(f"Key 'set_{file_name}' not found in settings. Available keys: {self.settings.keys()}")

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
        print(f"Settings for {file_name} stored successfully")

        return self.settings

    def create_windows_indices_and_times(self, accel_data: pd.DataFrame):
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
            start_index = max(0, start_point - self.window_buffer)
            end_index = min(len(accel_data) - 1, end_point + self.window_buffer)

            # Get the corresponding times
            start_time = accel_data["T(ms)"].iloc[start_index]
            end_time = accel_data["T(ms)"].iloc[end_index]

            # Append the window to the lists
            windows_indices.append((start_index, end_index))
            windows_times.append((start_time, end_time))

        return windows_indices, windows_times


    def detect_events_with_sta_lta(self, accel_data: pd.DataFrame, nsta: int, nlta: int,
                                   trigger_on: float, trigger_off: float):
        """
        Detect events using the STA/LTA method and create windows around these events. This method
        uses the recursive_sta_lta function from ObsPy to compute the STA/LTA ratio and the trigger_onset
        function to detect the events. A buffer is added to the start and end of each event to create the
        windows. The windows are then returned as indices and times.

        Args:
            accel_data (pd.DataFrame): The dataframe containing the data from the location_name
            nsta (int): The number of samples in the STA window
            nlta (int): The number of samples in the LTA window
            trigger_on (float): The trigger on threshold
            trigger_off (float): The trigger off threshold

        Returns:
            windows_indices (list): A list of tuples containing the start and end indices of each window
            windows_times (list): A list of tuples containing the start and end times of each window
        """
        # Combine the accelerometer signals
        combined_signal = accel_data.iloc[:, 1:].sum(axis=1).values

        # Compute the STA/LTA ratio
        sta_lta_ratio = recursive_sta_lta(combined_signal, nsta, nlta)

        # Detect events using the trigger_onset function from ObsPy
        events = trigger_onset(sta_lta_ratio, trigger_on, trigger_off)

        # Create the windows around the detected events
        windows_indices = []
        windows_times = []

        # For each event, create a window around it
        for event in events:
            # Extend the window by the window_size_extension at the start and end
            start_index = max(0, event[0] - self.window_buffer)
            end_index = min(len(accel_data) - 1, event[1] + self.window_buffer)
            # Compute the corresponding times for the start and end indices
            start_time = accel_data["T(ms)"].iloc[start_index]
            end_time = accel_data["T(ms)"].iloc[end_index]
            # Append the window to the lists
            windows_indices.append((start_index, end_index))
            windows_times.append((start_time, end_time))


        print(f"Windows indices: {windows_indices}")
        print(f"Windows times: {windows_times}")
        print(f"Number of windows: {len(windows_indices)}")

        return windows_indices, windows_times


    def plot_accel_signal_and_windows(self, accel_data: pd.DataFrame, windows_indices: list, nsta: int = None,
                                      nlta: int = None, trigger_on: float = None, trigger_off: float = None):
        """
        Plot the signal, STA/LTA ratio, and detected event windows with shaded areas.

        Args:
            accel_data (pd.DataFrame): The dataframe containing the data from the location_name
            windows_indices (list): A list of tuples containing the start and end indices of each window
            nsta (int): The number of samples in the STA window
            nlta (int): The number of samples in the LTA window
            trigger_on (float): The trigger on threshold
            trigger_off (float): The trigger off threshold
        """
        # Combine the accelerometer signals
        combined_signal = accel_data.iloc[:, 1:].sum(axis=1).values
        time_values = accel_data.iloc[:, 0]

        if nsta and nlta and trigger_on and trigger_off:
            # Compute the STA/LTA ratio if parameters are provided
            sta_lta_ratio = recursive_sta_lta(combined_signal, nsta, nlta)
        else:
            sta_lta_ratio = None

        # Plot the results
        plt.figure(figsize=(15, 8))

        # Plot the original seismic data
        plt.subplot(2, 1, 1)
        plt.plot(time_values, combined_signal, label='Seismic Data', color='black')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Raw vibration signal')
        plt.legend()

        if sta_lta_ratio is not None:
            # Plot the STA/LTA ratio
            plt.subplot(2, 1, 2)
            plt.plot(time_values[:len(sta_lta_ratio)], sta_lta_ratio, label='STA/LTA Ratio', color='blue')
            plt.axhline(y=trigger_on, color='r', linestyle='--', label='Trigger On Threshold')
            plt.axhline(y=trigger_off, color='b', linestyle='--', label='Trigger Off Threshold')

        # Highlight detected events in both subplots
        for start_index, end_index in windows_indices:
            plt.subplot(2, 1, 1)
            plt.axvspan(time_values[start_index], time_values[end_index], color='green', alpha=0.3)
            if sta_lta_ratio is not None:
                plt.subplot(2, 1, 2)
                plt.axvspan(time_values[start_index], time_values[end_index], color='green', alpha=0.3)

        if sta_lta_ratio is not None:
            plt.xlabel('Time')
            plt.ylabel('STA/LTA Ratio')
            plt.title('STA/LTA Ratio and Detected Events')
            plt.legend()

        plt.tight_layout()
        plt.show()

    def filter_windows_with_logbook(self, window_indices: list, window_times: list, time_buffer: int = 5):
        """
        Compare the windows created with the sta/lta method with the records from the accelerometer
        measurements logbook. If the windows are within the time range of the records, keep them.
        Otherwise, discard them. Because the logbook is not always accurate, we can add a buffer
        to the start and end of the windows to keep the windows that are close to the records.

        Args:
            time_buffer (int): The buffer to add to the start and end of the windows. Default is 5 seconds.
        """
        # Read the logbook file into a DataFrame
        logbook = pd.read_excel(self.logbook_path)

        # Drop the rows with missing 'Time' values
        logbook = logbook.dropna(subset=['time'])

        # Drop the rows with 'type' values that are not 's' or 'd'
        logbook = logbook[logbook['type'].isin(['s', 'd'])]

        # Convert 'day' to a timestamp format
        logbook['day'] = pd.to_datetime(logbook['day'], format='%d-%m-%Y')
        # Convert 'time' to a timedelta format (HH:MM:SS)
        logbook['time'] = pd.to_timedelta(logbook['time'].astype(str))
        # Combine 'day' and 'time' into a single datetime column
        logbook['datetime'] = logbook['day'] + logbook['time']

        # Convert the time_buffer to a timedelta
        time_buffer = timedelta(seconds=time_buffer)

        # Create a list for the filtered window times and indices
        filtered_windows_indices, filtered_windows_times = [], []

        # Create a list to keep track of used logbook times
        used_logbook_times = []

        # For each window, check if it falls within the logbook times
        for window_index, window_time in zip(window_indices, window_times):
            # Get the start and end times of the window
            start_time, end_time = window_time
            # Extend the start and end times by the buffer
            extended_start_time = start_time - time_buffer
            extended_end_time = end_time + time_buffer

            best_logbook_time = None
            best_midpoint_difference = None

            # Check if any logbook times fall within the extended window times
            for logbook_time in logbook['datetime']:
                if logbook_time in used_logbook_times:
                    continue
                if extended_start_time <= logbook_time <= extended_end_time:
                    # Calculate the midpoint of the extended window
                    extended_midpoint = extended_start_time + (extended_end_time - extended_start_time) / 2
                    # Calculate the difference between the logbook time and the midpoint
                    midpoint_difference = abs((logbook_time - extended_midpoint).total_seconds())
                    # If this is the best (smallest) midpoint difference found so far, update the best logbook time
                    if best_midpoint_difference is None or midpoint_difference < best_midpoint_difference:
                        best_midpoint_difference = midpoint_difference
                        best_logbook_time = logbook_time

            if best_logbook_time is not None:
                # If a best logbook time is found, use it to filter the window
                filtered_windows_indices.append(window_index)
                filtered_windows_times.append(window_time)
                # Add the best logbook time to the used list
                used_logbook_times.append(best_logbook_time)
                # Print the matching extended window time
                print(
                    f"Window {window_index} with original time {window_time} matches logbook time {best_logbook_time} within extended window ({extended_start_time}, {extended_end_time})")
            else:
                print(f"No time fits the window {window_time}")

        print(f"Filtered windows indices: {filtered_windows_indices}")
        print(f"Filtered windows times: {filtered_windows_times}")
        print(f"Length of filtered windows: {len(filtered_windows_indices)}")

        return filtered_windows_indices, filtered_windows_times



