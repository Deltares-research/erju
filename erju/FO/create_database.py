import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from erju.Accel.create_data_windows import AccelDataTimeWindows
from erju.FO.find_trains_base import BaseFindTrains
from utils.utils import get_files_in_dir, extract_timestamp_from_name
from collections import defaultdict

class CreateDatabase:
    """
    A class used to create a database of FO and accelerometer data.
    It will create a database with at least 4 elements inside: the first element is
    the start time of the event, the second element is the end time of the event, the third
    element is the FO signal and the fourth, fith and so on, will be all the accelerometer
    signals, with as many elements as sensors we have.
    """

    def __init__(self, fo_data_path: str, acc_data_path: str, logbook_path: str):
        """
        Initialize the class instance to perform the database creation.
        Key elements are the paths to the FO and accelerometer data files and
        using the AccelDataTimeWindows and BaseFindTrains classes for the operations.

        Args:
            fo_data_path (str): The path to the folder containing the FO data files.
            acc_data_path (str): The path to the folder containing the accelerometer data files.
        """
        self.fo_data_path = fo_data_path
        self.acc_data_path = acc_data_path
        self.logbook_path = logbook_path
        self.window_indices = None
        self.window_times = None
        self.database = None

    def from_accel_get_windows(self, window_buffer: int = 10, threshold: float = 0.02, nsta: float = 0.5,
                               nlta: float = 5.0, trigger_on: float = 7, trigger_off: float = 1):
        """
        Get the time windows from the accelerometer data using the STA/LTA method.

        Args:
            window_buffer (int): The buffer to add to the time window.
            threshold (float): The threshold to use for the time window.
            nsta (float): The length of the average short time average window.
            nlta (float): The length of the average long time average window.
            trigger_on (float): The threshold to trigger the event on.
            trigger_off (float): The threshold to trigger the event off.

        Returns:
            accel_data_df (pd.DataFrame): The dataframe with the accelerometer data.
        """
        # Create an instance of the AccelDataTimeWindows class
        accel_windows = AccelDataTimeWindows(accel_data_path=self.acc_data_path,
                                             logbook_path=self.logbook_path,
                                             window_buffer=window_buffer,
                                             threshold=threshold)

        # Get the list of files names in the folder
        file_names = get_files_in_dir(folder_path=self.acc_data_path, file_format='.asc', keep_extension=False)

        # Create a dataframe with the data from the first location, specifying the number of columns
        # (in this case 3, because we use the first 3 columns of data from the file) and get the data
        # from the first file in the list
        accel_data_df = accel_windows.extract_accel_data_from_file(file_name=file_names[0], no_cols=3)

        # Detect the events using the STA/LTA method
        nsta = int(nsta * 1000) # convert to seconds with a fz of 1000 Hz
        nlta = int(nlta * 1000) # convert to seconds with a fz of 1000 Hz
        self.window_indices, self.window_times = accel_windows.detect_events_with_sta_lta(accel_data=accel_data_df,
                                                                                nsta=nsta,
                                                                                nlta=nlta,
                                                                                trigger_on=trigger_on,
                                                                                trigger_off=trigger_off)

        # Return
        return self.window_indices, self.window_times

    def from_windows_get_times(self):
        """
        Using the windows found with from_accel_get_windows, create a database with the start and end times.
        """

        # Create an empty dataframe to store the data
        self.database = pd.DataFrame()

        # Create 2 columns called 'start_time' and 'end_time' with the window times.
        # The window_times is a list of tuples with the start and end times of the events.
        self.database['start_time'] = [t[0] for t in self.window_times]
        self.database['end_time'] = [t[1] for t in self.window_times]

        # Convert the start_time and end_time columns to datetime objects
        self.database['start_time'] = pd.to_datetime(self.database['start_time'])
        self.database['end_time'] = pd.to_datetime(self.database['end_time'])

        # Calculate the time difference (dt) between start_time and end_time
        self.database['dt'] = self.database['end_time'] - self.database['start_time']

        return self.database


    def from_windows_get_fo_signal(self):
        """
        Get the FO signal data for the time windows.
        """

        # Get a list of all the file names in format .tdms in the folder
        file_names = get_files_in_dir(folder_path=self.fo_data_path, file_format='.tdms')

        # Extract the timestamps from the file names
        timestamps = extract_timestamp_from_name(file_names)

        # Loop through the rows in the dataframe and look for the timestamps with times that fit within each start and end time
        for index, row in self.database.iterrows():
            start_time = row['start_time']
            end_time = row['end_time']

            # Initialize a list to hold the files that match the time window
            matching_files = []

            # Loop through the timestamps and find the ones that fit within the start and end time
            for i, timestamp in enumerate(timestamps):
                # Ensure the timestamp is a pandas Timestamp object
                if isinstance(timestamp, datetime):
                    timestamp = pd.Timestamp(timestamp)
                if start_time <= timestamp <= end_time:
                    # Get the file name
                    file_name = file_names[i]
                    # Append the file name to the list of matching files
                    matching_files.append(file_name)

            # Join the list of matching files into a single string separated by commas (or store as a list if preferred)
            self.database.at[index, 'files'] = ', '.join(matching_files)

        return self.database


    def join_fo_channel_per_day(self, channel_no: int = 4270):
        '''
        Join the FO channel data per day for the given channel number.

        Args:
            channel_no (int): The channel number to join.
            folder_path (str): The path to the folder containing the files.

        Returns:
            combined_data (pd.DataFrame): The combined data for the given channel number.
        '''
        # Get the list of file names in the folder
        file_names = get_files_in_dir(folder_path=self.fo_data_path, file_format='.tdms')

        # Extract the timestamps from the file names
        timestamps = extract_timestamp_from_name(file_names)

        # Order the file names chronographically
        timestamps, file_names = zip(*sorted(zip(timestamps, file_names)))
        print('Ordered file names:', file_names)

        # Get the unique dates in the timestamps
        unique_dates = set([ts.date() for ts in timestamps])

        # For each unique date, get the files that match the date
        file_dict = {}
        for date in unique_dates:
            # Get the files that match the date
            matching_files = [file for file in file_names if date.strftime('%Y%m%d') in file]
            # Order the files by time in the file name
            matching_files.sort()

            print('Matching files:', matching_files)

            # Create an empty list to store the data
            data = []
            # Loop through the matching files
            for file in matching_files:
                # Create an instance of the BaseFindTrains class
                base_find_trains = BaseFindTrains.create_instance(dir_path=self.fo_data_path, first_channel=channel_no, last_channel=channel_no + 1, reader='silixa')

                # Get the properties of the TDMS file
                base_find_trains.extract_properties()
                # Extract the data from the file and append it to the list
                data_from_file = base_find_trains.get_data_per_file(file)
                print(f'Extracted data from file: {file}')  # Debug print
                data.append(data_from_file)

            # Combine the data into a single variable and store it in the dictionary per day
            file_dict[date] = data

        return file_dict

    from collections import defaultdict
    import numpy as np

    def join_fo(self, channel_no: int = 4270):
        # Get all the tdms file names
        file_names = get_files_in_dir(folder_path=self.fo_data_path, file_format='.tdms')

        # Extract the timestamps from the file names
        timestamps = extract_timestamp_from_name(file_names)

        # Order the file names chronologically
        timestamps, file_names = zip(*sorted(zip(timestamps, file_names)))

        # Initialize the FindTrains class instance to get the data
        file_instance = BaseFindTrains.create_instance(
            dir_path=self.fo_data_path,
            first_channel=channel_no,
            last_channel=channel_no,
            reader='silixa'
        )

        # Extract the properties of the TDMS file
        file_instance.extract_properties()

        # Group file names by day
        files_by_day = defaultdict(list)
        for timestamp, file_name in zip(timestamps, file_names):
            date = timestamp.date()
            files_by_day[date].append(file_name)

        # Initialize the dictionary to store the data
        all_data_by_day = {}

        # Extract and concatenate data for each day
        for day, day_file_names in files_by_day.items():
            print(f'Processing files for {day}:', day_file_names)

            # Initialize an empty list to store the data for the day
            daily_data = []

            # Loop through the file names for the day and extract the data
            for file_name in day_file_names:
                print(f'Processing file: {file_name}')
                data_dict = file_instance.get_data_per_file([file_name])

                # Assuming data_dict is a dictionary where values are numpy arrays
                for key, data in data_dict.items():
                    daily_data.append(data)

            # Concatenate all data arrays for the day into a single time series
            if daily_data:
                all_data_by_day[day] = np.concatenate(daily_data, axis=1)
                print(f'All data for {day}:', all_data_by_day[day])

        # Return the dictionary with all concatenated data per day
        return all_data_by_day


def plot_timeseries(data_by_day, date_str):
    """
    Plots the time series for a given day based on the input string in YYYYMMDD format.

    Parameters:
    data_by_day (dict): A dictionary with dates as keys and concatenated time series as values.
    date_str (str): The date string in YYYYMMDD format to plot the time series for.
    """
    # Convert the date string to a datetime.date object
    try:
        selected_date = datetime.strptime(date_str, '%Y%m%d').date()
    except ValueError:
        print("Invalid date format. Please use YYYYMMDD format.")
        return

    # Check if the date exists in the data
    if selected_date in data_by_day:
        time_series_data = data_by_day[selected_date]

        # Debug: Print the type and contents of time_series_data
        print(f"Data for {selected_date}:")
        print(type(time_series_data))
        print(time_series_data)

        # Ensure the data is not empty and is a numpy array
        if isinstance(time_series_data, np.ndarray) and time_series_data.size > 0:
            # Check if the data is 2D and has only one row
            if time_series_data.ndim == 2 and time_series_data.shape[0] == 1:
                time_series_data = time_series_data[0]  # Take the first (and only) row

            # Plot the time series data
            plt.figure(figsize=(15, 3))
            plt.plot(time_series_data)
            plt.title(f'Time Series for {selected_date}')
            plt.xlabel('Time')
            plt.ylabel('Measurement')
            plt.grid(True)
            plt.show()
        else:
            print(f"No valid data available for {selected_date}")
    else:
        print(f"No data available for {selected_date}")


# Example usage:
# Assuming all_data_by_day is the dictionary returned from join_fo function
# plot_timeseries(all_data_by_day, '20201122')  # Plot the time series for the specified date


# Example usage:
# Assuming all_data_by_day is the dictionary returned from join_fo function
# plot_timeseries(all_data_by_day, '20201122')  # Plot the time series for the specified date


# Example usage:
# Assuming all_data_by_day is the dictionary returned from join_fo function
# plot_timeseries(all_data_by_day, '20201122')  # Plot the time series for the specified date


##### TEST THE CODE ###################################################################################################
'''


# Define the path to the fo data
#fo_data_path = r'D:\RAIL4EARTH_PROJECT\DAS_DATA'
fo_data_path = r'D:\RAIL4EARTH_PROJECT\DAS_DATA'

# Get all the tdms file names
file_names = get_files_in_dir(folder_path=fo_data_path, file_format='.tdms')
print('File names: ', file_names)

# Extract the timestamps from the file names
timestamps = extract_timestamp_from_name(file_names)

# Order the file names chronographically
timestamps, file_names = zip(*sorted(zip(timestamps, file_names)))
print('Ordered file names:', file_names)

# Get the data of the first file
first_file = file_names[0]

# Initialize the FindTrains class instance
file_instance = BaseFindTrains.create_instance(dir_path=fo_data_path, first_channel=4270, last_channel=4271, reader='silixa')
# Extract the properties of the TDMS file
properties = file_instance.extract_properties()

# From the selected files, extract the data
all_data = file_instance.get_data_per_file(file_names)
print('All data:', all_data)
'''
fo_data_path = r'C:\Projects\erju\data'
# Create database instance
create_db = CreateDatabase(fo_data_path=fo_data_path, acc_data_path=None, logbook_path=None)

fo_test = create_db.join_fo(channel_no=4270)

print(fo_test)


plot_timeseries(data_by_day=fo_test, date_str='20201111')
