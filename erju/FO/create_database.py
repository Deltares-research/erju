import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta

from erju.Accel.create_data_windows import AccelDataTimeWindows
from erju.FO.find_trains_base import BaseFindTrains
from utils.utils import get_files_in_dir, extract_timestamp_from_name
from collections import defaultdict
from matplotlib.dates import DateFormatter

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


    def get_fo_data(self, channel_no: int = 4270):
        """
        This function uses the BaseFindTrains class to extract the data from the FO files.
        It uses the get_data_per_file method to extract the data from each file and store it in a
        dataframe with columns TIME and SIGNAL.

        Args:
        channel_no (int): The channel number to extract the data for. If you just want to extract data from one channel,
        use the same channel number for first_channel and last_channel when creating the BaseFindTrains instance.
        """
        # The FO data path is already defined in the CreateDatabase class instance
        # Get the list of file names in the FO data path
        file_names = get_files_in_dir(folder_path=self.fo_data_path, file_format='.tdms')

        # Create an instance of the BaseFindTrains class to extract the data
        file_instance = BaseFindTrains.create_instance(dir_path=self.fo_data_path,
                                                       first_channel=channel_no,
                                                       last_channel=channel_no,
                                                       reader='silixa')
        # Get the properties in general for the instance
        file_instance.extract_properties()
        sampling_frequency = file_instance.properties['SamplingFrequency[Hz]']
        no_data_points = file_instance.properties['n_samples_per_channel']
        # Calculate the deltatime for each data point as 1/fz
        delta_time = 1 / sampling_frequency

        # Initialize an empty list to store the dataframes
        dataframes = []

        # Loop through the files and extract the data
        for file in file_names:
            print(f'Processing file: {file}')

            # Extract the data from the file
            signal_data_dict = file_instance.get_data_per_file([file])
            # Get the data out of the dictionary for later be able to create a dataframe
            # the dictionary only has one key, the name of the file, and inside it there is
            # a ndarrau of size (1, n_samples_per_channel)
            signal_data = signal_data_dict[file]
            # Delete the last column of the signal data
            signal_data = signal_data[:, :-1]

            # Get the properties for the specific file
            properties = file_instance.extract_properties_per_file(file)
            initial_time = properties['GPSTimeStamp']
            # Calculate the time for each data point
            time = [initial_time + timedelta(seconds=i * delta_time) for i in range(no_data_points)]

            # Create an empty dataframe with two columns called 'time' and 'signal'
            # put the time in the first column and the signal in the second column
            # time is a list of datetime objects and signal is a numpy array (1, n_samples_per_channel)
            # after each file is processed, the data is concatenated in the dataframe
            data_df = pd.DataFrame({'time': time, 'signal': signal_data[0]})

            # Append the dataframe to the list of dataframes
            dataframes.append(data_df)

        # Concatenate all dataframes into a single dataframe
        data_df = pd.concat(dataframes, ignore_index=True)

        return data_df





def plot_data_for_date(data_df: pd.DataFrame, date_str: str):
    """
    Plots the data for a given date.

    Args:
    data_df (pd.DataFrame): The DataFrame containing the time and signal data.
    date_str (str): The date string in the format 'YYYY-MM-DD' to filter the data.
    """
    # Convert the date string to a datetime object
    date = pd.to_datetime(date_str)

    # Filter the DataFrame for the given date
    filtered_df = data_df[(data_df['time'].dt.date == date.date())]

    if filtered_df.empty:
        print(f"No data available for the date {date_str}")
        return

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_df['time'], filtered_df['signal'], label='Signal')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title(f'Signal Data for {date_str}')
    plt.legend()
    plt.grid(True)

    # Set the x-axis to show time only
    date_format = DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(date_format)

    plt.show()

# Example usage:
# Assuming data_df is your DataFrame and you want to plot data for '2023-07-15'
# plot_data_for_date(data_df, '2023-07-15')



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

# Create an instance of the CreateDatabase class
database = CreateDatabase(fo_data_path=fo_data_path, acc_data_path=None, logbook_path=None)
# Get the data for channel 4270
data = database.get_fo_data(channel_no=4270)

# Plot the data
plot_data_for_date(data_df=data, date_str='2020-11-22')
# Plot the data
plot_data_for_date(data_df=data, date_str='2020-11-20')
# Plot the data
plot_data_for_date(data_df=data, date_str='2020-11-11')

