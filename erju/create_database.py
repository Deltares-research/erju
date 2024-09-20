import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import pickle

from erju.create_accel_windows import AccelDataTimeWindows
from erju.process_FO_base import BaseFOdata
from utils.file_utils import get_files_in_dir, extract_timestamp_from_name
from matplotlib.dates import DateFormatter


class CreateDatabase:
    """
    A class used to create a database of FO and accelerometer data.
    It will create a database with at least 4 elements inside: the first element is
    the start time of the event, the second element is the end time of the event, the third
    element is the FO signal and the fourth, fifth and so on, will be all the accelerometer
    signals, with as many elements as sensors we have.
    """

    def __init__(self, fo_data_path: str, acc_data_path: str = None, logbook_path: str = None):
        """
        Initialize the class instance to perform the database creation.
        Key elements are the paths to the FO and accelerometer data files and
        using the AccelDataTimeWindows and BaseFindTrains classes for the operations.

        Args:
            fo_data_path (str): The path to the folder containing the FO data files.
            acc_data_path (str): The path to the folder containing the accelerometer data files.
            logbook_path (str): The path to the logbook file.
        """
        self.fo_data_path = fo_data_path
        self.acc_data_path = acc_data_path
        self.logbook_path = logbook_path
        self.window_indices = None
        self.window_times = None
        self.database = None

    ##########################################################################################################

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
        nsta = int(nsta * 1000)  # convert to seconds with a fz of 1000 Hz
        nlta = int(nlta * 1000)  # convert to seconds with a fz of 1000 Hz
        self.window_indices, self.window_times = accel_windows.detect_accel_events_sta_lta(accel_data=accel_data_df,
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


    #############################################################################################################
    # The previous 3 functions look like I dont use anymore. Keep them there in case I do need them later.
    # From here on out, I will use the following functions to create the database.


    def extract_accel_windows(self, file_name: str, nsta: int = 1, nlta: int = 8):
        """
        Extracts time windows and indices from the accelerometer data using the STA/LTA method. The function
        returns the indices of the detected events and the start and end times of the detected events. It also
        filters the detected events with the logbook data. The function returns the indices, times and the
        accelerometer data for the file.

        Args:
            file_name (str): The name of the accelerometer file to extract the data from.
            nsta (int): The length of the average short time average window in seconds.
            nlta (int): The length of the average long time average window in seconds.

        Returns:
            accel_windows_indices (list): Indices of detected events.
            accel_windows_times (list): Start and end times of detected events.
        """
        # With the accelerometer data path, create the AccelDataTimeWindows instance
        accel_windows = AccelDataTimeWindows(accel_data_path=self.acc_data_path,
                                             logbook_path=self.logbook_path)

        # Extract the accelerometer data from the file with default number of columns (3)
        accel_data_per_file = accel_windows.extract_accel_data_from_file(file_name=file_name)

        # Scale the nsta and nlta to milliseconds
        nsta = int(nsta * 1000)  # convert to milliseconds
        nlta = int(nlta * 1000)  # convert to milliseconds

        # Create the accelerometer windows indices and times with the STA/LTA method
        accel_windows_indices, accel_windows_times = accel_windows.detect_accel_events_sta_lta(
            accel_data=accel_data_per_file,
            nsta=nsta,
            nlta=nlta,
            trigger_on=1.5,
            trigger_off=1
        )

        # Filter the accelerometer windows with the logbook
        accel_windows_indices, accel_windows_times = accel_windows.filter_windows_with_logbook(
            time_buffer=15,
            window_indices=accel_windows_indices,
            window_times=accel_windows_times
        )

        return accel_windows_indices, accel_windows_times, accel_data_per_file


    def find_matching_fo_files(self, accel_windows_times: list, buffer_seconds: int = 35, file_time_coverage: int = 30):
        """
        Finds the FO files that match the given time windows. We give the accelerometer time windows a buffer to
        account for the 30 seconds of time coverage of each FO file. We also check if the FO files cover the entire
        time window. If the coverage is incomplete, we append an empty list.

        Args:
            accel_windows_times (list): Start and end times of detected events.
            buffer_seconds (int): Buffer time in SECONDS before and after each window.
            file_time_coverage (int): The time coverage of each file in SECONDS.


        Returns:
            fo_file_names_per_window (list): List of lists containing file names for each time window.
        """
        # Get the list of all the file names in the FO folder
        fo_file_names = get_files_in_dir(folder_path=self.fo_data_path, file_format='.tdms')

        # Extract the timestamps from the file names
        fo_timestamps = extract_timestamp_from_name(fo_file_names)

        # Initialize an empty list to store the file names for each time window
        fo_file_names_per_window = []

        # Loop through the windows and find the files that match the time window
        for window_time in accel_windows_times:
            # Get the start and end time of the window
            start_time = window_time[0] - timedelta(seconds=buffer_seconds)
            end_time = window_time[1] + timedelta(seconds=buffer_seconds)

            # Find matching files within the time window
            matching_files = [
                fo_file_names[i] for i, timestamp in enumerate(fo_timestamps)
                if start_time <= timestamp <= end_time
            ]

            # Check if the matching files cover the entire accelerometer time window
            if matching_files:
                # Sort the matching files by their timestamps
                sorted_files = sorted(matching_files, key=lambda name: extract_timestamp_from_name([name])[0])

                # Get the start and end times of the fo files
                file_times = [(extract_timestamp_from_name([file])[0],
                               extract_timestamp_from_name([file])[0] + timedelta(seconds=file_time_coverage))
                               for file in sorted_files]

                # Determine the coverage
                coverage_start = file_times[0][0]
                coverage_end = file_times[-1][1]

                # Check if the coverage completely includes the time window (without buffer)
                if coverage_start <= window_time[0] and coverage_end >= window_time[1]:
                    fo_file_names_per_window.append(matching_files)
                else:
                    # If coverage is incomplete, append an empty list
                    fo_file_names_per_window.append([])
            else:
                # If no files match, append an empty list
                fo_file_names_per_window.append([])

        return fo_file_names_per_window


    def extract_and_join_fo_data(self, fo_file_names:str, channel_no: int = 4270):
        """
        This function uses the BaseFindTrains class to extract the data from the FO files.
        It uses the get_data_per_file method to extract the data from each file and store it in a
        dataframe with columns TIME and SIGNAL.

        Args:
            fo_file_names (list): The list of file names to extract the data from.
            channel_no (int): The channel number to extract the data from.

        Returns:
            data_df (pd.DataFrame): The DataFrame containing the time and signal data.
        """
        # The FO data path is already defined in the CreateDatabase class instance
        # Get the list of file names in the FO data path
        #file_names = get_files_in_dir(folder_path=self.fo_data_path, file_format='.tdms')
        file_names = fo_file_names

        # Create an instance of the BaseFindTrains class to extract the data
        file_instance = BaseFOdata.create_instance(dir_path=self.fo_data_path,
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
            #print(f'Processing file: {file}')

            # Extract the data from the file
            signal_data_dict = file_instance.get_data_per_file([file])
            # Get the data out of the dictionary for later be able to create a dataframe
            # the dictionary only has one key, the name of the file, and inside it there is
            # a ndarray of size (1, n_samples_per_channel)
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


    def create_pickle_database(self, channel_no: int = 4270, save_path: str = None):
        """
        This function takes as input the channel number of interest, and creates a pickle database from the
        accelerometer time windows and the fo signal data corresponding to the time windows. The pickle database
        files are saved in a day by day basis.

        Args:
            channel_no (int): The channel number of interest from which all fo data is extracted.
            save_path (str): The path to save the pickle files. If None, the files are saved in the current directory.
        """

        # From the accelerometer data, get the name of all the files in the folder
        accel_file_names = get_files_in_dir(folder_path=self.acc_data_path, file_format='.asc', keep_extension=False)
        # Initialize a counter for file numbering
        file_counter = 1

        # Loop through the accelerometer files one at a time
        for accel_file in accel_file_names:
            # Print the current file being processed
            print(f"Processing file: {accel_file} ....................................................................")

            # Get accelerometer time windows and indices for a specific file
            accel_window_indices, accel_window_times, accel_data_per_file = self.extract_accel_windows(file_name=accel_file)
            # Find the matching FO file names for each time window
            fo_file_names_per_window = self.find_matching_fo_files(accel_window_times)

            # Loop through each time window, join the FO from the selected files, crop the signal to the
            # exact time of the accelerometer window and extract the FO signal.
            for i, accel_window in enumerate(accel_window_times):
                # Check if fo_file_names_per_window is empty
                if not fo_file_names_per_window[i]:
                    print(f"No FO data available for window {i+1}/{len(accel_window_times)}")
                    continue

                # Join the selected list of FO files into a single signal dataframe
                fo_data_df = self.extract_and_join_fo_data(fo_file_names=fo_file_names_per_window[i],
                                                           channel_no=channel_no)

                # Trim the FO signal data to match exactly the accelerometer time window
                start_time = accel_window[0]
                end_time = accel_window[1]

                # Extract from the FO signal data only the data that is within the time window
                fo_data_in_window = fo_data_df[(fo_data_df['time'] >= start_time) &
                                               (fo_data_df['time'] <= end_time)].copy()

                # Trim the accelerometer data to match exactly the FO signal data
                accel_data_in_window = accel_data_per_file[(accel_data_per_file['T(ms)'] >= start_time) &
                                                              (accel_data_per_file['T(ms)'] <= end_time)].copy()


                # Create a dictionary with the data to save in the pickle file
                data_dict = {
                    'date': start_time.date(),
                    'accel_file': accel_file,
                    'window_index': i,
                    'start_time': start_time,
                    'end_time': end_time,
                    'accel_data': accel_data_in_window,
                    'fo_data': fo_data_in_window  # Add FO data to the dictionary
                }

                # Sanitize the start_time to create a valid file name
                safe_start_time = start_time.strftime('%Y%m%d_%H%M%S%f')  # Format to safe filename
                pickle_file_name = f'{file_counter}_{safe_start_time}.pkl'

                # Increment the file counter
                file_counter += 1

                # Save the data dictionary as a pickle file
                with open(os.path.join(save_path, pickle_file_name), 'wb') as file:
                    pickle.dump(data_dict, file)

                # Print the file name that was saved
                print(f"Saved pickle file: {pickle_file_name}, for window {i+1}/{len(accel_window_times)}")


    def extract_all_events(self, selected_channel: int = 4270, window_size: int = 30, step_size: int = 30):
        """
        This function takes as input a given channel, and scans through all the FO data using
        windows of 90 seconds. This means that for files of 30 seconds, we will consider 3
        files at a time. The step of the window is 30 seconds (1 file). We then extract the signal for
        the given channel in the 3 files and stitch them together to create a single signal. From
        that single signal we extract the events using the STA/LTA method. We then save the events
        in a pickle file.

        Args:
            selected_channel (int): The channel number of interest.
            window_size (int): The size of the window in seconds.
            step_size (int): The step size of the window in seconds.

        Returns:
            None
        """
        # Get all the files in the directory
        file_names = get_files_in_dir(folder_path=self.fo_data_path, file_format='.tdms')

        # Create an instance of the BaseFindTrains class to extract the data
        file_instance = BaseFOdata.create_instance(dir_path=self.fo_data_path,
                                                   first_channel=selected_channel,
                                                   last_channel=selected_channel,
                                                   reader='silixa')

        # Join all the FO data from a single channel into a single signal
        all_data = self.extract_and_join_fo_data(fo_file_names=file_names, channel_no=selected_channel)

        # Find the events with sta/lta method from all the data
        window_indices, window_times, stalta_ratio = file_instance.detect_FO_events_sta_lta(FO_signal=all_data,
                                                                              window_buffer=20*1000,
                                                                              nsta=50,
                                                                              nlta=10000,
                                                                              trigger_on=14,
                                                                              trigger_off=1)


        return window_indices, window_times, stalta_ratio




##### TEST THE CODE ###################################################################################################
# Define the paths to the FO and accelerometer data
fo_data_path = r'C:\Projects\erju\data\culemborg\das_20201120'
acc_data_path = r'C:\Projects\erju\data\accel_data'
logbook_path = r'C:\Projects\erju\data\logbook_20201109_20201111.xlsx'
path_save_database = r'C:\Projects\erju\outputs'

# Create an instance of the CreateDatabase class
database = CreateDatabase(fo_data_path=fo_data_path)

fo_file_names = get_files_in_dir(folder_path=fo_data_path, file_format='.tdms')

# Join all the FO data from a single channel into a single signal
all_data = database.extract_and_join_fo_data(fo_file_names=fo_file_names,
                                             channel_no=4270)

# Find the events with sta/lta method
window_indices, window_times, stalta_ratio = database.extract_all_events(selected_channel=4270)

print(all_data.head())

# Make a plot with 2 subplots in the vertical direction, on top all_data and on the bottom the stalta_ratio
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
ax[0].plot(all_data['time'], all_data['signal'], color='blue')
# Add a shaded area for the detected events using the window_indices or window_times
for i, window in enumerate(window_times):
    ax[0].axvspan(window[0], window[1], color='gray', alpha=0.5)

ax[0].set_title('FO Signal')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Signal')

ax[1].plot(all_data['time'], stalta_ratio, color='red')
# Add a horizontal line at the trigger_on and trigger_off values
ax[1].axhline(y=14, color='green', linestyle='--')
ax[1].axhline(y=1, color='red', linestyle='--')
ax[1].set_title('STA/LTA Ratio')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Ratio')

# Format the x-axis to show the time
date_form = DateFormatter("%H:%M:%S")
ax[0].xaxis.set_major_formatter(date_form)
ax[1].xaxis.set_major_formatter(date_form)

plt.tight_layout()
plt.show()




# Create a pickle database
#database.create_pickle_database(channel_no=4270, save_path=path_save_database)

