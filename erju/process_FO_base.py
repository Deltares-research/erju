import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

from utils.file_utils import get_files_in_dir


class BaseFOdata:
    """
    Base class for finding files with trains passing in the middle
    of the channel array and extracting the data from the TDMS files
    """

    def __init__(self, dir_path: str, first_channel: int, last_channel: int):
        """
        Initialize the class instance

        Args:
            dir_path (str): The path to the folder containing the TDMS file
            first_channel (int): The first channel to be extracted
            last_channel (int): The last channel to be extracted
        """

        self.dir_path = dir_path
        self.first_channel = first_channel
        self.last_channel = last_channel
        self.properties = None
        self.data = None

    @staticmethod
    def create_instance(dir_path: str, first_channel: int, last_channel: int, reader: str):
        """
        Create an instance of the class

        Args:
            dir_path (str): The path to the folder containing the TDMS file
            first_channel (int): The first channel to be extracted
            last_channel (int): The last channel to be extracted
            reader (str): The name of the reader to be used

        Returns:
            instance (BaseFindTrains): The instance of the class
        """
        # If reader is 'silixa', use the SilixaFindTrains class
        if reader == 'silixa':
            # Import the SilixaFindTrains class here to avoid circular import
            from erju.process_FO_silixa import SilixaFOdata
            return SilixaFOdata(dir_path, first_channel, last_channel)

        # If reader is 'nptdms', use the NptdmsFindTrains class
        elif reader == 'nptdms':
            # Import the NptdmsFindTrains class here to avoid circular import
            from erju.process_FO_nptdms import NptdmsFOdata
            return NptdmsFOdata(dir_path, first_channel, last_channel)

        # If reader is 'optasense', use the OptasenseFindTrains class
        elif reader == 'optasense':
            # Import the OptasenseFindTrains class here to avoid circular import
            from erju.process_FO_optasense import OptasenseFOdata
            return OptasenseFOdata(dir_path, first_channel, last_channel)

        # If reader is none of the above, raise a ValueError
        else:
            raise ValueError(f'Invalid reader: {reader}')


    def extract_properties(self, file_name: str = None):
        """
        Extract the file properties and the measurement data as a dictionary and an array respectively

        Args:
            file_name (str): The name of the file to extract the properties from

        Returns:
            properties (dict): The extracted properties
        """
        # If the file_name is None, raise a ValueError, else the reader method will be called
        raise NotImplementedError('Subclass must implement abstract method')


    def extract_properties_per_file(self, file_name: str):
        """
        Extract the file properties as a dictionary for a given file in the directory.

        Args:
            file_name (str): The name of the file to extract the properties from

        Returns:
            properties (dict): The extracted properties
        """
        # If the file_name is None, raise a ValueError, else the reader method will be called
        raise NotImplementedError('Subclass must implement abstract method')


    def extract_data(self, file_name: str = None, first_channel: int = None, last_channel: int = None,
                     start_time: int = None, end_time: int = None, frequency: int = None):
        """
        Extract the data from the file

        Args:
            file_name (str): The name of the file to extract the data from

        Returns:
            data (np.ndarray): The extracted data
        """
        # If the file_name is None, raise a ValueError, else the reader method will be called
        raise NotImplementedError('Subclass must implement abstract method')


    def get_measurement_duration(self):
        """
        From the length of the data and the sampling frequency, calculate the measurement time
        for the time series.

        Args:
            data (np.ndarray): The data to extract the measurement time

        Returns:
            measurement_time (int): The measurement time
        """
        # Get the data length
        data_length = self.data.shape[1]
        # Get the sampling frequency
        fs = self.properties['SamplingFrequency[Hz]']

        # Calculate the time interval between samples
        time_interval = 1 / fs
        # Calculate the measurement duration
        measurement_duration = data_length * time_interval

        return self.measurement_duration


    def _calculate_cutoff_times(self, start_rate: float = 0.2, end_rate: float = 0.8):
        """
        Helper function to calculate the start and end times
        by removing the first and last 20% of the data (or any other percentage)

        Args:
            start_rate (float): The percentage to remove from the start time
            end_rate (float): The percentage to remove from the end time

        Returns:
            start_time (int): The start time index
            end_time (int): The end time index
        """

        # Pull the measurement time from the properties
        # in the culemborg data this is 30 seconds
        measurement_time = self.properties['measurement_time']
        # Calculate the start and end times
        start_time = round(measurement_time * start_rate) # 30 * 0.2 = 6
        end_time = round(measurement_time * end_rate) # 30 * 0.8 = 24



        return start_time, end_time


    def search_params(self):
        """
        Define the middle of the domain, the time window
        and the spatial window for the search

        Args:
            None

        Returns:
            scan_channel (int): The middle of the interest domain
            start_time (int): The start time of the search
            end_time (int): The end time of the search
        """

        # Define the middle of the interest domain to scan there
        scan_channel = int(np.mean([self.first_channel, self.last_channel]))

        # Define the time window for the search
        #start_time, end_time = self._calculate_cutoff_times(start_rate=0.3, end_rate=0.7)
        # 0 and 1 means the start and end of the data is the same as 0 and 30 seconds
        start_time, end_time = self._calculate_cutoff_times(start_rate=0, end_rate=1)

        return scan_channel, start_time, end_time


    def signal_averaging(self, plot: bool = False, save_to_path: str = None, channel: int = None, threshold: int = 500):
        """
        Look in a folder for all the TDMS files and extract the mean signal
        value from the search parameters

        Args:
            plot (bool): Plot the mean signal values
            save_to_path (str): The path to save the figure
            channel (int): The channel to extract the mean signal from
            threshold (int): The threshold value for posterior filtering

        Returns:
            mean_signal (np.array): The mean signal values
        """

        # Get the search parameters
        scan_channel, start_time, end_time = self.search_params()

        # Check if user defined the channel to extract the mean signal from
        # If he did, use that channel, if not, use the scan channel
        if channel is not None:
            relative_scan_channel = channel
        else:
            relative_scan_channel = scan_channel - self.first_channel

        # Get the tdms files in the directory
        tdms_files = get_files_in_dir(folder_path=self.dir_path, file_format='.tdms')
        # Build the complete path to the files
        tdms_files = [os.path.join(self.dir_path, file) for file in tdms_files]

        mean_signal = []
        # Loop through all the TDMS files in the directory
        for file in tdms_files:
            # Get the mean signal for each file in the relative scan channel
            # and using the start_index and end_index, and get them in a list
            data = self.extract_data(file, relative_scan_channel, relative_scan_channel + 1, start_time, end_time)
            mean_signal.append(np.mean(np.abs(data)))
        # Convert the list to a numpy array
        mean_signal = np.array(mean_signal)

        # Plot the mean signal if plot is True
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(mean_signal, label='Mean Signal')
            # Add a black dotted horizontal line for the threshold
            plt.axhline(y=threshold, color='black', linestyle='--', label='Threshold')
            plt.xlabel('File Index')
            plt.ylabel('Mean Signal Value')
            plt.title('Mean signal values for all files in the directory')
            plt.legend()

            # Save the figure
            file_name_suffix = 'Mean_Signal_silixa'
            full_file_name = f'{file_name_suffix}.jpg'
            save_path = os.path.join(save_to_path, full_file_name)
            plt.savefig(save_path, dpi=300)
            print('Mean Signal figure saved')

        return mean_signal


    def get_files_above_threshold(self, mean_signal: np.ndarray, threshold: float):
        """
        Get the list of file names based on a threshold value

        Args:
            mean_signal (np.ndarray): The mean signal values
            threshold (float): The threshold value

        Returns:
            selected_files (list): The list of file names above the threshold
        """

        # Get the list of TDMS files in the directory
        tdms_files = [f for f in os.listdir(self.dir_path) if f.endswith('.tdms')]

        # Filter files based on the threshold
        selected_files = [file_name for file_name, mean_value in zip(
            tdms_files, mean_signal) if mean_value >= threshold]

        return selected_files


    def get_data_per_file(self, selected_files: list, resample: bool = False, new_n_channels=None,
                          new_sampling_frequency=None):
        """
        Extract measurement data for selected files. This is done per file
        using the same 30 seconds time window present in the file itself.
        Data resampling can be turned on or off

        Args:
            selected_files (list): The list of selected files
            resample (bool): Resample the data
            new_n_channels (int, optional): The new number of channels
            new_sampling_frequency (int, optional): The new sampling frequency

        Returns:
            all_selected_data (dict): The dictionary of extracted data
        """

        # Initialize an empty dictionary to store the selected data
        all_selected_data = {}

        # Loop through the selected files and extract the data
        for file_name in selected_files:
            #print(f'Extracting data from {file_name} in get_data_per_file')
            data = self.extract_data(file_name)

            # Resample the data if resampling is True
            if resample == True:
                data = self.resample_data(data, new_n_channels, new_sampling_frequency)

            # Store the data in the dictionary
            all_selected_data[file_name] = data

        return all_selected_data


    def get_data_with_window(self, file_name: str, window_before: int, window_after: int,
                             resample: bool = False, new_n_channels=None, new_sampling_frequency=None):
        """
        Extract the measurement data from a selected file, but also include a time window
        before and after the train passes. This is done by concatenating the data from the
        files before and after the selected file, until the time window is filled.
        It can also resample data if needed.

        Args:
            file_name (str): The name of the file to extract the data from
            window_before (int): The time window (in seconds) before the train passes
            window_after (int): The time window (in seconds) after the train passes
            resample (bool): Resample the data
            new_n_channels (int, optional): The new number of channels
            new_sampling_frequency (int, optional): The new sampling frequency

        Returns:
            signal_data (np.ndarray): The resampled data
        """

        # Get the number of channels from the properties
        n_channels = self.properties['MeasureLength[m]']

        # If resampling is activated, get new shape of the data
        if resample == True and new_n_channels is not None:
            n_channels_resampled = new_n_channels
        else:
            n_channels_resampled = n_channels

        # Create an empty np.array to store the concatenated data
        signal_data = np.empty(shape=(n_channels_resampled, 0))

        # Get the list of TDMS files in the directory
        all_file_names = get_files_in_dir(folder_path=self.dir_path, file_format='.tdms')

        # Find the number of files to concatenate to match the time window before and after (each file is 30 seconds)
        n_files_before = int(np.ceil(window_before / 30))
        n_files_after = int(np.ceil(window_after / 30))

        # From all the file names, find the index of the selected file
        file_index = all_file_names.index(file_name)

        # Check if there are enough files before and after the selected file
        if file_index < n_files_before or file_index + n_files_after >= len(all_file_names):
            max_before = file_index * 30
            max_after = (len(all_file_names) - file_index - 1) * 30
            raise ValueError(f"Not enough files to cover the requested time window. "
                             f"The maximum possible window_before is {max_before} seconds, "
                             f"and the maximum possible window_after is {max_after} seconds.")

        # Get the files before and after the selected file
        file_names_before = all_file_names[file_index - n_files_before:file_index]
        file_names_after = all_file_names[file_index + 1:file_index + n_files_after + 1]
        selected_files = file_names_before + [file_name] + file_names_after

        # Use the extract_data function to extract the data from the selected files
        for file in selected_files:
            # Extract the data one file at a time
            data = self.extract_data(file)

            # Resample the data if resampling is True
            if resample == True:
                data = self.resample_data(data, new_n_channels, new_sampling_frequency)

            # Fill the signal_data array with the extracted data and concatenate for each file
            signal_data = np.concatenate((signal_data, data), axis=1)

        return signal_data


    def save_txt_with_file_names(self, save_to_path: str, selected_files: list, file_names: list,
                                 include_indexes: bool = True):
        """
        Print the name of the files which are above the threshold in a txt file

        Args:
            save_to_path (str): The path to save the txt file
            selected_files (list): The list of selected files
            file_names (list): The list of all file names
            include_indexes (bool): Whether to include the indexes of the files in the txt file

        Returns:
            None
        """
        # Create the full file path
        file_path = os.path.join(save_to_path, 'files_with_trains.txt')

        # Open the txt file
        with open(file_path, 'w') as f:
            # Loop through the selected files and write the name of the files in the txt file
            for file_name in selected_files:
                if include_indexes:
                    index = file_names.index(file_name)
                    f.write(f'Index: {index} -> {file_name}\n')
                else:
                    f.write(f'{file_name}\n')

        print(f'The file files_with_trains.txt has been created at {save_to_path}')

        return None


    def plot_array_channels(self, file_to_plot: str,  save_to_path: str = None, save_figure: bool = False,
                            window_before: int = 30, window_after: int = 30,
                            resample: bool = False, new_sampling_frequency: int = 100):
        """
        Plot an array of channels as an image plot, including a time window before and after the train passes.
        The data can also be resampled if needed.

        I THINK THIS IS NO LONGER NEEDED WITH THE NEW PLOT 2D BUFFER FUNCTION IN THE PLOT DATA CLASS.

        Args:
            file_to_plot (str): The name of the file to plot
            save_to_path (str): The path to save the figure
            save_figure (bool): Save the figure as a jpg file
            window_before (int): The time window (in seconds) before the train passes
            window_after (int): The time window (in seconds) after the train passes
            resample (bool): Resample the data
            new_sampling_frequency (int): The new sampling frequency

        Returns:
            None
        """
        # If resample is True, resample the data
        if resample:
            data = self.get_data_with_window(file_name=file_to_plot, window_before=window_before, window_after=window_after,
                                         resample=resample, new_sampling_frequency=new_sampling_frequency)
        else:
            data = self.get_data_with_window(file_name=file_to_plot, window_before=window_before, window_after=window_after)

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(10, 15))

        # Transpose the data and take the absolute value
        data = np.abs(data.T)

        # Plot the data as an image plot
        im = ax.imshow(data, aspect='auto', cmap='jet', vmax=data.max() * 0.30)
        ax.set_xlabel('Channel count')
        ax.set_ylabel('Time [s]')

        # Set the number of ticks based on the dimensions of the data
        num_time_points = data.shape[0]
        num_channels = data.shape[1]

        # Set the x-ticks and their labels
        x_ticks = np.linspace(0, num_channels - 1, num=min(10, num_channels))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{int(x)}' for x in x_ticks])

        # Set the y-ticks and their labels
        y_ticks = np.linspace(0, num_time_points - 1, num=min(10, num_time_points))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{int(y / 1000)}' for y in y_ticks])

        # Show colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity')

        # If save_figure is True, save the figure with a specific file name
        if save_figure:
            file_name_suffix = 'Figure_2D'
            full_file_name = f'{file_name_suffix}_{file_to_plot}.jpg'
            save_path = os.path.join(save_to_path, full_file_name) if save_to_path else os.path.join(save_to_path, full_file_name)
            plt.savefig(save_path, dpi=300)
        plt.close()


    def resample_data(self, data: np.array, new_n_channels: int = None, new_sampling_frequency: int = None):
        """
        Resample the data to a new number of channels and/or a new sampling frequency using scipy.signal.resample

        Args:
            data (np.ndarray): The data to be resampled
            new_n_channels (int): The new number of channels
            new_sampling_frequency (int): The new sampling frequency

        Returns:
            resampled_data (np.ndarray): The resampled data
        """
        # Make a copy of the data to avoid modifying the original data
        data_resampled = np.copy(data)

        # Resample the channels if new_n_channels is specified
        if new_n_channels is not None:
            data_resampled = signal.resample(data_resampled, new_n_channels, axis=0)

        # Resample the time if new_sampling_frequency is specified
        if new_sampling_frequency is not None:
            n_samples = int(data.shape[1] * (new_sampling_frequency / self.properties['SamplingFrequency[Hz]']))
            data_resampled = signal.resample(data_resampled, n_samples, axis=1)

        return data_resampled


    def detect_FO_events_sta_lta(self, FO_signal: pd.DataFrame, window_buffer: int, nsta: int, nlta: int,
                                    trigger_on: float, trigger_off: float):
        """
        Detect events using the STA/LTA method and create windows around these events. This method
        uses the recursive_sta_lta function from ObsPy to compute the STA/LTA ratio and the trigger_onset
        function to detect the events. A buffer is added to the start and end of each event to create the
        windows. The windows are then returned as indices and times. This method is used to detect events
        in the FO data

        Args:
            FO_signal (pd.DataFrame): The dataframe containing the data from the location_name
            window_buffer (int): The buffer to add to the start and end of each window
            nsta (int): The number of samples in the STA window
            nlta (int): The number of samples in the LTA window
            trigger_on (float): The trigger on threshold
            trigger_off (float): The trigger off threshold

        Returns:
            windows_indices (list): A list of tuples containing the start and end indices of each window
            windows_times (list): A list of tuples containing the start and end times of each window
        """

        # Extract only the signal from the dataframe
        signal_data = FO_signal["signal"]

        # Compute the STA/LTA ratio
        sta_lta_ratio = recursive_sta_lta(signal_data, nsta, nlta)

        # Detect events using the trigger_onset function from ObsPy
        events = trigger_onset(sta_lta_ratio, trigger_on, trigger_off)

        # Create the windows around the detected events
        windows_indices = []
        windows_times = []

        # For each event, create a window around it
        for event in events:
            # Extend the window by the window_size_extension at the start and end
            start_index = max(0, event[0] - window_buffer)
            end_index = min(len(FO_signal) - 1, event[1] + window_buffer)
            # Compute the corresponding times for the start and end indices
            start_time = FO_signal["time"].iloc[start_index]
            end_time = FO_signal["time"].iloc[end_index]
            # Append the window to the lists
            windows_indices.append((start_index, end_index))
            windows_times.append((start_time, end_time))

        return windows_indices, windows_times, sta_lta_ratio


    def plot_fo_signal_and_windows(self, fo_data: pd.DataFrame, windows_indices: list, nsta: int = None,
                                   nlta: int = None, trigger_on: float = None, trigger_off: float = None):
        """
        Plot the FO signal, STA/LTA ratio, and detected event windows with shaded areas.

        Args:
            fo_data (pd.DataFrame): The dataframe containing FO data with 'time' and 'signal' columns
            windows_indices (list): A list of tuples containing the start and end indices of each window
            nsta (int): The number of samples in the STA window
            nlta (int): The number of samples in the LTA window
            trigger_on (float): The trigger on threshold
            trigger_off (float): The trigger off threshold
        """
        # Use 'signal' column from the FO data
        signal = fo_data['signal'].values
        time_values = fo_data['time'].values

        if nsta and nlta and trigger_on and trigger_off:
            # Compute the STA/LTA ratio if parameters are provided
            sta_lta_ratio = recursive_sta_lta(signal, nsta, nlta)
        else:
            sta_lta_ratio = None

        # Plot the results
        plt.figure(figsize=(15, 8))

        # Plot the FO data (signal vs. time)
        plt.subplot(2, 1, 1)
        plt.plot(time_values, signal, label='FO Signal', color='black')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Raw FO Signal')
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

