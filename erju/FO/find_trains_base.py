import os
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal


class BaseFindTrains:
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
            from erju.FO.find_trains_silixa import SilixaFindTrains
            return SilixaFindTrains(dir_path, first_channel, last_channel)
        # If reader is 'nptdms', use the NptdmsFindTrains class
        elif reader == 'nptdms':
            # Import the NptdmsFindTrains class here to avoid circular import
            from erju.FO.find_trains_nptdms import NptdmsFindTrains
            return NptdmsFindTrains(dir_path, first_channel, last_channel)
        # If reader is not 'silixa' or 'nptdms', raise a ValueError
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
        measurement_time = self.properties['measurement_time']
        # Calculate the start and end times
        start_time = round(measurement_time * start_rate)
        end_time = round(measurement_time * end_rate)
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
        start_time, end_time = self._calculate_cutoff_times(start_rate=0.3, end_rate=0.7)

        return scan_channel, start_time, end_time

    def signal_averaging(self, plot: bool = False, save_to_path: str = None):
        """
        Look in a folder for all the TDMS files and extract the mean signal
        value from the search parameters

        Args:
            plot (bool): Plot the mean signal

        Returns:
            mean_signal (np.array): The mean signal values
        """

        # Get the search parameters
        scan_channel, start_time, end_time = self.search_params()

        # Adjust the scan_channel to be relative to the first_channel
        relative_scan_channel = scan_channel - self.first_channel

        mean_signal = []
        # Loop through all the TDMS files in the directory
        for file_name in os.listdir(self.dir_path):
            # Get the mean signal for each file in the relative scan channel
            # and using the start_index and end_index, and get them in a list
            data = self.extract_data(file_name, relative_scan_channel, relative_scan_channel + 1, start_time, end_time)
            mean_signal.append(np.mean(np.abs(data)))
        # Convert the list to a numpy array
        mean_signal = np.array(mean_signal)

        # Plot the mean signal if plot is True
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(mean_signal, label='Mean Signal')
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
refund woul
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
        all_file_names = self.get_file_list()

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

    def save_txt_with_file_names(self, save_to_path: str, selected_files: list, file_names: list):
        """
        Print the name of the files which are above the threshold in a txt file

        Args:
            save_to_path (str): The path to save the txt file
            selected_files (list): The list of selected files
            file_names (list): The list of all file names

        Returns:
            None
        """
        # Create the full file path
        file_path = os.path.join(save_to_path, 'files_with_trains.txt')

        # Open the txt file
        with open(file_path, 'w') as f:
            # Loop through the selected files and write the name of the files in the txt file
            for file_name in selected_files:
                index = file_names.index(file_name)
                f.write(f'Index: {index}-> {file_name}\n')

        print(f'The file files_with_trains.txt has been created at {save_to_path}')

        return None

    def get_file_list(self):
        """
        Get a list of unique file names inside the dir_path folder. For this
        we look into one specific file format (.asc) and remove the extension.

        Args:
            None

        Returns:
            file_list (list): List of file names in the folder without extensions
        """

        # Get the list of files in the folder with .asc extension
        self.tdms_files = [f for f in os.listdir(self.dir_path) if f.endswith('.tdms')]

        return self.tdms_files


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
