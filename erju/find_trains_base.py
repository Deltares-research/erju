import os
import numpy as np
import matplotlib.pyplot as plt


class BaseFindTrains:
    """
    Base class for finding trains
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
            from erju.find_trains_silixa import SilixaFindTrains
            return SilixaFindTrains(dir_path, first_channel, last_channel)
        # If reader is 'nptdms', use the NptdmsFindTrains class
        elif reader == 'nptdms':
            # Import the NptdmsFindTrains class here to avoid circular import
            from erju.find_trains_nptdms import NptdmsFindTrains
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
        start_time, end_time = self._calculate_cutoff_times()

        return scan_channel, start_time, end_time

    def signal_averaging(self, plot: bool = False):
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
            plt.title('Mean Signal Line Plot')
            plt.legend()

            # Save the figure
            file_name_suffix = 'Mean_Signal_silixa'
            full_file_name = f'{file_name_suffix}.jpg'
            save_path = os.path.join('..', 'test', 'test_output', full_file_name)
            plt.savefig(save_path, dpi=300)
            print('Mean Signal figure saved')

        return mean_signal

    def get_files_above_threshold(self, mean_signal: np.ndarray, threshold: float):
        """
        Get the list of file names based on a threshold value.

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

    def get_data(self, selected_files: list):
        """
        Extract measurement data for selected files.

        Args:
            selected_files (list): The list of selected files

        Returns:
            all_selected_data (dict): The dictionary of extracted data
        """

        # Initialize an empty dictionary to store the selected data
        all_selected_data = {}

        # Loop through the selected files and extract the data
        for file_name in selected_files:
            data = self.extract_data(file_name)
            # Store the data in the dictionary
            all_selected_data[file_name] = data

        return all_selected_data
