import os
import numpy as np
from nptdms import TdmsFile as td
import matplotlib.pyplot as plt


class FindTrains:
    """
    Read the TDMS iDAS file and find the files
    that contain train measurements
    """

    def __init__(self, dir_path, first_channel, last_channel):
        """
        Initialize the class instance

        @param dir_path: path to the directory containing the TDMS files
        @param first_channel: first channel to be extracted
        @param last_channel: last channel to be extracted
        """

        self.dir_path = dir_path
        self.first_channel = first_channel
        self.last_channel = last_channel
        self.properties = None
        self.data = None

    def extract_properties(self, file_name=None):
        """
        Extract the file properties and the measurement data
        as a dictionary and an array respectively

        @param file_name: name of the TDMS file to extract properties from

        @return: properties_dict
        """

        # Get a list of all TDMS files in the directory
        tdms_files = [f for f in os.listdir(self.dir_path) if f.endswith('.tdms')]

        # If no file name is specified, use the first file
        if file_name is None:
            file_name = tdms_files[0]

        # Construct the full file path
        file_path = os.path.join(self.dir_path, file_name)

        print('Extracting the properties...')
        with td.read(file_path) as tdms_file:
            # Get the properties of the TDMS file
            properties = tdms_file.properties

            # List of property names
            property_names = ['name', 'SamplingFrequency[Hz]', 'SpatialResolution[m]', 'StartPosition[m]',
                              'MeasureLength[m]', 'Start Distance (m)', 'Stop Distance (m)', 'PeakVoltage[V]',
                              'Pulse 2 Delay (ns)', 'PulseWidth[ns]', 'OffsetLength', 'Reference Level 1',
                              'Reference Level 2', 'Reference Level 3', 'FibreIndex', 'Fibre Length Multiplier',
                              'UserZeroRef', 'Unit Calibration (nm)', 'Attenuator 1', 'Attenuator 2',
                              'Fibre Length per Metre', 'Zero Offset (m)', 'Pulse Width 2 (ns)', 'GaugeLength',
                              'GPSTimeStamp']

            # From properties extract the important ones and store them in a dictionary
            properties_dict = {name: properties.get(name) for name in property_names}

            # Get group and channel names
            group_name = tdms_file.groups()[0].name
            first_channel_name = tdms_file.groups()[0].channels()[0].name

            # Add the 'n_samples_per_channel' key to the dictionary
            n_samples_per_channel = len(tdms_file[group_name][first_channel_name])
            properties_dict['n_samples_per_channel'] = n_samples_per_channel
            # Add the 'measurement_time (in seconds)' key to the dictionary
            properties_dict['measurement_time'] = n_samples_per_channel / properties_dict['SamplingFrequency[Hz]']
            # Add the 'distance' key to the dictionary
            properties_dict['distance'] = np.arange(properties_dict['MeasureLength[m]'] + 1) * \
                                          properties_dict['SpatialResolution[m]'] * \
                                          properties_dict['Fibre Length Multiplier'] + \
                                          properties_dict['Zero Offset (m)']

            self.properties = properties_dict

        return properties_dict

    def extract_data(self, file_name=None, first_channel=None, last_channel=None,
                     start_time=None, end_time=None, frequency=None):
        """
        Extract the file properties and the measurement data as a dictionary and an array respectively.

        @param file_name: name of the TDMS file to extract data from
        @param first_channel: first channel to be extracted
        @param last_channel: last channel to be extracted
        @param start_time: start time of the data to be extracted
        @param end_time: end time of the data to be extracted
        @param frequency: sampling frequency of the data

        @return: data
        """

        # Use the first TDMS file in the directory if file_name is not specified
        file_name = file_name or next(f for f in os.listdir(self.dir_path) if f.endswith('.tdms'))

        # Use instance's channels if first_channel or last_channel are not specified
        first_channel = first_channel if first_channel is not None else self.first_channel
        last_channel = last_channel if last_channel is not None else self.last_channel

        # Use the instance's properties if frequency is not specified
        frequency = frequency or self.properties['SamplingFrequency[Hz]']

        # Convert start_time and end_time to indices
        start_index = int(start_time * frequency) if start_time is not None else 0
        end_index = int(end_time * frequency) if end_time is not None else int(
            self.properties['measurement_time'] * frequency)

        # Construct the full file path
        file_path = os.path.join(self.dir_path, file_name)

        print('Extracting the data...')
        with td.read(file_path) as tdms_file:
            # Pre-filter the channels
            # Extract channels from TDMS groups within the specified channel range
            channels = [channel for group in tdms_file.groups() for channel in group.channels()
                        if first_channel <= int(channel.name) < last_channel]

            # Calculate the shape of the data array based on the number of channels and time range
            data_shape = (len(channels), end_index - start_index) if all(
                (end_index is not None, start_index is not None)) else None

            # Initialize data as an empty numpy array if data_shape is not None, otherwise set data to None
            data = np.empty(data_shape) if data_shape is not None else None

            # Populate data if it is not None
            if data is not None:
                for i, channel in enumerate(channels):
                    # Access numpy array of data for channel within the time range:
                    data[i] = channel[start_index:end_index]

            # Store the data in the class instance
            self.data = data

            return data

    def _calculate_cutoff_times(self, start_rate=0.2, end_rate=0.8):
        """
        Helper function to calculate the start and end times
        by removing the first and last 20% of the data

        @param start_rate: start rate of the data to be extracted
        @param end_rate: end rate of the data to be extracted

        @return: start_time, end_time
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

        @return: scan_channel, start_time, end_time
        """

        # Define the middle of the interest domain to scan there
        scan_channel = int(np.mean([self.first_channel, self.last_channel]))

        # Define the time window for the search
        start_time, end_time = self._calculate_cutoff_times()

        return scan_channel, start_time, end_time

    def signal_averaging(self, plot=False):
        """
        Look in a folder for all the TDMS files and extract the mean signal
        value from the search parameters

        @param plot: boolean to plot the mean signal

        @return: mean_signal
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
            plt.savefig('mean_signal.jpg', dpi=300)

        return mean_signal


    def get_files_above_threshold(self, mean_signal, threshold):
        """
        Get the list of file names based on a threshold value.

        @param mean_signal: list of mean signal values for each file
        @param threshold: threshold value to filter the files

        @return: selected_files
        """

        # Get the list of TDMS files in the directory
        tdms_files = [f for f in os.listdir(self.dir_path) if f.endswith('.tdms')]

        # Filter files based on the threshold
        selected_files = [file_name for file_name, mean_value in zip(
            tdms_files, mean_signal) if mean_value >= threshold]

        return selected_files

    def get_data(self, selected_files):
        """
        Extract measurement data for selected files.

        @param selected_files: list of file names to extract data from

        @return: all_selected_data
        """

        # Initialize an empty dictionary to store the selected data
        all_selected_data = {}

        # Loop through the selected files and extract the data
        for file_name in selected_files:
            data = self.extract_data(file_name)
            # Store the data in the dictionary
            all_selected_data[file_name] = data

        return all_selected_data