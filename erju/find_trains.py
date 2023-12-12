import os
import numpy as np
from nptdms import TdmsFile as td


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

    def extract_data(self, file_name=None):
        """
        Extract the file properties and the measurement data
        as a dictionary and an array respectively
        """
        # Get a list of all TDMS files in the directory
        tdms_files = [f for f in os.listdir(self.dir_path) if f.endswith('.tdms')]

        # If no file name is specified, use the first file
        if file_name is None:
            file_name = tdms_files[0]

        # Construct the full file path
        file_path = os.path.join(self.dir_path, file_name)

        print('Extracting the data...')
        with td.read(file_path) as tdms_file:
            # Initialize an empty list to store the selected data
            data = []
            # Loop over all the groups in the TDMS file
            for group in tdms_file.groups():
                # Loop over all the channels in the group
                for channel in group.channels():
                    # Get the channel number
                    channel_number = int(channel.name)
                    # Check if the channel is within the range of selected channels
                    if self.first_channel <= channel_number < self.last_channel:
                        # Access numpy array of data for channel:
                        measurements = channel[:]
                        # Append the measurements to the list of data
                        data.append(measurements)

        # Convert the list of selected data to a numpy array and transpose it
        data = np.array(data)

        # Store the data in the class instance
        self.data = data

        return data

    def _calculate_cutoff_times(self, start_rate=0.2, end_rate=0.8):
        """
        Helper function to calculate the start and end times
        by removing the first and last 20% of the data
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
        """

        # Define the middle of the interest domain to scan there
        scan_channel = int(np.mean([self.first_channel, self.last_channel]))

        # Define the time window for the search
        start_time, end_time = self._calculate_cutoff_times()

        return scan_channel, start_time, end_time

    def signal_averaging(self):
        """
        Look in a folder for all the TDMS files and extract the mean signal
        value from the search parameters
        """

        # Get the search parameters
        scan_channel, start_time, end_time = self.search_params()

        # Adjust the scan_channel to be relative to the first_channel
        relative_scan_channel = scan_channel - self.first_channel

        # Convert the start and end times to indices
        sampling_frequency = self.properties['SamplingFrequency[Hz]']
        start_index = int(start_time * sampling_frequency)
        end_index = int(end_time * sampling_frequency)

        mean_signal = []
        # Loop through all the TDMS files in the directory
        for file_name in os.listdir(self.dir_path):
            # Get the mean signal for each file in the relative scan channel
            # and using the start_index and end_index, and get them in a list
            tdms_data = self.extract_data(file_name)
            mean_signal.append(np.mean(np.abs(
                self.data[relative_scan_channel, start_index:end_index])))
        # Convert the list to a numpy array
        mean_signal = np.array(mean_signal)

        return mean_signal


