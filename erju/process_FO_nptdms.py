import os
import numpy as np
from datetime import datetime, timedelta

from nptdms import TdmsFile as td
from erju.process_FO_base import BaseFOdata

class NptdmsFOdata(BaseFOdata):
    """
    Class for finding trains using the nptdms library
    """

    def __init__(self, dir_path: str, first_channel: int, last_channel: int):
        """
        Initialize the class instance

        Args:
            dir_path (str): The path to the folder containing the TDMS file
            first_channel (int): The first channel to be extracted
            last_channel (int): The last channel to be extracted
        """
        super().__init__(dir_path, first_channel, last_channel)
        self.tdms_file = None  # Cache TDMS file to avoid multiple reads

    def _read_tdms_file(self, file_name: str):
        """
        Read TDMS file and cache it for future use.

        Args:
            file_name (str): The name of the TDMS file to read
        """
        if self.tdms_file is None:
            file_path = os.path.join(self.dir_path, file_name)
            self.tdms_file = td.read(file_path)

    def extract_properties(self, file_name: str = None):
        """
        Extract the file properties and the measurement data

        Args:
            file_name (str): name of the TDMS file to extract data from

        Returns:
            properties_dict (dict): dictionary containing the properties of the TDMS file
        """
        # Get a list of all TDMS files in the directory
        tdms_files = [f for f in os.listdir(self.dir_path) if f.endswith('.tdms')]

        # If no file name is specified, use the first file
        file_name = file_name or tdms_files[0]

        self._read_tdms_file(file_name)
        tdms_file = self.tdms_file

        print('Extracting the properties...')
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

        # Extract properties
        properties_dict = {name: properties.get(name) for name in property_names}

        # Get group and channel names
        group_name = tdms_file.groups()[0].name
        first_channel_name = tdms_file.groups()[0].channels()[0].name

        # Add additional properties
        n_samples_per_channel = len(tdms_file[group_name][first_channel_name])
        properties_dict['n_samples_per_channel'] = n_samples_per_channel
        properties_dict['measurement_time'] = n_samples_per_channel / properties_dict['SamplingFrequency[Hz]']
        properties_dict['distance'] = np.arange(properties_dict['MeasureLength[m]'] + 1) * \
                                      properties_dict['SpatialResolution[m]'] * \
                                      properties_dict['Fibre Length Multiplier'] + \
                                      properties_dict['Zero Offset (m)']

        # Convert GPSTimeStamp to datetime
        properties_dict['GPSTimeStamp'] = properties_dict['GPSTimeStamp'].astype(datetime)

        # Calculate and add more properties
        properties_dict['TimeInterval'] = 1 / properties_dict['SamplingFrequency[Hz]']
        properties_dict['MeasurementDuration'] = properties_dict['TimeInterval'] * n_samples_per_channel
        properties_dict['FileStartTime'] = properties_dict['GPSTimeStamp']
        properties_dict['FileEndTime'] = properties_dict['FileStartTime'] + timedelta(seconds=properties_dict['MeasurementDuration'])

        self.properties = properties_dict

        return properties_dict

    def extract_data(self, file_name: str = None, first_channel: int = None, last_channel: int = None,
                     start_time: int = None, end_time: int = None, frequency: int = None):
        """
        Extract the measurement data as a numpy array.

        Args:
            file_name (str): name of the TDMS file to extract data from
            first_channel (int): The first channel to be extracted
            last_channel (int): The last channel to be extracted
            start_time (int): The start time in seconds
            end_time (int): The end time in seconds
            frequency (int): The frequency of the data

        Returns:
            data (np.ndarray): The measurement data as a numpy array
        """
        # Use the first TDMS file in the directory if file_name is not specified
        file_name = file_name or next(f for f in os.listdir(self.dir_path) if f.endswith('.tdms'))

        self._read_tdms_file(file_name)
        tdms_file = self.tdms_file

        # Use instance's channels if first_channel or last_channel are not specified
        first_channel = first_channel if first_channel is not None else self.first_channel
        last_channel = last_channel if last_channel is not None else self.last_channel

        # Use the instance's properties if frequency is not specified
        frequency = frequency or self.properties['SamplingFrequency[Hz]']

        # Convert start_time and end_time to indices
        start_index = int(start_time * frequency) if start_time is not None else 0
        end_index = int(end_time * frequency) if end_time is not None else int(self.properties['measurement_time'] * frequency)

        # Extract the data
        # Pre-filter the channels
        channels = [channel for group in tdms_file.groups() for channel in group.channels()
                    if first_channel <= int(channel.name) < last_channel]

        # Calculate the shape of the data array
        data_shape = (len(channels), end_index - start_index) if (end_index is not None and start_index is not None) else None

        # Initialize data as an empty numpy array
        data = np.empty(data_shape) if data_shape is not None else None

        # Populate data if it is not None
        if data is not None:
            for i, channel in enumerate(channels):
                # Access numpy array of data for channel within the time range:
                data[i] = channel[start_index:end_index]

        # Store the data in the class instance
        self.data = data

        return data
