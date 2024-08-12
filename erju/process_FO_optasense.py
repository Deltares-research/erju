import os
import numpy as np
import h5py
import pandas as pd
from datetime import datetime

from utils.TDMS_Read import TdmsReader
from erju.process_FO_base import BaseFOdata

class OptasenseFOdata(BaseFOdata):
    """
    Class for finding trains using the Optasense type format based on .h5 files
    """

    def __init__(self, dir_path: str, first_channel: int, last_channel: int):
        """
        Initialize the class instance

        Args:
            dir_path (str): The path to the folder containing the TDMS file
            first_channel (int): The first channel to be extracted
            last_channel (int): The last channel to be extracted. ** If you want just 1 channel, set first_channel = last_channel
        """
        # Call the __init__ method of the BaseFindTrains class
        super().__init__(dir_path, first_channel, last_channel)

    def convert_microseconds_to_datetime(self, microseconds: int) -> datetime:
        """
        Convert a timestamp in microseconds since epoch to a UTC datetime.

        Args:
            microseconds (int): Timestamp in microseconds.

        Returns:
            datetime (datetime): Corresponding UTC datetime.
        """
        # Convert microseconds to seconds
        seconds = microseconds * 1e-6

        # Return the corresponding datetime
        return datetime.utcfromtimestamp(seconds)

    def extract_properties_per_file(self, file_name: str):
        """
        Extract the file properties as a dictionary for a given file. Note that this function is different
        from the extract_properties function, because it does not add the additional manually calculated properties.
        Those are only added in the extract_properties function.

        Args:
            file_name (str): The name of the file to extract the properties from

        Returns:
            properties (dict): The extracted properties
        """
        # From the file name build the full file path
        file_path = os.path.join(self.dir_path, file_name)

        # Open the .h5 file
        with h5py.File(file_path, 'r') as file:

            # Calculate some parameters that are not directly available
            # Get the "RawDataTime" dataset
            rawDataTime = file['Acquisition']['Raw[0]']['RawDataTime']
            # Get the first and last entry in "RawDataTime"
            file_start_time = self.convert_microseconds_to_datetime(rawDataTime[0])
            file_end_time = self.convert_microseconds_to_datetime(rawDataTime[-1])
            # Calculate the duration of the measurement
            measurement_duration = (file_end_time - file_start_time).total_seconds()
            # Calculate the time interval between samples
            time_interval = (rawDataTime[1] - rawDataTime[0]) * 1e-6 # Convert to seconds

            # Get the properties from different sections in the file
            properties = {
                'AcquisitionId': file['Acquisition'].attrs['AcquisitionId'],
                'GaugeLength': file['Acquisition'].attrs['GaugeLength'],
                'GaugeLengthUnit': file['Acquisition'].attrs['GaugeLengthUnit'],
                'MaximumFrequency': file['Acquisition'].attrs['MaximumFrequency'],
                'MinimumFrequency': file['Acquisition'].attrs['MinimumFrequency'],
                'NumberOfLoci': file['Acquisition'].attrs['NumberOfLoci'],
                'PulseRate': file['Acquisition'].attrs['PulseRate'],
                'PulseWidth': file['Acquisition'].attrs['PulseWidth'],
                'PulseWidthUnit': file['Acquisition'].attrs['PulseWidthUnit'],
                'SpatialSamplingInterval': file['Acquisition'].attrs['SpatialSamplingInterval'],
                'SpatialSamplingIntervalUnit': file['Acquisition'].attrs['SpatialSamplingIntervalUnit'],
                'StartLocusIndex': file['Acquisition'].attrs['StartLocusIndex'],
                'TriggeredMeasurement': file['Acquisition'].attrs['TriggeredMeasurement'],
                'GPS Enabled': file['Acquisition']['Custom'].attrs['GPS Enabled'],
                'Num Elements Per Channel': file['Acquisition']['Custom'].attrs['Num Elements Per Channel'],
                'Num Outputs Channels': file['Acquisition']['Custom'].attrs['Num Output Channels'],
                'OutputDataRate': file['Acquisition']['Raw[0]'].attrs['OutputDataRate'],
                'RawDataUnit': file['Acquisition']['Raw[0]'].attrs['RawDataUnit'],
                'RawDescription': file['Acquisition']['Raw[0]'].attrs['RawDescription'],
                'Count': file['Acquisition']['Raw[0]']['RawData'].attrs['Count'],
                'Dimensions': file['Acquisition']['Raw[0]']['RawData'].attrs['Dimensions'],
                'PartEndTime': file['Acquisition']['Raw[0]']['RawData'].attrs['PartEndTime'],
                'PartStartTime': file['Acquisition']['Raw[0]']['RawData'].attrs['PartStartTime'],
                'StartIndex': file['Acquisition']['Raw[0]']['RawData'].attrs['StartIndex'],
                'FileStartTime': file_start_time,
                'FileEndTime': file_end_time,
                'MeasurementDuration': measurement_duration,
                'TimeInterval': time_interval,
                'NumberOfMeasurements': file['Acquisition']['Raw[0]']['RawData'].shape[0]

            }

        return properties





    def extract_data(self, file_name: str = None, first_channel: int = None, last_channel: int = None,
                     start_time: int = None, end_time: int = None, frequency: int = None):
        """
        Extract the file properties and the measurement data as a dictionary and an array respectively.

        Args:
            file_name (str): The name of the file to extract the data from
            first_channel (int): The first channel to be extracted
            last_channel (int): The last channel to be extracted ** If you want just 1 channel, set first_channel = last_channel
            start_time (int): The start time of the data to be extracted
            end_time (int): The end time of the data to be extracted
            frequency (int): The sampling frequency of the data

        Returns:
            data (np.array): The extracted data
        """

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
        #print('Extracting the data from file:', file_path, 'in extract_data')

        # Create the TDMS instance
        tdms_instance = TdmsReader(file_path)

        # Get the data
        data = tdms_instance.get_data(first_channel, last_channel, start_index, end_index)

        # Store the data in the class instance
        self.data = data.T

        # TO NOTE: The data is returned with shape (n_samples_per_ch, n_channels)
        return data.T