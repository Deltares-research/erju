import os
import numpy as np
import h5py
import pandas as pd
from datetime import datetime

from utils.TDMS_Read import TdmsReader
from erju.process_FO_base import BaseFOdata
from scipy.signal import iirfilter, sosfilt, zpk2sos, sosfilt, windows

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


    def from_opticalphase_to_strain(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Take the raw OptaSense data and convert it to units of strain.

        Args:
            signal_data (np.ndarray): The raw OptaSense data

        Returns:
            data (np.ndarray): The strain data
        """
        # Remove the mean from the data. Since it is a 2D of (150000>time, 5000>location) [rows, columns]
        # We remove the mean over time for each location with axis=0 (operation over rows)
        raw_data = raw_data - np.mean(raw_data, axis=0)

        # Convert into units of radians
        raw_data = raw_data * (2*np.pi / 2**16)

        # Get from the properties the values I need to convert to strain
        n = self.properties['Fibre Refractive Index']
        L = self.properties['GaugeLength']
        # Convert into units of strain
        data = raw_data * ((1550.12 * 1e-9)/(0.78 * 4 * np.pi * n * L))

        return data

    def bandpass(self, data, freqmin, freqmax, fs, corners, zerophase=True):
        """
        Apply a bandpass filter to the data.

        Args:
            data (np.array): The data to be filtered.
            freqmin (float): The lower frequency bound of the filter.
            freqmax (float): The upper frequency bound of the filter.
            fs (float): The sampling frequency.
            corners (int): The number of corners in the filter.
            zerophase (bool): Whether to apply the filter in both directions.

        Returns:
            np.array: The filtered data
        """
        fe = 0.5 * fs
        low = freqmin / fe
        high = freqmax / fe
        z, p, k = iirfilter(corners, [low, high], btype='band', ftype='butter', output='zpk')
        sos = zpk2sos(z, p, k)

        if zerophase:
            firstpass = sosfilt(sos, data)
            return sosfilt(sos, firstpass[::-1])[::-1]
        else:
            return sosfilt(sos, data)


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
            self.properties = {
                'AcquisitionId': file['Acquisition'].attrs['AcquisitionId'],
                'GaugeLength': file['Acquisition'].attrs['GaugeLength'],
                'GaugeLengthUnit': file['Acquisition'].attrs['GaugeLengthUnit'],
                'MaximumFrequency': file['Acquisition'].attrs['MaximumFrequency'],
                'MinimumFrequency': file['Acquisition'].attrs['MinimumFrequency'],
                'NumberOfLoci': file['Acquisition'].attrs['NumberOfLoci'],
                'RawSamplingFrequency[Hz]': file['Acquisition'].attrs['PulseRate'],
                'PulseWidth': file['Acquisition'].attrs['PulseWidth'],
                'PulseWidthUnit': file['Acquisition'].attrs['PulseWidthUnit'],
                'SpatialSamplingInterval': file['Acquisition'].attrs['SpatialSamplingInterval'],
                'SpatialSamplingIntervalUnit': file['Acquisition'].attrs['SpatialSamplingIntervalUnit'],
                'StartLocusIndex': file['Acquisition'].attrs['StartLocusIndex'],
                'TriggeredMeasurement': file['Acquisition'].attrs['TriggeredMeasurement'],
                'Fibre Refractive Index': file['Acquisition']['Custom'].attrs['Fibre Refractive Index'],
                'GPS Enabled': file['Acquisition']['Custom'].attrs['GPS Enabled'],
                'Num Elements Per Channel': file['Acquisition']['Custom'].attrs['Num Elements Per Channel'],
                'Num Outputs Channels': file['Acquisition']['Custom'].attrs['Num Output Channels'],
                'SamplingFrequency[Hz]': file['Acquisition']['Raw[0]'].attrs['OutputDataRate'],
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


        return self.properties


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

        # Create the file path
        file_path = os.path.join(self.dir_path, file_name)

        # Open the .h5 file
        with h5py.File(file_path, 'r') as file:

            # Create an instance of the raw data for easier access
            all_raw_data = file['Acquisition']['Raw[0]']['RawData']
            # Get the selected channels
            raw_signal_data = all_raw_data[:, first_channel:last_channel+1]

            # Apply the tukey window to the raw data in order to reduce the edge effects prior to filtering
            signal_window = windows.tukey(M=raw_signal_data.shape[0], alpha=0.1)
            # Create a new array to store the filtered data
            filtered_data = np.zeros(np.shape(raw_signal_data))
            # Filter the data
            for i in range(raw_signal_data.shape[1]):
                filtered_data[:, i] = self.bandpass(data=raw_signal_data[:, i] * signal_window,
                                                    freqmin=1,
                                                    freqmax=50,
                                                    fs=self.properties['SamplingFrequency[Hz]'],
                                                    corners=5)


        # Convert the raw data to strain
        data = self.from_opticalphase_to_strain(filtered_data)

        # Store the data in the class instance and transpose it to make it fit the other code
        self.data = data.T

        # TO NOTE: The data is returned with shape (n_samples_per_ch, n_channels)
        return self.data