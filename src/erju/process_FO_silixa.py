import os
import numpy as np
from datetime import timedelta

from src.utils.TDMS_Read import TdmsReader
from src.erju.process_FO_base import BaseFOdata
from scipy.signal import iirfilter, zpk2sos, sosfilt


class SilixaFOdata(BaseFOdata):
    """
    Class for finding trains using the Silixa library
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


    def extract_properties(self):
        """
        Extract the file properties as a dictionary. In this function this is done only
        for the first file in the directory, because all files have ~ALMOST the same properties.
        If you need specific properties for each file, you can use the extract_properties_per_file function.

        Args:
            None

        Returns:
            properties (dict): The extracted properties

        """

        # Get a list of all TDMS files in the directory
        tdms_files = [f for f in os.listdir(self.dir_path) if f.endswith('.tdms')]
        # Get the first file name
        # We choose the first file because all files have the same properties, thus it doesn't matter
        #TODO: Now it matters, I need to use the initial time of each file...
        file_name = tdms_files[0]

        # Construct the full file path
        file_path = os.path.join(self.dir_path, file_name)

        #print('Extracting the properties...')
        tdms_instance = TdmsReader(file_path)
        # Get the properties of the TDMS file
        properties = tdms_instance.get_properties()

        data = tdms_instance.get_data()

        # Add the 'n_samples_per_channel' key to the dictionary
        n_samples_per_channel = len(data)
        properties['n_samples_per_channel'] = n_samples_per_channel
        # Add the 'measurement_time (in seconds)' key to the dictionary
        properties['measurement_time'] = n_samples_per_channel / properties['SamplingFrequency[Hz]']
        # Add the 'distance' key to the dictionary
        properties['distance'] = np.arange(properties['MeasureLength[m]'] + 1) * \
                                 properties['SpatialResolution[m]'] * \
                                 properties['Fibre Length Multiplier'] + \
                                 properties['Zero Offset (m)']
        # Add the 'TimeInterval' key to the dictionary by calculating it from the 'SamplingFrequency[Hz]'
        properties['TimeInterval'] = 1 / properties['SamplingFrequency[Hz]']
        # Add the 'MeasurementDuration' key to the dictionary by calculating it from the 'n_samples_per_channel'
        properties['MeasurementDuration'] = properties['TimeInterval'] * n_samples_per_channel
        # Add the 'FileStartTime' ket to the dictionary from the 'GPSTimeStamp'
        properties['FileStartTime'] = properties['GPSTimeStamp']
        # Add the 'FileEndTime' key to the dictionary by calculating with the measurement duration
        time_delta = timedelta(seconds=properties['MeasurementDuration'])
        properties['FileEndTime'] = properties['FileStartTime'] + time_delta

        self.properties = properties

        return properties


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

        # Create the TDMS instance
        tdms_instance = TdmsReader(file_path)
        # Get the properties of the TDMS file
        properties = tdms_instance.get_properties()

        data = tdms_instance.get_data()

        # Add the 'n_samples_per_channel' key to the dictionary
        n_samples_per_channel = len(data)
        properties['n_samples_per_channel'] = n_samples_per_channel
        # Add the 'measurement_time (in seconds)' key to the dictionary
        properties['measurement_time'] = n_samples_per_channel / properties['SamplingFrequency[Hz]']
        # Add the 'distance' key to the dictionary
        properties['distance'] = np.arange(properties['MeasureLength[m]'] + 1) * \
                                 properties['SpatialResolution[m]'] * \
                                 properties['Fibre Length Multiplier'] + \
                                 properties['Zero Offset (m)']
        # Add the 'TimeInterval' key to the dictionary by calculating it from the 'SamplingFrequency[Hz]'
        properties['TimeInterval'] = 1 / properties['SamplingFrequency[Hz]']
        # Add the 'MeasurementDuration' key to the dictionary by calculating it from the 'n_samples_per_channel'
        properties['MeasurementDuration'] = properties['TimeInterval'] * n_samples_per_channel
        # Add the 'FileStartTime' ket to the dictionary from the 'GPSTimeStamp'
        properties['FileStartTime'] = properties['GPSTimeStamp']
        # Add the 'FileEndTime' key to the dictionary by calculating with the measurement duration
        time_delta = timedelta(seconds=properties['MeasurementDuration'])
        properties['FileEndTime'] = properties['FileStartTime'] + time_delta

        self.properties = properties

        return self.properties


    def bandpass(self, data: np.array, freqmin: float, freqmax: float, fs: float, corners: int, zerophase=True):
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

        filtered_data = self.bandpass(data = data,
                                      freqmin=1,
                                      freqmax=100,
                                      fs=frequency,
                                      corners=5)

        # TO NOTE: The data is returned with shape (n_samples_per_ch, n_channels)
        #return data.T

        return self.data