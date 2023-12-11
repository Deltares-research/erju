import numpy as np
import glob

class ScanForTrains:
    """
    Class to scan selected data for signals
    that are larger than a given threshold
    """

    def __init__(self, tdms_data, tdms_properties, first_channel, last_channel):
        """
        Initialize the class instance
        @param data: selected data from the TDMS file
        @param properties: properties of the TDMS file
        @param first_channel: first channel to be extracted
        @param last_channel: last channel to be extracted
        """
        self.data = tdms_data
        self.properties = tdms_properties
        self.first_channel = first_channel
        self.last_channel = last_channel

    def _calculate_cutoff_times(self, measurement_time, start_rate=0.2, end_rate=0.8):
        """
        Helper function to calculate the start and end times
        by removing the first and last 20% of the data
        """
        return round(measurement_time * start_rate), round(measurement_time * end_rate)

    def search_params(self):
        """
        Define the middle of the domain, the time window
        and the spatial window for the search
        """
        # Define the middle of the interest domain to scan there
        scan_channel = int(np.mean([self.first_channel, self.last_channel]))

        # Define the time window for the search
        measurement_time = self.properties['measurement_time']
        start_time, end_time = self._calculate_cutoff_times(measurement_time)

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
        sampling_frequency = self.properties['sampling_frequency']
        start_index = int(start_time * sampling_frequency)
        end_index = int(end_time * sampling_frequency)

        # Get the average measurement for the scan channel at the time window
        mean_signal = np.mean(np.abs(self.data[relative_scan_channel, start_index:end_index]))

        return mean_signal


