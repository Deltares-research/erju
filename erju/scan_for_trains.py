class ScanForTrains:
    """
    Class to scan selected data for signals
    that are larger than a given threshold
    """
    def __init__(self, tdms_data, tdm_properties, first_channel, last_channel):
        """
        Initialize the class instance
        @param data: selected data from the TDMS file
        @param properties: properties of the TDMS file
        @param first_channel: first channel to be extracted
        @param last_channel: last channel to be extracted
        """
        self.data = tdms_data
        self.properties = tdm_properties
        self.first_channel = first_channel
        self.last_channel = last_channel

    def search_params(self):
        """
        Define the middle of the domain, the time window
        and the spatial window for the search
        """
        # Define the middle of the interest domain to scan there
        scan_channel = self.first_channel + int((self.last_channel - self.first_channel) / 2)

        # Define the time window for the search
        measurement_time = self.properties['measurement_time']
        start_cutoff_rate = 0.2  # 20% of the total time
        end_cutoff_rate = 0.8  # 80% of the total time
        start_time, end_time = (
            round(measurement_time * start_cutoff_rate),
            round(measurement_time * end_cutoff_rate)
        )

        return scan_channel, start_time, end_time


    def signal_averaging(self):
        """
        For the selected data, average the signals values
        """
        # Get the selected data from the TDMS file
        data = self.data

        # Get the scan parameters
        scan_channel, start_time, end_time = self.search_params()


        # Get the average signal
        average_signal = np.mean(data[:, start_time:end_time], axis=1)

        return average_signal
