import numpy as np
from nptdms import TdmsFile as td
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class ReadTDMS:
    """"
    Class to extract properties and data from TDMS files
    """
    def __init__(self, file_path, first_channel, last_channel):
        """
        Initialize the class
        @param file_path: path to the TDMS file
        @param first_channel: first channel to be extracted
        @param last_channel: last channel to be extracted
        @param data: extracted data
        """
        self.file_path = file_path
        self.first_channel = first_channel
        self.last_channel = last_channel
        self.data = None

    def get_properties(self):
        """
        Get the properties of the TDMS file

        @return: properties of the TDMS file
        """
        # Open the TDMS file
        print('Reading the TDMS file...')
        with td.read(self.file_path) as tdms_file:
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

        return properties_dict

    def get_data(self):
        """
        Get the FO measurements from the TDMS file in a given range of channels

        @param first_channel: first channel to be extracted
        @param last_channel: last channel to be extracted
        @return: extracted data
        """
        # Initialize an empty list to store the selected data
        data = []

        # Open the TDMS file
        with td.read(self.file_path) as tdms_file:

            # Loop over all the groups in the TDMS file
            print('Extracting the measurements...')
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

    def plot_data(self, save_figure=False):
        """
        Plot the data from the TDMS file
        If 1 channel is selected, plot the data as a line plot
        If more than 1 channel is selected, plot the data as an image plot
        """
        if self.data is None:
            print("No data to plot. Please call get_data first.")
            return

        # Check if the data is a 1D array (single channel)
        if self.data.shape[0] == 1:
            # call the single channel function
            self.plot_single_channel()
            if save_figure == True:
                plt.savefig('figura1D.jpg', dpi=300)
                print('Figure for 1D data saved')
            elif save_figure == False:
                pass

        # Check if the data is a 2D array (multiple channels)
        elif self.data.shape[0] > 1:
            # call the multiple channels function
            self.plot_array_channels()
            if save_figure == True:
                plt.savefig('figura2D.jpg', dpi=300)
                print('Figure for 2D data saved')
            elif save_figure == False:
                pass


    def plot_single_channel(self):
        """
        Plot a single channel as a line plot
        """

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Squeeze out the second dimension
        data = np.squeeze(self.data)

        # Plot the data as a line plot
        ax.plot(data)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude')

        # Use a lambda function to display the time in seconds
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: x / 1000))


    def plot_array_channels(self):
        """
        Plot an array of channels as an image plot
        """
        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Transpose the data and take the absolute value
        data = np.abs(self.data.T)

        # Plot the data as an image plot
        ax.imshow(data, aspect='auto', cmap='jet', vmax=data.max() * 0.30)
        ax.set_xlabel('Channel count')
        ax.set_ylabel('Time [s]')

        #TODO: Find a more elegant solution for the labels in the x-axis

        # Generate evenly spaced values for x-ticks
        x_ticks = np.linspace(0, self.data.shape[0] - 1, num=10)
        x_labels = np.linspace(self.first_channel, self.last_channel,
                               num=10)

        # Set the x-ticks and their labels
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels.astype(int))  # Convert labels to integers

        # Use a lambda function to display the time in seconds
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: x / 1000))

