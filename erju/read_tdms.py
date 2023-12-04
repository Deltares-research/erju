import numpy as np
from nptdms import TdmsFile as td
import matplotlib.pyplot as plt
import matplotlib



class ReadTDMS:
    """"
    Class to extract properties and data from TDMS files
    """
    def __init__(self, file_path):
        """
        Initialize the class
        @param file_path: path to the TDMS file
        """
        self.file_path = file_path
        self.data = None


    def get_properties(self):
        """
        Get the properties of the TDMS file

        @return: properties of the TDMS file
        """
        # Open the TDMS file
        with td.read(self.file_path) as tdms_file:
            # Get the properties of the TDMS file
            properties = tdms_file.properties

            # From properties extract the important ones and store them in a dictionary
            properties_dict = {
                'name': properties.get('name'),
                'sampling_frequency': properties.get('SamplingFrequency[Hz]'),
                'spatial_resolution': properties.get('SpatialResolution[m]'),
                'start_position': properties.get('StartPosition[m]'),
                'n_channels': properties.get('MeasureLength[m]'),
                'start_distance': properties.get('Start Distance (m)'),
                'stop_distance': properties.get('Stop Distance (m)'),
                'peak_voltage': properties.get('PeakVoltage[V]'),
                'pulse_2_delay': properties.get('Pulse 2 Delay (ns)'),
                'pulse_width': properties.get('PulseWidth[ns]'),
                'offset_length': properties.get('OffsetLength'),
                'reference_level_1': properties.get('Reference Level 1'),
                'reference_level_2': properties.get('Reference Level 2'),
                'reference_level_3': properties.get('Reference Level 3'),
                'fibre_index': properties.get('FibreIndex'),
                'fibre_length_multiplier': properties.get('Fibre Length Multiplier'),
                'user_zero_ref': properties.get('UserZeroRef'),
                'unit_calibration': properties.get('Unit Calibration (nm)'),
                'attenuator1': properties.get('Attenuator 1'),
                'attenuator2': properties.get('Attenuator 2'),
                'fibre_length_per_metre': properties.get('Fibre Length per Metre'),
                'zero_offset': properties.get('Zero Offset (m)'),
                'pulse_width_2': properties.get('Pulse Width 2 (ns)'),
                'gauge_length': properties.get('GaugeLength'),
                'gps_time': properties.get('GPSTimeStamp'),

                'group_name': tdms_file.groups()[0].name,
                'first_channel_name': tdms_file.groups()[0].channels()[0].name
            }

        # Add the 'n_samples_per_channel' key to the dictionary
        properties_dict['n_samples_per_channel'] = len(
            tdms_file[properties_dict['group_name']][properties_dict['first_channel_name']])
        # Add the 'measurement_time (in seconds)' key to the dictionary
        properties_dict['measurement_time'] = properties_dict['n_samples_per_channel'] / \
                                              properties_dict['sampling_frequency']
        # Add the 'distance' key to the dictionary
        properties_dict['distance'] = np.arange(properties_dict['n_channels']+1) * \
                                      properties_dict['spatial_resolution'] * \
                                      properties_dict['fibre_length_multiplier'] + \
                                      properties_dict['zero_offset']

        return properties_dict


    def get_data(self, first_channel, last_channel):
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
            for group in tdms_file.groups():
                # Loop over all the channels in the group
                for channel in group.channels():
                    # Get the channel number
                    channel_number = int(channel.name)
                    # Check if the channel is within the range of selected channels
                    if first_channel <= channel_number < last_channel:
                        # Access numpy array of data for channel:
                        measurements = channel[:]
                        # Append the measurements to the list of data
                        data.append(measurements)

        # Convert the list of selected data to a numpy array and transpose it
        data = np.array(data)

        # Store the data in the class instance
        self.data = data

        return data

    def plot_data(self):
        """
        Plot the data stored in the class instance
        """
        if self.data is None:
            print("No data to plot. Please call get_data first.")
            return

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Check if the data is a 1D array (single channel)
        if self.data.shape[0] == 1:
            # Squeeze out the second dimension
            data = np.squeeze(self.data)
            # Plot the data as a line plot
            ax.plot(data)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Amplitude')

            # Use a lambda function to display the time in seconds
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: x / 1000))

        # Check if the data is a 2D array (multiple channels)
        elif self.data.shape[0] > 1:
            # Transpose the data and take the absolute value
            data = np.abs(self.data.T)

            # Plot the data as an image plot
            ax.imshow(data, aspect='auto', cmap='jet', vmax=data.max() * 0.30)
            ax.set_xlabel('Channel / Distance [m]')
            ax.set_ylabel('Time [s]')

            # Use a lambda function to display the time in seconds
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: x / 1000))

        plt.show()

    def plot_imshow(self, data, start_channel, n_traces, start_time, end_time, save_figure=True):
        """
        Function for plotting DAS traces

        INPUT:

            data    :       Extracted iDAS traces
            rec_name:       name of selected tdms record
            first_channel:  First channel assigned to extract traces
            start_time  : user selectable initial time
            end_time    : user seletable end time of selected file
            save_figure : string for saving plotted figure. If True figure is saved.

        """


        matplotlib.rc('xtick', labelsize=14)
        matplotlib.rc('ytick', labelsize=14)
        fig, ax = plt.subplots(figsize=(10, 6))
        Z = np.abs(data)
        plt.imshow((Z), interpolation='kaiser',
                   extent=[start_channel, start_channel + n_traces, end_time, start_time],
                   aspect='auto', cmap='jet', vmax=Z.max() * 0.30)
        plt.xlabel('Distance[m]', fontsize=14)
        plt.ylabel('Time [s]', fontsize=14)
        plt.show(block=False)

        if save_figure == True:

            plt.savefig('mifigurita.jpg', dpi=300)

        elif save_figure == False:

            pass





file_path = r'C:\Projects\erju\data\iDAS_continous_measurements_30s_UTC_20201121_101949.913.tdms'
file_1 = ReadTDMS(file_path)
properties = file_1.get_properties()
data = file_1.get_data(2, 3)
file_1.plot_data()

#file_1.plot_imshow(data, 4200, 100, 0, 30, save_figure=True)

print(data.shape)

