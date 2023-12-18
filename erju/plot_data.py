import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# class to plot the data that takes in the path and file names
class PlotData:
    """
    Class to plot the data from the TDMS file
    """

    def __init__(self, dir_path, file_name, all_data):
        """
        Initialize the class
        @param dir_path: path to the TDMS file
        @param selected_files: list of selected files
        @param data: extracted data
        """
        self.dir_path = dir_path
        self.file_name = file_name
        self.all_data = all_data

        # Extract the numpy array for the specified file_name
        if file_name in all_data:
            self.selected_data = all_data[file_name]
        else:
            # Handle the case where the specified file_name is not found
            self.selected_data = np.array([])  # or raise an exception, set to None, etc.

    def plot_data(self, save_figure=False):
        """
        Plot the data from the TDMS file
        If 1 channel is selected, plot the data as a line plot
        If more than 1 channel is selected, plot the data as an image plot

        @param save_figure: boolean to save the figure
        """
        if self.selected_data is None:
            print("No data to plot. Please call get_data first.")
            return

        # Check if the data is a 1D array (single channel)
        if self.selected_data.shape[0] == 1:
            # call the single channel function
            self.plot_single_channel()
            if save_figure == True:
                file_name_suffix = 'Figure_1D'
                full_file_name = f'{self.file_name}_{file_name_suffix}.jpg'
                save_path = os.path.join('..', 'test', 'test_output', full_file_name)
                plt.savefig(save_path, dpi=300)
                print('Figure for 1D data saved')
            elif save_figure == False:
                pass

        # Check if the data is a 2D array (multiple channels)
        elif self.selected_data.shape[0] > 1:
            # call the multiple channels function
            self.plot_array_channels()
            if save_figure == True:
                file_name_suffix = 'Figure_2D'
                full_file_name = f'{self.file_name}_{file_name_suffix}.jpg'
                save_path = os.path.join('..', 'test', 'test_output', full_file_name)
                plt.savefig(save_path, dpi=300)
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
        data = np.squeeze(self.selected_data)

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
        data = np.abs(self.selected_data.T)

        # Plot the data as an image plot
        ax.imshow(data, aspect='auto', cmap='jet', vmax=data.max() * 0.30)
        ax.set_xlabel('Channel count')
        ax.set_ylabel('Time [s]')

        #TODO: Find a more elegant solution for the labels in the x-axis

        # Generate evenly spaced values for x-ticks
        x_ticks = np.linspace(0, self.selected_data.shape[0] - 1, num=10)

        # Set the x-ticks and their labels
        ax.set_xticks(x_ticks)

        # Use a lambda function to display the time in seconds
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: x / 1000))


