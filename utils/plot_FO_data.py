import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import logging

# Class to plot the data that takes in the path and file names
class PlotData:
    """
    Class to plot the data from the TDMS file with different options
    """

    def __init__(self, file_name: str, all_data: dict):
        """
        Initialize the class

        Args:
            file_name (str): The name of the file to be plotted
            all_data (dict): The dictionary containing all the data
        
        Returns:
            None
        """
        self.file_name = file_name
        self.all_data = all_data

        # Extract the numpy array for the specified file_name from the all_data dictionary
        if file_name in all_data:
            self.selected_data = all_data[file_name]
        else:
            # Raise an error if the file_name is not in the dictionary
            raise ValueError(f'{file_name} is not in the dictionary')

    def plot_single_channel(self, channel_index: int, save_to_path: str = None, save_figure: bool = False):
        """
        Plot a single channel as a line plot

        Args:
            save_to_path (str): The path to save the figure
            channel_index (int): The index of the channel to be plotted
            save_figure (bool): A flag to save the figure

        Returns:
            None
        """
        # Check if the specified channel_index is valid
        if channel_index < 0 or channel_index >= self.selected_data.shape[0]:
            raise ValueError(f'Invalid channel index: {channel_index}. It should be between 0 and {self.selected_data.shape[0] - 1}.')

        # Extract the data for the specified channel
        channel_data = np.squeeze(self.selected_data[channel_index])

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the data as a line plot
        ax.plot(channel_data)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'Amplitude (Channel {channel_index})')

        # Use a lambda function to display the time in seconds
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: x / 1000))

        # If save_figure is True, save the figure with a specific file name
        if save_figure:
            file_name_suffix = f'Figure_1D_Channel_{channel_index}'
            full_file_name = f'{self.file_name}_{file_name_suffix}.jpg'
            save_path = os.path.join(save_to_path, full_file_name) if save_to_path else os.path.join('..', 'test', 'test_output', full_file_name)
            plt.savefig(save_path, dpi=300)
            logging.info(f'Single channel figure for file {self.file_name} and channel {channel_index} saved. File name: {full_file_name}')
        plt.close()

    def plot_array_channels(self, save_to_path: str = None, save_figure: bool = False):
        """
        Plot an array of channels as an image plot

        Args:
            save_to_path (str): The path to save the figure
            save_figure (bool): A flag to save the figure

        Returns:
            None
        """
        # Check if the selected_data is None
        if self.selected_data is None:
            logging.warning("No data to plot. Please call get_data first.")
            return

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Transpose the data and take the absolute value
        data = np.abs(self.selected_data.T)

        # Plot the data as an image plot
        im = ax.imshow(data, aspect='auto', cmap='jet', vmax=data.max() * 0.30)
        ax.set_xlabel('Channel count')
        ax.set_ylabel('Time [s]')

        # Set the number of ticks based on the dimensions of the data
        num_channels = self.selected_data.shape[0]
        num_time_points = self.selected_data.shape[1]

        # Set the x-ticks and their labels
        x_ticks = np.linspace(0, num_channels - 1, num=min(10, num_channels))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{int(x)}' for x in x_ticks])

        # Set the y-ticks and their labels
        y_ticks = np.linspace(0, num_time_points - 1, num=min(6, num_time_points))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{int(y / 1000)}' for y in y_ticks])

        # Show colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity')

        # If save_figure is True, save the figure with a specific file name
        if save_figure:
            file_name_suffix = 'Figure_2D'
            full_file_name = f'{self.file_name}_{file_name_suffix}.jpg'
            save_path = os.path.join(save_to_path, full_file_name) if save_to_path else os.path.join('..', 'test', 'test_output', full_file_name)
            plt.savefig(save_path, dpi=300)
            logging.info(f'2D figure for file {self.file_name} saved. File name: {full_file_name}')
        plt.close()


    def plot_2d_buffer(self,  save_to_path: str = None, save_figure: bool = False, data = None):
        """
        Plot an array of channels as an image plot, including a time window before and after the train passes.
        The data can also be resampled if needed.

        Args:
            save_to_path (str): The path to save the figure
            save_figure (bool): Save the figure as a jpg file
            window_before (int): The time window (in seconds) before the train passes
            window_after (int): The time window (in seconds) after the train passes
            resample (bool): Resample the data
            data (int): The new sampling frequency

        Returns:
            None
        """

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(10, 15))

        # Transpose the data and take the absolute value
        data = np.abs(data.T)

        # Plot the data as an image plot
        im = ax.imshow(data, aspect='auto', cmap='jet', vmax=data.max() * 0.30)
        ax.set_xlabel('Channel count')
        ax.set_ylabel('Time [s]')

        # Set the number of ticks based on the dimensions of the data
        num_time_points = data.shape[0]
        num_channels = data.shape[1]

        # Set the x-ticks and their labels
        x_ticks = np.linspace(0, num_channels - 1, num=min(10, num_channels))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{int(x)}' for x in x_ticks])

        # Set the y-ticks and their labels
        y_ticks = np.linspace(0, num_time_points - 1, num=min(10, num_time_points))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{int(y / 1000)}' for y in y_ticks])

        # Show colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity')

        # If save_figure is True, save the figure with a specific file name
        if save_figure:
            file_name_suffix = 'Figure_2D'
            full_file_name = f'{file_name_suffix}_{self.file_name}.jpg'
            save_path = os.path.join(save_to_path, full_file_name) if save_to_path else os.path.join(save_to_path, full_file_name)
            plt.savefig(save_path, dpi=300)
        plt.close()
