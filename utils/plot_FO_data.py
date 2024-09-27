import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import logging
import pandas as pd
from datetime import datetime

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
            duration (float): The duration of the data in seconds
        
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


    def plot_single_channel(self, channel_index: int, start_time: datetime, end_time: datetime,
                            save_to_path: str = None, save_figure: bool = False):
        """
        Plot a single channel as a line plot with time duration

        Args:
            channel_index (int): The index of the channel to be plotted
            start_time (datetime): The start time of the measurement
            end_time (datetime): The end time of the measurement
            save_to_path (str): The path to save the figure
            save_figure (bool): A flag to save the figure

        Returns:
            None
        """
        # Check if the specified channel_index is valid
        if channel_index < 0 or channel_index >= self.selected_data.shape[0]:
            raise ValueError(
                f'Invalid channel index: {channel_index}. It should be between 0 and {self.selected_data.shape[0] - 1}.')

        # Extract the data for the specified channel
        channel_data = np.squeeze(self.selected_data[channel_index])

        # Convert datetime objects to timestamps (seconds since the epoch)
        start_timestamp = start_time.timestamp()
        end_timestamp = end_time.timestamp()

        # Calculate the total duration and create a time vector
        num_points = len(channel_data)
        time_vector = np.linspace(start_timestamp, end_timestamp, num_points)

        # Convert the time vector from timestamps to a more readable format (seconds since start_time)
        time_vector -= start_timestamp

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the data as a line plot
        ax.plot(time_vector, channel_data)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'Amplitude (Channel {channel_index})')

        # Format x-axis to show time in seconds
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.1f}'))

        # If save_figure is True, save the figure with a specific file name
        if save_figure:
            file_name_suffix = f'Figure_1D_Channel_{channel_index}'
            full_file_name = f'{self.file_name}_{file_name_suffix}.jpg'
            save_path = os.path.join(save_to_path, full_file_name) if save_to_path else os.path.join('..', 'test',
                                                                                                     'test_output',
                                                                                                     full_file_name)
            plt.savefig(save_path, dpi=300)
            logging.info(
                f'Single channel figure for file {self.file_name} and channel {channel_index} saved. File name: {full_file_name}')

        plt.close()


    def plot_array_channels(self, start_time: datetime, end_time: datetime, save_to_path: str = None,
                            guide_line:int = None, save_figure: bool = False):
        """
        Plot an array of channels as an image plot with time scale

        Args:
            start_time (datetime): The start time of the measurement
            end_time (datetime): The end time of the measurement
            save_to_path (str): The path to save the figure
            save_figure (bool): A flag to save the figure

        Returns:
            None
        """
        # Check if the selected_data is None
        if self.selected_data is None:
            logging.warning("No data to plot. Please call get_data first.")
            return

        # Convert datetime objects to timestamps (seconds since the epoch)
        start_timestamp = start_time.timestamp()
        end_timestamp = end_time.timestamp()

        # Calculate the total duration and create a time vector
        num_time_points = self.selected_data.shape[1]
        time_vector = np.linspace(start_timestamp, end_timestamp, num_time_points)

        # Convert the time vector from timestamps to a more readable format (seconds since start_time)
        time_vector -= start_timestamp

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Transpose the data and take the absolute value
        data = np.abs(self.selected_data.T)

        # Plot the data as an image plot
        # im = ax.imshow(data, aspect='auto', cmap='jet', vmax=data.max() * 0.30) # This is the original line
        im = ax.imshow(np.abs(np.log10(data)), aspect='auto', cmap='jet') # better visualization for both silixa and optasense
        ax.set_xlabel('Channel count')
        ax.set_ylabel('Time [s]')

        # Set the number of ticks based on the dimensions of the data
        num_channels = self.selected_data.shape[0]

        # Set the x-ticks and their labels
        x_ticks = np.linspace(0, num_channels - 1, num=min(10, num_channels))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{int(x)}' for x in x_ticks])

        # Set the y-ticks and their labels
        y_ticks = np.linspace(0, num_time_points - 1, num=min(6, num_time_points))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{int(time_vector[int(y)]):,}' for y in y_ticks])

        # If the user provides a guide_line, add a vertical dotted line at that X
        if guide_line:
            ax.axvline(x=guide_line, color='black', linestyle='--')

        # Show colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity')

        # If save_figure is True, save the figure with a specific file name
        if save_figure:
            file_name_suffix = 'Figure_2D'
            full_file_name = f'{self.file_name}_{file_name_suffix}.jpg'
            save_path = os.path.join(save_to_path, full_file_name) if save_to_path else os.path.join('..', 'test',
                                                                                                     'test_output',
                                                                                                     full_file_name)
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


    def plot_stalta_signal(self)        :
        """
        Plot the signal and STA/LTA ratio with detected events using STA/LTA method
        """





# I made this for the database plotting
def plot_data_for_date(data_df: pd.DataFrame, date_str: str):
    """
    Plots the data for a given date.

    Args:
    data_df (pd.DataFrame): The DataFrame containing the time and signal data.
    date_str (str): The date string in the format 'YYYY-MM-DD' to filter the data.
    """
    # Convert the date string to a datetime object
    date = pd.to_datetime(date_str)

    # Filter the DataFrame for the given date
    filtered_df = data_df[(data_df['time'].dt.date == date.date())]

    if filtered_df.empty:
        print(f"No data available for the date {date_str}")
        return

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_df['time'], filtered_df['signal'], label='Signal')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title(f'Signal Data for {date_str}')
    plt.legend()
    plt.grid(True)

    # Set the x-axis to show time only
    date_format = DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(date_format)

    plt.show()


import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os


def plot_signals_and_stalta(signal, stalta_ratio, window_times, trigger_on, trigger_off, file, output_folder):
    """
    Plots the FO signal and STA/LTA ratio with detected events highlighted and saves the plot.

    Parameters:
        signal (DataFrame): DataFrame containing 'time' and 'signal' columns.
        stalta_ratio (array-like): Array of STA/LTA ratio values.
        window_times (list): List of tuples indicating the start and end times of detected events.
        trigger_on (float): The value for the 'on' trigger line.
        trigger_off (float): The value for the 'off' trigger line.
        file (str): The filename or label for the plot title.
        output_folder (str): The directory where the plot will be saved.
    """
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    # Plot the extended signal
    ax[0].plot(signal['time'], signal['signal'], color='blue')
    # Add shaded areas for detected events
    for window in window_times:
        ax[0].axvspan(window[0], window[1], color='gray', alpha=0.5)
    ax[0].set_title(f'FO signal and STA/LTA ratio for: {file}')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Signal')

    # Plot the STA/LTA ratio
    ax[1].plot(signal['time'], stalta_ratio, color='red')
    # Add horizontal lines for trigger values
    ax[1].axhline(y=trigger_on, color='green', linestyle='--', label='Trigger On')
    ax[1].axhline(y=trigger_off, color='red', linestyle='--', label='Trigger Off')
    ax[1].set_title('STA/LTA Ratio')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Ratio')

    # Format the x-axis to show the time
    date_form = DateFormatter("%H:%M:%S")
    ax[0].xaxis.set_major_formatter(date_form)
    ax[1].xaxis.set_major_formatter(date_form)

    ax[1].legend()  # Show legend for the trigger lines

    plt.tight_layout()

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Construct the filename and save the plot
    filename = f'STA_LTA_Ratio_and_Signal_for_file_{file}.png'
    plt.savefig(os.path.join(output_folder, filename))

    plt.close()

