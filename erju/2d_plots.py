import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

from scipy.signal import butter, filtfilt, iirfilter, sosfilt, zpk2sos




def calculate_sampling_frequency(file: h5py.File) -> float:
    """
    Calculate the sampling frequency from an open HDF5 file by measuring the time interval
    between consecutive samples in the 'RawDataTime' dataset.

    Args:
        file (h5py.File): An open HDF5 file object.

    Returns:
        float: The calculated sampling frequency in Hz.
    """
    try:
        # Access the 'RawDataTime' dataset to get timestamps for calculating sampling interval
        raw_data_time = file['Acquisition']['Raw[0]']['RawDataTime']

        # Calculate time interval between the first two samples in seconds
        time_interval = (raw_data_time[1] - raw_data_time[0]) * 1e-6  # Convert from microseconds to seconds

        # Sampling frequency is the inverse of the time interval
        sampling_frequency = 1 / time_interval
        # Make sampling frequency an integer
        sampling_frequency = int(sampling_frequency)

        return sampling_frequency
    except KeyError:
        raise ValueError("The 'RawDataTime' dataset is missing in the file structure.")
    except IndexError:
        raise ValueError("The 'RawDataTime' dataset has insufficient data for frequency calculation.")



def highpass(data: np.ndarray, cutoff: float = 0.1) -> np.ndarray:
    b, a = butter(1, cutoff, btype="high", analog=False)
    return filtfilt(b, a, data)




def read_and_plot_all_signals(csv_path: Path, data_base_path: Path, save_path: Path):
    """
    Reads a CSV with filenames, opens each file from a base path, and generates a 2D plot for all channels.

    Args:
        csv_path (Path): Path to the CSV containing the list of files.
        data_base_path (Path): Base path where the data files are located.
        save_path (Path): Path to save the plots.
    """
    # Read CSV with filenames
    df = pd.read_csv(csv_path)

    # Ensure the save directory exists
    save_path.mkdir(parents=True, exist_ok=True)

    # Iterate over each file in the CSV and generate the plot
    for index, row in df.iterrows():
        file_name = row['file_name']
        file_path = data_base_path / file_name  # Prepend the base path to the filename

        if file_path.exists():
            try:
                logger.info(f"Processing file: {file_path}")

                # Open the HDF5 file
                with h5py.File(file_path, "r") as f:
                    # Read the data
                    data = f['Acquisition']['Raw[0]']['RawData'][:]  # Get all channels' data

                    # Filter the data with the highpass function
                    # Iterate over each channel (column)
                    for channel in range(data.shape[1]):
                        data[:, channel] = highpass(data[:, channel], cutoff=0.1)

                    # Plotting all channels
                    fig, ax = plt.subplots(figsize=(12, 6))
                    im = ax.imshow(np.log10(np.abs(data)), aspect='auto', cmap='jet')  # log scale for better visibility
                    ax.set_xlabel('Channel count')
                    ax.set_ylabel('Time [s]')

                    # Set the number of ticks based on the dimensions of the data
                    num_channels = data.shape[1]  # Number of channels
                    num_time_points = data.shape[0]  # Number of time points

                    # Set the x-ticks and labels for channels
                    x_ticks = np.linspace(0, num_channels - 1, num=min(10, num_channels))
                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels([f'{int(x)}' for x in x_ticks])

                    # Set the y-ticks and labels for time
                    y_ticks = np.linspace(0, num_time_points - 1, num=min(6, num_time_points))
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels([f'{int(y)}' for y in y_ticks])

                    # Colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Intensity')

                    # Save the plot
                    save_filename = f"2D_plot_{file_path.stem}.png"  # Use the file name without extension
                    save_file_path = save_path / save_filename
                    plt.savefig(save_file_path, dpi=300)
                    logger.info(f"Saved plot for file: {file_path} to {save_file_path}")

                    plt.close(fig)  # Close the figure to avoid memory issues

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        else:
            logger.warning(f"File not found: {file_path}")


# Path to the CSV, data base directory, and save directory
csv_path = Path(r'C:\Projects\erju\outputs\holten\trains_recording_2024-08-29T08_01_16Z_5kHzping_1kHzlog_1mCS_10mGL_3000channels_res250_low0.5_up6.csv')
data_base_path = Path(
    r'C:\Projects\erju\data\holten\recording_2024-08-29T08_01_16Z_5kHzping_1kHzlog_1mCS_10mGL_3000channels')
save_path = Path(r'C:\Projects\erju\outputs\holten\plots')

# Call the function to process files and save plots
read_and_plot_all_signals(csv_path, data_base_path, save_path)
