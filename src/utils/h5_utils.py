import h5py
import numpy as np
from datetime import datetime



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


def convert_microseconds_to_datetime(microseconds: int) -> datetime:
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


def from_opticalphase_to_strain(raw_data: np.ndarray, fibre_refractive_index, gauge_length):
    """
    Take the raw OptaSense data and convert it to units of strain.

    Args:
        raw_data (np.ndarray): The raw OptaSense data

    Returns:
        data (np.ndarray): The strain data
    """
    # Remove the mean from the data. Since it is a 2D of (150000>time, 5000>location) [rows, columns]
    # We remove the mean over time for each location with axis=0 (operation over rows)
    print(f"raw_data shape: {raw_data.shape}")


    raw_data = raw_data - np.mean(raw_data, axis=0)

    # Convert into units of radians
    raw_data = raw_data * (2*np.pi / 2**16)

    # Get from the properties the values I need to convert to strain
    n = fibre_refractive_index
    L = gauge_length
    # Convert into units of strain
    data = raw_data * ((1550.12 * 1e-9)/(0.78 * 4 * np.pi * n * L))

    return data