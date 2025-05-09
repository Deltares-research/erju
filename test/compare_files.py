import os
import pickle
import gzip  # Import gzip for compression
import numpy as np
import time
import matplotlib.pyplot as plt
import netCDF4 as nc
import matplotlib.pyplot as plt
import netCDF4 as nc
from datetime import datetime, timedelta


def read_pickle_file(pickle_file_path):
    """
    Reads the gzip-compressed pickle file and returns the data dictionary.

    Args:
        pickle_file_path (str): Path to the gzip-compressed pickle file.

    Returns:
        dict: The data stored in the pickle file.
    """
    with gzip.open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def read_netcdf_file(netcdf_file_path):
    """
    Reads the NetCDF file and returns the data and metadata in a dictionary.

    Args:
        netcdf_file_path (str): Path to the NetCDF file.

    Returns:
        dict: A dictionary containing the signal data and metadata.
    """
    with nc.Dataset(netcdf_file_path, 'r') as dataset:
        data_dict = {
            'start_time': dataset.start_time,
            'end_time': dataset.end_time,
            'file': dataset.file,
            'window_index': dataset.window_index,
            'frequency': dataset.frequency,
            'fo_data': dataset.variables['signal'][:]
        }
    return data_dict


def compare_files(pickle_data, netcdf_data):
    """
    Compares the data and metadata from the pickle and NetCDF files.

    Args:
        pickle_data (dict): Data from the pickle file.
        netcdf_data (dict): Data from the NetCDF file.

    Returns:
        bool: True if the contents of both files are the same, False otherwise.
    """
    # Compare the set of keys between both dictionaries
    pickle_keys = set(pickle_data.keys())
    netcdf_keys = set(netcdf_data.keys())

    if pickle_keys != netcdf_keys:
        print(f"Key mismatch! Pickle keys: {pickle_keys}, NetCDF keys: {netcdf_keys}")
        return False

    # Compare signal data (fo_data)
    signal_equal = np.array_equal(pickle_data['fo_data'], netcdf_data['fo_data'])

    # Compare metadata
    metadata_equal = (pickle_data['start_time'].isoformat() == netcdf_data['start_time'] and
                      pickle_data['end_time'].isoformat() == netcdf_data['end_time'] and
                      pickle_data['file'] == netcdf_data['file'] and
                      pickle_data['window_index'] == netcdf_data['window_index'] and
                      pickle_data['frequency'] == netcdf_data['frequency'])

    return signal_equal and metadata_equal


def print_contents(pickle_data, netcdf_data):
    """
    Prints the contents of the pickle and NetCDF files for inspection.

    Args:
        pickle_data (dict): Data from the pickle file.
        netcdf_data (dict): Data from the NetCDF file.

    Returns:
        None
    """
    print("\nContents of the Pickle file:")
    for key, value in pickle_data.items():
        print(f"{key}: {value}")

    print("\nContents of the NetCDF file:")
    for key, value in netcdf_data.items():
        print(f"{key}: {value}")


def main(pickle_file_path, netcdf_file_path):
    """
    Main function to read and compare the pickle and NetCDF files.

    Args:
        pickle_file_path (str): Path to the gzip-compressed pickle file.
        netcdf_file_path (str): Path to the NetCDF file.

    Returns:
        None
    """
    # Read the pickle and NetCDF files and calculate the time needed to read each file
    # Start the timer
    pickle_start_time = time.time()
    pickle_data = read_pickle_file(pickle_file_path)
    pickle_end_time = time.time()
    pickle_time = pickle_end_time - pickle_start_time
    print(f"Time taken to read the pickle file: {pickle_time:.4f} seconds")
    # Start the timer
    netcdf_start_time = time.time()
    netcdf_data = read_netcdf_file(netcdf_file_path)
    netcdf_end_time = time.time()
    netcdf_time = netcdf_end_time - netcdf_start_time
    print(f"Time taken to read the NetCDF file: {netcdf_time:.4f} seconds")


    # Print the contents of both files
    print_contents(pickle_data, netcdf_data)

    # Compare the data
    files_are_equal = compare_files(pickle_data, netcdf_data)

    if files_are_equal:
        print("\nThe contents of the pickle and NetCDF files are identical.")
    else:
        print("\nThe contents of the pickle and NetCDF files are different.")

    # Plot the time series of both files in a 2,1 column plot
    pickle_timeseries = pickle_data['fo_data']
    netcdf_timeseries = netcdf_data['fo_data']
    # Plot the time series
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(pickle_timeseries, label='Pickle Data')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Pickle Time Series')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(netcdf_timeseries, label='NetCDF Data')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('NetCDF Time Series')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()



# Example usage
#pickle_file = r'C:\Projects\erju\outputs\culemborg\20201120_094844413777.pkl.gz'  # Updated to .gz extension
netcdf_file = r'C:\Projects\erju\outputs\culemborg\20201120_094844413777.nc'

#main(pickle_file, netcdf_file)

# New check netcdf file function

def read_and_check_netcdf(file_path):
    """
    Reads and checks the structure and contents of a NetCDF file.

    Args:
        file_path (str): Path to the NetCDF file.

    Returns:
        None
    """
    try:
        # Open the NetCDF file
        with nc.Dataset(file_path, 'r') as ds:
            print(f"\nOpened NetCDF file: {file_path}")

            # Print global attributes
            print("\nGlobal attributes:")
            for attr in ds.ncattrs():
                print(f"  {attr}: {getattr(ds, attr)}")

            # Print dimensions
            print("\nDimensions:")
            for dim in ds.dimensions.values():
                print(f"  {dim.name}: size = {dim.size}")

            # Print variables
            print("\nVariables:")
            for var_name, var in ds.variables.items():
                print(f"  {var_name}: {var.dimensions}, shape = {var.shape}, dtype = {var.dtype}")

            # Optionally read a sample of data from each variable
            print("\nSample data from variables:")
            for var_name, var in ds.variables.items():
                data = var[:]
                print(f"  {var_name}:")
                print(f"    Data shape: {data.shape}")
                print(f"    Sample values: {data.flatten()[:5]}...")  # Print the first 5 values

            print("\nNetCDF file check completed successfully.")

    except Exception as e:
        print(f"An error occurred while reading the NetCDF file: {e}")




import matplotlib.pyplot as plt
import netCDF4 as nc
from datetime import datetime, timedelta

def plot_netcdf_contents(file_path, use_real_time=False):
    """
    Plots the contents of a NetCDF file with time on the x-axis and signal on the y-axis.

    Args:
        file_path (str): Path to the NetCDF file.
        use_real_time (bool): If True, plot the real time using the start_time in the NetCDF file. Otherwise, plot relative time starting from zero.

    Returns:
        None
    """
    try:
        # Open the NetCDF file
        with nc.Dataset(file_path, 'r') as ds:
            print(f"\nOpened NetCDF file: {file_path}")

            # Extract time and signal data
            time_data = ds.variables['time'][:]
            signal_data = ds.variables['signal'][:]

            if use_real_time:
                # Extract start_time and convert to datetime
                start_time_str = ds.start_time
                try:
                    start_time = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")

                # Calculate real time
                real_time_data = [start_time + timedelta(seconds=float(t)) for t in time_data]
                time_data = real_time_data

            # Plot the signal data over time
            plt.figure(figsize=(10, 5))
            plt.plot(time_data, signal_data, label='Signal')
            plt.xlabel('Time' if not use_real_time else 'Real Time')
            plt.ylabel('Signal')
            plt.title('Signal Data Over Time')
            plt.legend()
            plt.grid(True)
            plt.show()

            print("\nNetCDF file plot completed successfully.")

    except Exception as e:
        print(f"An error occurred while reading the NetCDF file: {e}")



netcdf_file = r"N:\Projects\11210000\11210064\B. Measurements and calculations\holten\fo_plot\event_20240907T060629.nc"

read_and_check_netcdf(netcdf_file)
plot_netcdf_contents(netcdf_file, use_real_time=True)

