"""
This script provides utilities for reading, comparing, and visualizing data stored in
gzip-compressed pickle files and NetCDF files. At some point I used this code to come to the conclusion
that in order to create the database, the NetCDF files are the better choice, as they are smaller and faster to read.

It includes functions to:
1. Load and parse compressed pickle (.pkl.gz) and NetCDF (.nc) files containing fiber optic signal data.
2. Compare the contents of both file types (signal arrays and metadata) for consistency.
3. Inspect the structure and attributes of NetCDF files.
4. Plot time-domain signals from both file formats for visual comparison.
5. Optionally plot NetCDF signals using real timestamps from metadata.

Useful for validating data conversion workflows and quickly inspecting FO signal datasets.
"""

import os
import gzip
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from datetime import datetime, timedelta


# ---------------------- FILE READING FUNCTIONS ----------------------

def read_pickle_file(pickle_file_path):
    """
    Reads a gzip-compressed pickle file and returns the data dictionary.

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
    Reads a NetCDF file and returns the data and metadata in a dictionary.

    Args:
        netcdf_file_path (str): Path to the NetCDF file.

    Returns:
        dict: A dictionary containing the signal data and metadata.
    """
    with nc.Dataset(netcdf_file_path, 'r') as dataset:
        return {
            'start_time': dataset.start_time,
            'end_time': dataset.end_time,
            'file': dataset.file,
            'window_index': dataset.window_index,
            'frequency': dataset.frequency,
            'fo_data': dataset.variables['signal'][:]
        }


# ---------------------- DATA COMPARISON AND INSPECTION ----------------------

def compare_files(pickle_data, netcdf_data):
    """
    Compares the data and metadata from the pickle and NetCDF files.

    Args:
        pickle_data (dict): Data from the pickle file.
        netcdf_data (dict): Data from the NetCDF file.

    Returns:
        bool: True if the contents of both files are the same, False otherwise.
    """
    # Compare keys
    pickle_keys = set(pickle_data.keys())
    netcdf_keys = set(netcdf_data.keys())
    if pickle_keys != netcdf_keys:
        print(f"Key mismatch! Pickle keys: {pickle_keys}, NetCDF keys: {netcdf_keys}")
        return False

    # Compare signal data
    signal_equal = np.array_equal(pickle_data['fo_data'], netcdf_data['fo_data'])

    # Compare metadata
    metadata_equal = (
            pickle_data['start_time'].isoformat() == netcdf_data['start_time'] and
            pickle_data['end_time'].isoformat() == netcdf_data['end_time'] and
            pickle_data['file'] == netcdf_data['file'] and
            pickle_data['window_index'] == netcdf_data['window_index'] and
            pickle_data['frequency'] == netcdf_data['frequency']
    )

    return signal_equal and metadata_equal


def print_contents(pickle_data, netcdf_data):
    """
    Prints the contents of the pickle and NetCDF files for inspection.

    Args:
        pickle_data (dict): Data from the pickle file.
        netcdf_data (dict): Data from the NetCDF file.
    """
    print("\n--- Contents of the Pickle file ---")
    for key, value in pickle_data.items():
        print(f"{key}: {value}")

    print("\n--- Contents of the NetCDF file ---")
    for key, value in netcdf_data.items():
        print(f"{key}: {value}")


# ---------------------- PLOTTING FUNCTIONS ----------------------

def plot_time_series(pickle_data, netcdf_data):
    """
    Plots the time series data from both pickle and NetCDF files for comparison.

    Args:
        pickle_data (dict): Data from the pickle file.
        netcdf_data (dict): Data from the NetCDF file.
    """
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(pickle_data['fo_data'], label='Pickle Data')
    plt.title('Pickle Time Series')
    plt.ylabel('Signal')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(netcdf_data['fo_data'], label='NetCDF Data', color='orange')
    plt.title('NetCDF Time Series')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_netcdf_contents(file_path, use_real_time=False):
    """
    Plots the contents of a NetCDF file with time on the x-axis and signal on the y-axis.

    Args:
        file_path (str): Path to the NetCDF file.
        use_real_time (bool): If True, plot the real time using start_time from metadata.
    """
    try:
        with nc.Dataset(file_path, 'r') as ds:
            print(f"\nOpened NetCDF file: {file_path}")

            time_data = ds.variables['time'][:]
            signal_data = ds.variables['signal'][:]

            if use_real_time:
                start_time_str = ds.start_time
                try:
                    start_time = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
                time_data = [start_time + timedelta(seconds=float(t)) for t in time_data]

            plt.figure(figsize=(10, 5))
            plt.plot(time_data, signal_data, label='Signal')
            plt.title('NetCDF Signal Data Over Time')
            plt.xlabel('Real Time' if use_real_time else 'Relative Time')
            plt.ylabel('Signal')
            plt.legend()
            plt.grid(True)
            plt.show()

    except Exception as e:
        print(f"An error occurred while plotting NetCDF contents: {e}")


# ---------------------- NETCDF FILE INSPECTION ----------------------

def read_and_check_netcdf(file_path):
    """
    Reads and prints the structure and contents of a NetCDF file.

    Args:
        file_path (str): Path to the NetCDF file.
    """
    try:
        with nc.Dataset(file_path, 'r') as ds:
            print(f"\nOpened NetCDF file: {file_path}")

            print("\nGlobal attributes:")
            for attr in ds.ncattrs():
                print(f"  {attr}: {getattr(ds, attr)}")

            print("\nDimensions:")
            for dim in ds.dimensions.values():
                print(f"  {dim.name}: size = {dim.size}")

            print("\nVariables:")
            for var_name, var in ds.variables.items():
                print(f"  {var_name}: {var.dimensions}, shape = {var.shape}, dtype = {var.dtype}")
                # Print first 5 sample values
                print(f"    Sample values: {var[:].flatten()[:5]}...")

    except Exception as e:
        print(f"An error occurred while reading the NetCDF file: {e}")


# ---------------------- MAIN FUNCTION ----------------------

def main(pickle_file_path, netcdf_file_path):
    """
    Main function to read, compare, and plot pickle and NetCDF files.

    Args:
        pickle_file_path (str): Path to the gzip-compressed pickle file.
        netcdf_file_path (str): Path to the NetCDF file.
    """
    # Read files and measure load times
    t0 = time.time()
    pickle_data = read_pickle_file(pickle_file_path)
    print(f"Pickle file read in {time.time() - t0:.3f} seconds")

    t1 = time.time()
    netcdf_data = read_netcdf_file(netcdf_file_path)
    print(f"NetCDF file read in {time.time() - t1:.3f} seconds")

    # Print contents
    print_contents(pickle_data, netcdf_data)

    # Compare
    if compare_files(pickle_data, netcdf_data):
        print("\n✅ The pickle and NetCDF files are identical.")
    else:
        print("\n❌ The pickle and NetCDF files are different.")

    # Plot time series
    plot_time_series(pickle_data, netcdf_data)

# ---------------------- EXAMPLE USAGE ----------------------

# Example file paths (update these for your system)
# pickle_file = r"C:\path\to\your\file.pkl.gz"
# netcdf_file = r"C:\path\to\your\file.nc"

# Run main comparison
# main(pickle_file, netcdf_file)

# Inspect and plot a NetCDF file directly
# read_and_check_netcdf(netcdf_file)
# plot_netcdf_contents(netcdf_file, use_real_time=True)
