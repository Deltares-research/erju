import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from protocol.message import FilePackageReader, RSMessage
from protocol.envelope import ReceiveServerOutputEnvelope, IndusDASEnvelope
from protocol.message.rs_file_reader import open_rs_file

# Define the path to the binary data file
file_path = r'C:\Projects\erju\data\focus\sensor-1-full-00024170-10181607-10181806 1.bin'


def read_data(file_path):
    """
    Reads the binary data file and returns the header, all_data, and signals_sum.

    Parameters:
        file_path (str): The path to the binary data file.

    Returns:
        header (dict): The header information decoded from the file.
        all_data (numpy.ndarray): The payload data from the file.
        signals_sum (numpy.ndarray): The summary data from the file.
    """
    # Initialize the file package reader and envelope decoder
    package_reader = FilePackageReader()
    indus_das_envelope = IndusDASEnvelope()

    # Open the binary data file
    package_reader.open_file(file_path)

    # Initialize variables to store data
    all_data = []
    nn = 0
    header = None
    signals_sum = None

    # First pass: Count the number of packages
    package_count = 0
    for offset in package_reader.yield_package_offsets_from_file():
        package_count += 1

    # Reset the file reader to the beginning
    package_reader.open_file(file_path)

    # Read and process packages from the file
    for offset in package_reader.yield_package_offsets_from_file():
        # Decode the header information
        header = indus_das_envelope.decode(package_reader.mm[
                                           package_reader.indus_offset:package_reader.indus_offset + IndusDASEnvelope.SUMMARY_DATA_OFFSET])

        # Get the full package data
        rs_message = RSMessage()
        full_data = rs_message.get_full_package(package_reader.mm[offset:offset + package_reader.package_length])

        # Initialize signals_sum based on the first package's summary data length
        if signals_sum is None:
            summary_length = len(full_data['summary'])
            signals_sum = np.zeros((summary_length, package_count))

        # Append payload data to the list
        all_data.append(full_data['payload'])

        # Store summary data in the signals_sum array
        signals_sum[:, nn] = full_data['summary']
        nn += 1

    # Convert list to numpy array for easier processing
    all_data = np.array(all_data)

    # Close the memory-mapped file
    #package_reader.close_file()

    return header, all_data, signals_sum


def extract_properties_from_header(header):
    """
    Extracts the file properties as a dictionary from the given header.

    Args:
        header (dict): The header information decoded from the file.

    Returns:
        properties (dict): The extracted properties.
    """
    properties = {
        'signature': header['signature'],
        'package_length': header['package_length'],
        'protocol_version': header['protocol_version'],
        'software_version': header['software_version'],
        'firmware_version': header['firmware_version'],
        'serial_no': header['serial_no'],
        'source_type': header['source_type'],
        'summary_type': header['summary_type'],
        'payload_type': header['payload_type'],
        'bps_source': header['bps_source'],
        'bps_summary': header['bps_summary'],
        'bps_payload': header['bps_payload'],
        'timestamp': header['timestamp'],
        'gps_counter': header['gps_counter'],
        'sampling_freq': header['sampling_freq'],
        'channel_num': header['channel_num'],
        'sampling_num': header['sampling_num'],
        'data_type_id': header['data_type_id'],
        'channel_spacing': header['channel_spacing'],
        'start': header['start'],
        'conversion_factor': header['conversion_factor'],
        'hpf_id': header['hpf_id'],
        'frame_count_err': header['frame_count_err'],
        'first_frame_no': header['first_frame_no'],
        'last_frame_no': header['last_frame_no'],
        'last_frame_timestamp': header['last_frame_timestamp'],
        'last_frame_ns': header['last_frame_ns'],
        'frame_decimation': header['frame_decimation'],
        'spatial_decimation': header['spatial_decimation'],
        'frames_in_package': header['frames_in_package'],
        'fpga_reg': header['fpga_reg'],
        'pcie_config': header['pcie_config'],
        'gpsCounter': header['gpsCounter'],
        'samplingFreq': header['samplingFreq'],
        'channelNum': header['channelNum'],
        'frameDecimation': header['frameDecimation']
    }

    # Calculate additional properties
    properties['TimeInterval'] = 1 / properties['sampling_freq']
    properties['MeasurementDuration'] = properties['TimeInterval'] * properties['frames_in_package']
    properties['FileStartTime'] = datetime.utcfromtimestamp(properties['timestamp'])
    time_delta = timedelta(seconds=properties['MeasurementDuration'])
    properties['FileEndTime'] = properties['FileStartTime'] + time_delta

    return properties


def plot_all_data(header, all_data):
    """
    Plots the first payload data (all_data).

    Parameters:
        header (dict): The header information containing the conversion factor.
        all_data (numpy.ndarray): The payload data to be plotted.
    """
    # Flatten the first payload data for plotting
    signal = all_data[0, ...].flatten()
    # Create a figure and axis for the plot
    fig, ax0 = plt.subplots(figsize=(10, 5))
    # Retrieve the conversion factor from the header
    factor = header['conversion_factor']
    # Plot the signal data
    ax0.plot(signal / factor)
    # Set the labels for the plot
    ax0.set_ylabel('Phase (rads)')
    ax0.set_xlabel('Samples')
    # Set the title for the plot
    plt.title('First Payload Data')
    # Display the plot
    plt.show()


def plot_time_vs_channels(header, signals_sum):
    """
    Plots the 2D time vs channels plot.

    Parameters:
        header (dict): The header information containing the frame decimation.
        signals_sum (numpy.ndarray): The summary data to be plotted.
    """
    # Calculate the frame decimation sampling frequency
    fs_dec = header['sampling_freq'] / header['frameDecimation']
    # Create a figure with two subplots for linear and log-scale plots
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 8))
    # Plot the linear scale summary data
    ax0.imshow(signals_sum, aspect='auto', cmap='seismic_r')
    ax0.set_ylim(0, 4000)
    ax0.set_title("2D Time vs Channels")
    # Plot the log-scale summary data
    ax1.imshow(np.log(np.abs(signals_sum)), aspect='auto', cmap='seismic_r')
    ax1.set_ylim(0, 4000)
    ax1.set_title("Using log-scale")
    # Display the plots
    plt.show()


def plot_single_channel(header, signals_sum):
    """
    Plots the single channel plots.

    Parameters:
        header (dict): The header information.
        signals_sum (numpy.ndarray): The summary data containing channel information.
    """
    # Create a figure with subplots for each channel
    fig, axs = plt.subplots(5, 1, figsize=(8, 10))
    # Iterate over the first 5 channels for plotting
    for i in range(5):
        # Retrieve the data for the current channel
        channel_data = signals_sum[:, i]
        # Check if the channel data is not all zeros
        if np.max(np.abs(channel_data)) != 0:
            # Normalize the channel data
            normalized_data = channel_data / np.max(np.abs(channel_data))
            # Plot the normalized data
            axs[i].plot(normalized_data)
            axs[i].set_xlim(500, 4000)
            axs[i].set_ylim(-0.2, 0.2)
            axs[i].set_title(f'Normalized amplitudes -- Channel {i + 1}')
        else:
            # Display 'No data' if the channel data is all zeros
            axs[i].text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
    # Adjust layout to prevent overlap
    fig.tight_layout()
    # Display the plots
    plt.show()


# Read the data from the file
header, all_data, signals_sum = read_data(file_path)

# Extract the properties from the header
properties = extract_properties_from_header(header)
print(f"Properties: {properties}")

# Plot the first payload data (all_data)
plot_all_data(header, all_data)

# Plot the 2D time vs channels plot
plot_time_vs_channels(header, signals_sum)

# Plot the single channel plots
plot_single_channel(header, signals_sum)
