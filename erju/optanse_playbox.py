import os
import h5py
from utils.file_utils import get_files_in_dir
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirfilter, sosfilt, zpk2sos, windows

from utils.file_utils import highpass

# Define the directory path
dir_path = r'C:\Projects\erju\data\holten\recording_2024-08-29T08_01_16Z_5kHzping_1kHzlog_1mCS_10mGL_3000channels'
# Get the file names in the directory
file_names = get_files_in_dir(folder_path=dir_path, file_format='.h5')
file_path = os.path.join(dir_path, file_names[1])


def bandpass(data, freqmin, freqmax, fs, corners, zerophase=True):
    fe = 0.5 * fs
    low = freqmin / fe
    high = freqmax / fe
    z, p, k = iirfilter(corners, [low, high], btype='band', ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)

    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)

def from_opticalphase_to_strain(raw_data: np.ndarray) -> np.ndarray:
    """
    Take the raw OptaSense data and convert it to units of strain.

    Args:
        signal_data (np.ndarray): The raw OptaSense data

    Returns:
        data (np.ndarray): The strain data
    """
    # Remove the mean from the data. Since it is a 2D of (150000>time, 5000>location) [rows, columns]
    # We remove the mean over time for each location with axis=0 (operation over rows)
    raw_data = raw_data - np.mean(raw_data, axis=0)

    # Convert into units of radians
    raw_data = raw_data * (2*np.pi / 2**16)

    # Get from the properties the values I need to convert to strain
    n = file['Acquisition']['Custom'].attrs['Fibre Refractive Index']
    L = file['Acquisition'].attrs['GaugeLength']
    # Convert into units of strain
    data = raw_data * ((1550.12 * 1e-9)/(0.78 * 4 * np.pi * n * L))

    return data


# Set the start and end channels for plotting
start_ch = 2000
end_ch = 2005

# Open the .h5 file
with h5py.File(file_path, 'r') as file:
    rawData = file['Acquisition']['Raw[0]']['RawData']
    fs = file['Acquisition']['Raw[0]'].attrs['OutputDataRate']
    num_outputs = file['Acquisition']['Custom'].attrs['Num Output Channels']
    num_measurements = file['Acquisition']['Raw[0]']['RawData'].shape[0]

    signals = rawData[:]
    window = windows.tukey(M=num_measurements, alpha=0.1)

    # Apply bandpass filter to all channels
    filt_array = np.zeros(np.shape(signals))
    fft_sig = np.zeros(np.shape(signals))
    for i in range(num_outputs):
        signal_filt = bandpass(data=window * signals[:, i],
                               freqmin=1,
                               freqmax=100,
                               fs=fs,
                               corners=5,
                               zerophase=True)
        filt_array[:, i] = signal_filt
        fft_sig[:, i] = np.abs(np.fft.fft(signal_filt))
        frequency = np.linspace(0, fs, num_measurements)

    processed_data = from_opticalphase_to_strain(filt_array)

    # Plot the raw and filtered data in a 1x2 subplot
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 5))
    cbar0 = ax0.imshow(np.log10(np.abs(signals)), aspect='auto', cmap='jet')
    fig.colorbar(cbar0, ax=ax0)
    ax0.set_title('RAW')
    cbar1 = ax1.imshow(np.log10(np.abs(processed_data)), aspect='auto', cmap='jet')
    fig.colorbar(cbar1, ax=ax1)
    ax1.set_title('FILTERED from 1 Hz - 100 Hz')
    plt.show()
    plt.close()

    # Plot the specified range of channels in separate subplots
    num_ch_to_plot = end_ch - start_ch + 1
    fig, axes = plt.subplots(num_ch_to_plot, 2, figsize=(10, 2 * num_ch_to_plot), sharex=True)

    for i, ch in enumerate(range(start_ch, end_ch + 1)):
        axes[i, 0].plot(filt_array[:, ch])
        axes[i, 0].set_title(f'Trace {ch}')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True)

        axes[i, 1].plot(frequency, fft_sig[:, ch])
        axes[i, 1].set_title(f'Trace {ch}')
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].grid(True)


    plt.xlabel('Sample')
    plt.tight_layout()
    plt.show()
