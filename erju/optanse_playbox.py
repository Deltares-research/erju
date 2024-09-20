import os
import h5py
from utils.file_utils import get_files_in_dir
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from scipy.signal import iirfilter, sosfilt, zpk2sos, sosfilt, windows

# Define the directory path
dir_path = r'C:\Projects\erju\data\holten\samples'
# Get the file names in the directory
file_names = get_files_in_dir(folder_path=dir_path, file_format='.h5')
file_path = os.path.join(dir_path, file_names[1])

def bandpass(data, freqmin, freqmax, fs, corners, zerophase=True):
    """
    Apply a bandpass filter to the data.

    Args:
        data (np.array): The data to be filtered.
        freqmin (float): The lower frequency bound of the filter.
        freqmax (float): The upper frequency bound of the filter.
        fs (float): The sampling frequency.
        corners (int): The number of corners in the filter.
        zerophase (bool): Whether to apply the filter in both directions.

    Returns:
        np.array: The filtered data
    """
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



# Open the .h5 file
with h5py.File(file_path, 'r') as file:
    # Create a new variable for the "RawData" h5py dataset
    rawData = file['Acquisition']['Raw[0]']['RawData']
    fs = file['Acquisition']['Raw[0]'].attrs['OutputDataRate']
    num_outputs = file['Acquisition']['Custom'].attrs['Num Output Channels']
    print('Number of outputs: ', num_outputs)
    num_measurements = file['Acquisition']['Raw[0]']['RawData'].shape[0]
    print('Number of measurements: ', num_measurements)

    signals = rawData[:]

    window = windows.tukey(M=num_measurements, alpha=0.1)

    filt_array = np.zeros(np.shape(signals))

    for i in range(num_outputs):
        signal_filt = bandpass(data=window * signals[:, i],
                               freqmin=1,
                               freqmax=100,
                               fs=fs,
                               corners=5,
                               zerophase=True)

        filt_array[:, i] = signal_filt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 5))

    cbar0 = ax0.imshow(np.log10(np.abs(signals)), aspect='auto', cmap='jet')
    fig.colorbar(cbar0, ax=ax0)
    ax0.set_title('RAW')
    cbar1 = ax1.imshow(np.log10(np.abs(filt_array)), aspect='auto', cmap='jet')
    fig.colorbar(cbar1, ax=ax1)
    ax1.set_title('FILTERED -- 0.1 Hz - 100 Hz')
    plt.show()
    plt.close()

    #plt.plot(signals[:, 2000])
    plt.plot(filt_array[:, 2000])
    plt.title('Trace 2000')
    plt.show()
