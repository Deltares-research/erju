import h5py
import obspy
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from obspy import Trace, Stream, UTCDateTime
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
import os

def Convert_3d_to_2d_array(data_3D):
    """
    DESCRIPTION:
    -----------------------
    Basic function for converting 3D numpy array into a 2D array.

    INPUT PARAMETERS:
    -----------------------
    data_3D: a 3d numpy array containing time domain signals

    RETURNS:
    -----------------------
    data_2D a 2d array
    """
    data_2D = data_3D.transpose(2, 0, 1).reshape(-1, data_3D.shape[1])
    return data_2D


def raw2strain(trace, n, L):
    """
    DESCRIPTION:
    -----------------------
    Function for converting raw DAS data into strain data after applying all OptaSense parameters.
    This function is written based on: https://github.com/ethanfwilliams/OOI_RCA_DAS_notebook/blob/main/OOI_RCA_OptaSense_DAS_intro.ipynb.

    INPUT PARAMETERS:
    -----------------------
    trace: 1d array coontaning a single trace (time domain)
    n    : index of refraction typically assigned as 1.46
    L    : Gauge lengt in meters

    RETURNS:
    -----------------------
    trace_strain  : 1d array containing a strain values

    """
    # Remove the mean from the trace
    trace = trace.astype(float)
    trace -= np.mean(trace)
    # Convert to units of radians
    trace *= (2 * np.pi) / 2 ** 16
    # Convert to units of strain
    trace *= (1550.12 * 1e-9) / (0.78 * 4 * np.pi * n * L)
    trace_strain = trace
    return trace_strain


def Array_to_ObsPy(data, fs):
    """
    DESCRIPTION:
    -----------------------
    Function for indexing traces into an Obspy object. This enables the possibility of using the Obspy processing features.

    INPUT PARAMETERS:
    -----------------------
    data  : 2d array coontaning a N number of traces
    fs    : sampling frequency in Hz

    RETURNS:
    -----------------------
    st  : Obspy object containing N number traces and sampling frequency
    """
    size = np.shape(data)  # Determining number of samples and traces in data
    st = Stream()  # Creating a list of multiple obspy traces objects

    # Loop to assign input traces inside st object
    for ii in range(0, size[1]):
        trace = Trace(header={'station': 'DAS', 'channel': 'Z' + str(ii)})
        trace.data = data[:, ii]
        year = 2022  # year -- Default value
        month = 1  # month -- Default value
        day = 1  # day -- Default value
        hours = 12  # hours -- Default value
        minutes = 0  # minutes -- Default value
        seconds = 0  # seconds -- Default value

        # Assign defaults values for date and time attributes
        trace.stats.starttime = UTCDateTime(year, month, day, hours, minutes, seconds)
        trace.stats.npts = len(trace.data)  # Assign number of samples per trace
        trace.stats.sampling_rate = fs  # Assign sampling frequency in Hz
        trace.stats.station = str(ii)  # Assign default station number
        trace.stats.channel = 'Z' + str(ii)  # Assign channel number
        trace.stats.distance = ii  # Assign default distance
        trace.stats.network = 'DAS'  # Assign default network name

        #  Create multi-trace Obspy object
        st.append(trace)

    return (st)


def Obspy_to_Array(st):
    """
    DESCRIPTION:
    -----------------------
    Function for extracting signals (processed) from the Obspy object.

    INPUT PARAMETERS:
    -----------------------
    st  : Obspy object containing N number traces and sampling frequency

    RETURNS:
    -----------------------
    Array  : 2d-numpy array contaning a N number of traces
    """
    size = np.shape(st)  # Defining the array dimenions in the st object
    Array = np.zeros((size[1], size[0]))  # Declaring the 2d array
    # Loop for assigning trace values inside Array
    for ii in range(0, size[0]):
        Array[:, ii] = st[ii].data

    return Array


def filtering(data, fs, fmin, fmax, filter_type='bandpass'):
    """
    DESCRIPTION:
    -----------------------
    Function for filtering raw signals. For this we use a Butterworth filter type of order 4 and zero-phase.

    INPUT PARAMETERS:
    -----------------------
    data  : 2d-numpy array contaning traces.
    fs    : Sampling frequency in Hz.
    fmin  : Minimum frequency in Hz.
    fmax  : Maxmimum frequency in Hz.
    filter_type = "bandpass","lowpass", and "highpass". For highpass filter, the corner frequency, freq, is set to fmin, so all frequencies bellow fmin are removed and fmax is not used, while
    for lowpass filter, the corner frequency, freq, is set to fmax, so all frequencies above fmax are removed and fmin is not used. We use filter_type="bandpass" as Default.

    RETURNS:
    -----------------------
    filt_data  : 2d-numpy array contaning band-pass filtered traces
    """
    st2filt = Array_to_ObsPy(data, fs)  # Converting data into an Obspy object
    st_f = st2filt.copy()
    st_f.detrend('constant')  # Signal detrend
    st_f.taper(0.01, type='cosine')  # Cosine taper of 1%
    # Implemente filtering based on filtering type selection.
    if filter_type == "bandpass":
        st_f.filter(filter_type, freqmin=fmin, freqmax=fmax, corners=4,
                    zerophase=True)  # Band-pass filter of butterworth type, order 4 and zero-phase
    elif filter_type == "lowpass":
        st_f.filter(filter_type, freq=fmax, corners=4,
                    zerophase=True)  # Low-pass filter of butterworth type, order 4 and zero-phase
    elif filter_type == "highpass":
        st_f.filter(filter_type, freq=fmin, corners=4,
                    zerophase=True)  # High-pass filter of butterworth type, order 4 and zero-phase
    else:
        print("please, select a proper filter type e.g. bandpass,lowpass or highpass")

    filt_data = Obspy_to_Array(st_f)  # Extracting filtered traces

    return filt_data


def open_OptaSense_file(file_path):
    """
    DESCRIPTION:
    -----------------------
    Function for opening OptaSense DAS data files.

    INPUT PARAMETERS:
    -----------------------
    file_path  : Path to the OptaSense DAS data file

    RETURNS:
    -----------------------
    fp  : List with all file properties
    """
    fp = h5py.File(file_path, 'r')  # Open the OptaSense DAS data file
    return fp


def OptaSense_Props(fp):
    """
    DESCRIPTION:
    -----------------------
    Function for inspecting main properties of the OptaSense DAS data. This function is written based on:
    https://github.com/ethanfwilliams/OOI_RCA_DAS_notebook/blob/main/OOI_RCA_OptaSense_DAS_intro.ipynb.

    INPUT PARAMETERS:
    -----------------------
    fp  : list with all file properties

    RETURNS:
    -----------------------
    dt : sampling interval of the DAS data in seconds
    fs : sampling frequency of the DAS data in Hz
    dx : channel spacing in meters
    nx : number of channels
    ns : number of samples
    """
    # Print DAS acqusition parameters
    print('\nThe gauge length is ', fp['Acquisition'].attrs['GaugeLength'],
          'm.')  # Print gauge lengt in meters.
    print('The pulse rate is ', fp['Acquisition'].attrs['PulseRate'], 'Hz.')  # Print pulse rate in Hz
    print('The spatial sampling interval is ', fp['Acquisition'].attrs['SpatialSamplingInterval'],
          'm.')  # Print spatial sampling interval

    # Print date/time
    rawDataTime = fp['Acquisition']['Raw[0]']['RawDataTime']  # Print the first entry in "RawDataTime"
    print('The first value in \"RawDataTime\" is', rawDataTime[0], \
          'which is the timestamp of the first sample in microseconds.')
    print('This equates to the date and time', datetime.utcfromtimestamp(rawDataTime[0] * 1e-6))  # Print UTC timestamp
    rawDataTimeArr = rawDataTime[:]  # We can load the full contents of "RawDataTime" into a NumPy array
    print('\nThere are', len(rawDataTimeArr), 'values in \"RawDataTime\"')  # Print the length of the array
    dt = (rawDataTime[1] - rawDataTime[0]) * 1e-6  # Get the sample interval either from the time stamps
    fs = fp['Acquisition']['Raw[0]'].attrs['OutputDataRate']  # Get sampling frequency, fs
    dx = fp['Acquisition'].attrs['SpatialSamplingInterval']  # Get channel spacing in m
    nx = fp['Acquisition']['Raw[0]'].attrs['NumberOfLoci']  # Get number of channels
    ns = fp['Acquisition']['Raw[0]']['RawDataTime'].attrs['Count']  # Get number of samples
    print('The sampling frequency is', fs, 'Hz.')  # Print the sampling frequency in Hz
    print('Number of channels is', nx)  # Print the number of traces

    return (dt, fs, dx, nx, ns)


def all_strain(fp, channel0, channel1, fmin, fmax, filter_type='bandpass'):
    """
    DESCRIPTION:
    -----------------------
    Function for extracting and filtering raw signals bounded by channels0 and channel1. This function is written based on:
    https://github.com/ethanfwilliams/OOI_RCA_DAS_notebook/blob/main/OOI_RCA_OptaSense_DAS_intro.ipynb.

    INPUT PARAMETERS:
    -----------------------
    fp       : List with all file properties
    channel0 : First channel
    channel1 : Last channel

    RETURNS:
    -----------------------
    strain : 2d-numpy array with strain values
    rawData_sel : 2d-numpy array with band-pass filtered raw data
    """
    #     rawData = fp['Acquisition']['Raw[0]']['RawData']                               # Extracting raw data from fp dictionary
    rawData_sel = fp['Acquisition']['Raw[0]']['RawData'][:, channel0:channel1]
    fs = fp['Acquisition']['Raw[0]'].attrs['OutputDataRate']  # Sampling frequency in Hz
    #     rawData_sel = rawData[:, channel0:channel1]                                    # Selecting rawData traces bounded by channel0 and channel1
    rawData_filt = filtering(rawData_sel, fs, fmin, fmax,
                             filter_type=filter_type)  # Band-pass filtered traces bounded by channel0 and channel1
    n = fp['Acquisition']['Custom'].attrs['Fibre Refractive Index']  # Extracting n factor from fp
    L = fp['Acquisition'].attrs['GaugeLength']  # Extracting L value from fp

    # Loop to convert raw data signals into strain
    strain = np.zeros((np.shape(rawData_sel)))  # Declaring strain array to store computed strain signals

    for col_num in range(np.shape(rawData_filt)[1]):
        trace = rawData_filt[:, col_num]
        trace_strain = raw2strain(trace, n, L)
        strain[:, col_num] = trace_strain

    return (strain, rawData_filt, rawData_sel)


def stalta(sta, lta, fs):
    '''
    INPUT PARAMETERS:
    -----------------------
    nsta (int) : Length of short time average window in samples
    nlta (int) : Length of long time average window in samples
    RETURNS:
    -----------------------
    Characteristic function of classic STA/LTA
    '''

    return (int(sta * fs), int(lta * fs))


def getON(signal, fs, ymin, ymax, sta, lta):
    '''
    Para calcular los tiempos inicial de cada se;al

    INPUT PARAMETERS:
    -----------------------
    data  : 1d-numpy array contaning traces.
    fs    : Sampling frequency in Hz.
    ymin  : Value below which trigger (of characteristic function) is deactivated (lower threshold)
    ymax  : Value above which trigger (of characteristic function) is activated (higher threshold)
    sta (float) : Length of short time average window in seconds
    lta (float) : Length of long time average window in seconds

    RETURNS:
    -----------------------
    Nested List of trigger on and of times in samples

    '''
    # Characteristic function and trigger onsets
    nsta, nlta = stalta(sta, lta, fs)
    cft = recursive_sta_lta(signal, nsta, nlta)
    on_of = trigger_onset(cft, ymax, ymin)
    return on_of, cft


def getON_seg(signal, fs, ymin, ymax, sta, lta):
    '''
    INPUT PARAMETERS:
    -----------------------
    signal  : 1d-numpy array contaning traces.
    fs    : Sampling frequency in Hz.
    ymin  : Value below which trigger (of characteristic function) is deactivated (lower threshold)
    ymax  : Value above which trigger (of characteristic function) is activated (higher threshold)
    sta (float) : Length of short time average window in seconds
    lta (float) : Length of long time average window in seconds

    RETURNS:
    -----------------------
    Nested List of trigger on and of times in samples

    '''
    # Characteristic function and trigger onsets
    nsta, nlta = stalta(sta, lta, fs)
    cft = recursive_sta_lta(signal, nsta, nlta)
    on_of = trigger_onset(cft, ymax, ymin)
    return on_of, cft


def plotONOF(signal, cft, on_of):
    # Plotting the results
    ax = plt.subplot(211)
    plt.plot(signal, 'k')
    ymin, ymax = ax.get_ylim()
    plt.vlines(on_of[:, 0], ymin, ymax, color='r', linewidth=2)
    plt.vlines(on_of[:, 1], ymin, ymax, color='b', linewidth=2)
    plt.subplot(212, sharex=ax)
    plt.plot(cft, 'k')
    plt.hlines([7, 0.5], 0, len(cft), color=['r', 'b'], linestyle='--')
    plt.axis('tight')
    plt.show()


def getShift(data1, data2, sel_time, fs, ymax, ymin, sta, lta):
    '''
    INPUT PARAMETERS:
    -----------------------
    data1  : 1d-numpy array.
    data2  : 1d-numpy array.
    fs    : Sampling frequency in Hz.
    sel_time : Window contaning the first signal (train passage) from each record
    ymin  : Value below which trigger (of characteristic function) is deactivated (lower threshold)
    ymax  : Value above which trigger (of characteristic function) is activated (higher threshold)
    sta (float) : Length of short time average window in seconds
    lta (float) : Length of long time average window in seconds

    RETURNS:
    -----------------------
    Shifted traces

    data_shifted1: 1d-numpy array.
    data_shifted2: 1d-numpy array.

    '''

    # Characteristic function and trigger onsets
    on_of1, _ = getON(data1[:sel_time], fs, ymin, ymax, sta, lta)
    on_of2, _ = getON(data2[:sel_time], fs, ymin, ymax, sta, lta)
    shift = np.min(on_of1) - np.min(on_of2)

    if shift >= 0:
        data_shifted1 = data1[shift:]
        data_shifted2 = data2[:len(data_shifted1)]

    else:
        data_shifted2 = data2[np.abs(shift):]
        data_shifted1 = data1[:len(data_shifted2)]

    return data_shifted1, data_shifted2