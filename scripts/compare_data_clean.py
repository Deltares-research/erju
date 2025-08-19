import time
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from src.utils.db_utils import fetch_accel_data, unpack_timeseries, estimate_sampling_frequency, unpack_accel_data

from SignalProcessingTools.time_signal import TimeSignalProcessing, IntegrationRules, Windows, FilterDesign

from src.erju.process_FO_base import BaseFOdata
from src.utils.file_utils import from_window_get_fo_file, compute_psd, bandpass, align_signals, \
    compute_cosine_similarity_windows, compute_psd_fixed, create_results_folder
from src.utils.plot_utils import plot_sig_acc_fo, plot_sig_fo_raw_and_processed, \
    plot_sig_psd_acc, plot_sig_psd_acc_fo, plot_sig_acc_raw_and_processed, \
    plot_sig_acc_fo_align, plot_cosine_sim_boxplot, plot_psd_summary, plot_fo_psd_ch_compare, plot_sig_fft_acc_fo, \
    plot_sig_fft_acc_fo


def main():
    """
    Main function to compare accelerometer data with FO data.
    """
    ##### CONFIGURATION #####
    # 1. Data paths
    path_stem_db = r"P:/11207352-stem/database/Wielrondheid_132887.db"
    path_fo_data = r"E:\recording_2024-08-26T12_59_54Z_5kHzping_1kHzlog_1mCS_2mGL_3000channels"
    # path_save_res = r"N:\Projects\11210000\11210064\B. Measurements and calculations\holten\2m GL"
    path_save_res = r"P:\11210064-erju\holten"

    # 2. Time window for analysis
    start_date = '2024-08-27 13:45:00'
    end_date = '2024-08-27 14:00:00'
    # start_date = '2024-08-27 15:00:00'
    # end_date = '2024-08-27 15:30:00'

    # 3. Parameters for querying the accelerometer database
    location_name = ['Meetjournal_MP8_Holten_zuid_4m_C']  # centre accelerometer
    # location_name = ['Meetjournal_MP7_Holten_zuid_4m_B']  # left accelerometer
    # location_name = ['Meetjournal_MP9_Holten_zuid_4m_D']  # right accelerometer
    campaigns = None
    traintype = "ICM"  # ICM
    track = "1"

    # 4. Parameters for the FO data
    first_channel = 800
    center_channel = 1194
    last_channel = 1400
    fo_reader = 'optasense'

    # 5. Parameter for the signal processing and filters

    # 6. Parameters for the PSD computation
    window_size = 1024

    # 7. Plot configuration
    interactive_plots = True  # Set to True if you want to save interactive plots
    PLOT_CONFIG = {
        "sig_acc_fo": True,
        "sig_fo_raw_and_processed": True,
        "sig_psd_acc": True,
        "sig_psd_acc_fo": True,
        "sig_fft_acc_fo": True,
        "sig_acc_raw_and_processed": False,
        "sig_acc_fo_align": True,
        "cosine_boxplot": True,
        "psd_summary": True,
    }

    # Create the dynamically named results folder
    results_folder = create_results_folder(base_path=path_save_res, start_date=start_date, end_date=end_date,
                                           traintype=traintype, center_channel=center_channel, track=track)

    logger.info(f"Results will be saved in: {results_folder}")
    # Start the timer
    start_time = time.time()
    logger.info("Starting the processing of events...")
    # Process all events
    process_all_events(path_db=path_stem_db, path_fo_data=path_fo_data, start_date=start_date,
                       end_date=end_date, location_name=location_name,
                       campaign=campaigns, traintype=traintype, track=track, window_size=window_size,
                       first_channel=first_channel, center_channel=center_channel, last_channel=last_channel,
                       reader=fo_reader)


def process_acc_event(record, window_size):
    # Unpack the accelerometer data from the extracted records
    absolute_time = record["absolute_time"]
    trace_x_raw = record["trace_x"]
    trace_y_raw = record["trace_y"]
    trace_z_raw = record["trace_z"]
    time_window = record["time_window"]

    # Estimate the sampling frequency of the accelerometer data
    freq_estimate = estimate_sampling_frequency(absolute_time)

    # Create the TimeSignalProcessing objects for accelerometer data
    trace_x = TimeSignalProcessing(time=absolute_time, signal=trace_x_raw, Fs=freq_estimate,
                                   window=Windows.HAMMING, window_size=window_size)
    trace_y = TimeSignalProcessing(time=absolute_time, signal=trace_y_raw, Fs=freq_estimate,
                                   window=Windows.HAMMING, window_size=window_size)
    trace_z = TimeSignalProcessing(time=absolute_time, signal=trace_z_raw, Fs=freq_estimate,
                                   window=Windows.HAMMING, window_size=window_size)
    # Apply the bandpass filter to the data
    for trace in [trace_x, trace_y, trace_z]:
        trace.filter(Fpass=[1, 100], N=4, type_filter='bandpass')

    return absolute_time, trace_x, trace_y, trace_z, time_window


def get_event_fo_data(path_fo_data, first_channel, last_channel, reader, time_window):
    # Create a BaseFOdata object for the FO data
    fo = BaseFOdata.create_instance(dir_path=path_fo_data, first_channel=first_channel, last_channel=last_channel,
                                    reader=reader)
    # find the files in the time window
    files_in_event = from_window_get_fo_file(path_fo_data, time_window)

    # Now we loop through the fo files one by one and extract the data
    # First lets create a container to store the fo data
    fo_data = []

    # Now the loop
    for fo_file in files_in_event:
        # Get properties from the first file
        if fo_file == files_in_event[0]:
            fo.extract_properties_per_file(fo_file)
            file_start_time = fo.properties['FileStartTime']
            sampling_frequency = int(fo.properties['SamplingFrequency[Hz]'])

        # Try using the extract_data from the BaseFOdata class
        # This function already has a bandpass filter implemented, as well as a
        # conversion form optical phase to strain
        _, signal_data = fo.extract_data(file_name=fo_file, first_channel=first_channel, last_channel=last_channel)

        # Append the fo data to the list
        fo_data.append(signal_data.T)

    # Concatenate the data from multiple files
    fo_data = np.concatenate(fo_data, axis=0)

    # Crop the FO data to the time window
    fo_data_croped = crop_fo_with_accel_time(fo_data, time_window, file_start_time, sampling_frequency)

    return fo_data_croped, sampling_frequency


def process_fo_event(path_fo_data, first_channel, center_channel, last_channel, reader, time_window, absolute_time,
                     window_size):
    # Get the FO data for the event
    fo_data, fo_fs = get_event_fo_data(path_fo_data, first_channel, last_channel, reader, time_window)

    # Create a TimeSignalProcessing object for the FO data
    fo = TimeSignalProcessing(time=absolute_time, signal=fo_data[:, center_channel - first_channel], Fs=fo_fs,
                              window=Windows.HAMMING, window_size=window_size)
    # Apply the bandpass filter to the FO data
    fo.filter(Fpass=[1, 100], N=4, type_filter='bandpass')

    return fo


def crop_fo_with_accel_time(fo_data, time_window, file_start_time, sampling_frequency):
    # Compute timestamps for FO data
    timestamps = [file_start_time + timedelta(seconds=i / sampling_frequency) for i in range(fo_data.shape[0])]
    # Convert timestamps to NumPy datetime64 for indexing
    timestamps_array = np.array(timestamps, dtype='datetime64[ns]')

    # Find closest start and end indices within the FO timestamps
    start_time_np = np.datetime64(time_window[0])
    end_time_np = np.datetime64(time_window[1])
    start_index = np.argmin(np.abs(timestamps_array - start_time_np))
    end_index = np.argmin(np.abs(timestamps_array - end_time_np))

    # Crop FO data to the time window
    timestamps = timestamps[start_index:end_index + 1]
    fo_data = fo_data[start_index:end_index + 1, :]

    return fo_data


def compute_PSD(trace_x, trace_y, trace_z, fo):
    trace_x.psd()
    trace_y.psd()
    trace_z.psd()
    fo.psd()

    return trace_x, trace_y, trace_z, fo


def process_all_events(path_db, path_fo_data, start_date, end_date, location_name, campaign, traintype, track,
                       window_size, first_channel, center_channel, last_channel, reader):
    # Get all the accelerometer data from the database
    all_events = unpack_accel_data(path_db=path_db, start_date=start_date, end_date=end_date, location=location_name,
                                   campaign=campaign, traintype=traintype, track=track, get_timeseries=True)

    # Loop through each event and process it
    for event in all_events:
        absolute_time, trace_x, trace_y, trace_z, time_window = process_acc_event(event, window_size=window_size)

        fo = process_fo_event(path_fo_data=path_fo_data, first_channel=first_channel, center_channel=center_channel,
                              last_channel=last_channel, reader=reader, time_window=time_window,
                              absolute_time=absolute_time, window_size=window_size)

        # Compute the PSD
        compute_PSD(trace_x, trace_y, trace_z, fo)

        plt.plot(absolute_time, trace_x.signal)
        plt.show()


if __name__ == "__main__":
    main()
