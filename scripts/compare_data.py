import numpy as np
from loguru import logger
import time
import os
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from src.utils.db_utils import fetch_accel_data, unpack_timeseries, unpack_accel_data, estimate_sampling_frequency

from SignalProcessingTools.time_signal import TimeSignalProcessing, IntegrationRules, Windows, FilterDesign

from src.erju.process_FO_base import BaseFOdata
from src.utils.file_utils import from_window_get_fo_file, compute_psd, bandpass, align_signals, \
    compute_cosine_similarity_windows, compute_psd_fixed, create_results_folder
from src.utils.plot_utils import plot_sig_acc_fo, plot_sig_fo_raw_and_processed, \
    plot_sig_psd_acc, plot_sig_psd_acc_fo, plot_sig_acc_raw_and_processed, \
    plot_sig_acc_fo_align, plot_cosine_sim_boxplot, plot_psd_summary, plot_fo_psd_ch_compare, plot_sig_fft_acc_fo, \
    plot_sig_fft_acc_fo

if __name__ == "__main__":
    # Define the paths
    path_stem_db = r"P:/11207352-stem/database/Wielrondheid_132887.db"
    path_fo_data = r"E:\recording_2024-08-29T08_01_16Z_5kHzping_1kHzlog_1mCS_10mGL_3000channels"
    # path_fo_data = r"C:\fo_samples\holten"
    # path_plots = r"N:\Projects\11210000\11210064\B. Measurements and calculations\holten\2m GL"
    path_save_res = r"N:\Projects\11210000\11210064\B. Measurements and calculations\holten\10m GL\vel"

    # Time range for extracting events

    # 2 m GL
    # start_date = '2024-08-26 13:00:00'
    # end_date = '2024-08-29 07:59:00'

    # 10 m GL
    start_date = '2024-08-29 08:10:00'
    end_date = '2024-08-31 09:00:00'

    # Parameters for querying the database
    location_name = ['Meetjournal_MP8_Holten_zuid_4m_C']  # centre accelerometer
    # location_name = ['Meetjournal_MP7_Holten_zuid_4m_B']  # left accelerometer
    # location_name = ['Meetjournal_MP9_Holten_zuid_4m_D']  # right accelerometer

    campaigns = None
    traintype = "SPR(A)"  # ICM
    track = "1"

    # fo channels
    first_channel = 1184
    center_channel = 1194
    last_channel = 1204

    window_size = 1024  # Size of the window for the PSD calculation
    Fpass = [1, 100]

    interactive_plots = False  # Set to True if you want to save interactive plots
    PLOT_CONFIG = {
        "sig_acc_fo": True,
        "sig_fo_raw_and_processed": False,
        "sig_psd_acc": False,
        "sig_psd_acc_fo": True,
        "sig_fft_acc_fo": False,
        "sig_acc_fo_align": True,
        "psd_summary": True,
    }

    #################################################################
    # Create the dynamically named results folder
    results_folder = create_results_folder(base_path=path_save_res,
                                           start_date=start_date,
                                           end_date=end_date,
                                           traintype=traintype,
                                           center_channel=center_channel,
                                           track=track,
                                           Fpass=Fpass)

    # # Fetch the accelerometer data
    events, tim, mis = fetch_accel_data(db_path=path_stem_db,
                                        start_date=start_date,
                                        end_date=end_date,
                                        locations=location_name,
                                        campaigns=campaigns,
                                        traintype=traintype,
                                        track=track,
                                        get_timeseries=True)

    # Get the event time series dictionary (assuming one location)
    event_series = list(tim.values())[0]

    # Initialize storage for PSDs
    psd_x_all, psd_y_all, psd_z_all, psd_fo_all = [], [], [], []
    freqs_shared = None

    start_time = time.time()
    logger.info(f"Starting processing of {len(events)} events...")
    counter = 1
    total_events = len(events)

    # Loop through each event and its corresponding time series
    for event, (event_id, data) in zip(events, event_series.items()):
        # Create a counter to keep track of the event number
        logger.info(f"Processing event {counter}/{total_events} → Event ID: {event_id}")
        # Unpack the time series data
        absolute_time, trace_x_raw, trace_y_raw, trace_z_raw, time_window = unpack_timeseries(event, data)

        # The frequency of the accelerometer data is 1000 Hz.
        freq_accel = estimate_sampling_frequency(absolute_time)
        print(f"Estimated sampling frequency: {freq_accel} Hz")

        trace_x = TimeSignalProcessing(absolute_time, trace_x_raw, Fs=freq_accel, window=Windows.HAMMING,
                                       window_size=window_size)
        trace_y = TimeSignalProcessing(absolute_time, trace_y_raw, Fs=freq_accel, window=Windows.HAMMING,
                                       window_size=window_size)
        trace_z = TimeSignalProcessing(absolute_time, trace_z_raw, Fs=freq_accel, window=Windows.HAMMING,
                                       window_size=window_size)
        trace_x.filter(Fpass=Fpass, N=5, type_filter="bandpass", filter_design=FilterDesign.BUTTERWORTH)
        trace_y.filter(Fpass=Fpass, N=5, type_filter="bandpass", filter_design=FilterDesign.BUTTERWORTH)
        trace_z.filter(Fpass=Fpass, N=5, type_filter="bandpass", filter_design=FilterDesign.BUTTERWORTH)

        # 2 Lets look at the FO data ##########################################

        # Lets create an instance of the BaseFOdata class
        fo = BaseFOdata.create_instance(dir_path=path_fo_data,
                                        first_channel=first_channel,
                                        last_channel=last_channel,
                                        reader='optasense')

        # For each event, find the files in the time window
        fo_files_in_event = from_window_get_fo_file(path_fo_data, time_window)

        # Now we loop through the fo files one by one and extract the data
        # First lets create a container to store the fo data
        fo_data = []
        super_raw_data = []
        # Now the loop
        for fo_file in fo_files_in_event:
            # Get properties from the first file
            if fo_file == fo_files_in_event[0]:
                fo.extract_properties_per_file(fo_file)
                file_start_time = fo.properties['FileStartTime']
                sampling_frequency = int(fo.properties['SamplingFrequency[Hz]'])

            # Try using the extract_data from the BaseFOdata class
            # This function already has a bandpass filter implemented, as well as a
            # conversion form optical phase to strain
            processed_data, raw_signal_data = fo.extract_data(file_name=fo_file,
                                                              first_channel=first_channel,
                                                              last_channel=last_channel)

            # In the original code, the data is transposed, so we will un-transpose it
            # Append the data to the list
            fo_data.append(processed_data.T)
            super_raw_data.append(raw_signal_data.T)

        # Concatenate FO data from multiple files
        fo_data = np.concatenate(fo_data, axis=0)
        super_raw_data = np.concatenate(super_raw_data, axis=0)

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
        super_raw_data = super_raw_data[start_index:end_index + 1, :]

        # plot_fo_psd_ch_compare(
        #     event_id=event_id,
        #     timestamps=timestamps,
        #     super_raw_data=super_raw_data,
        #     sampling_frequency=sampling_frequency,
        #     center_channel=center_channel,
        #     first_channel=first_channel,
        #     last_channel=last_channel,
        #     window_size=window_size,
        #     save_dir=results_folder,
        #     step=50
        # )

        fibre_optics = TimeSignalProcessing(timestamps, super_raw_data[:, center_channel - first_channel],
                                            Fs=sampling_frequency, window=Windows.HAMMING, window_size=window_size)

        fibre_optics.filter(Fpass=Fpass, N=5, type_filter="bandpass", filter_design=FilterDesign.BUTTERWORTH)

        # Compute PSDs
        trace_x.psd(nb_points=10000)
        trace_y.psd(nb_points=10000)
        trace_z.psd(nb_points=10000)
        fibre_optics.psd(nb_points=10000)

        ch_index = center_channel - first_channel
        fo_trace = fo_data[:, ch_index]
        aligned_fo, lag = align_signals(trace_x.signal, fo_trace)

        # Save frequencies once
        if freqs_shared is None:
            freqs_shared = trace_x.frequency_Pxx

        psd_x_all.append(trace_x.Pxx)
        psd_y_all.append(trace_y.Pxx)
        psd_z_all.append(trace_z.Pxx)
        psd_fo_all.append(fibre_optics.Pxx)

        # Inside the event loop, after computing FO PSDs:
        max_psd_per_channel = []

        # Loop through each channel in the FO data
        for ch in range(last_channel - first_channel + 1):
            ch_trace = super_raw_data[:, ch]  # FO time series for this channel

            # Create TimeSignalProcessing object
            ch_tsp = TimeSignalProcessing(
                timestamps, ch_trace,
                Fs=sampling_frequency,
                window=Windows.HAMMING,
                window_size=window_size
            )
            ch_tsp.filter(Fpass=Fpass, N=5, type_filter="bandpass", filter_design=FilterDesign.BUTTERWORTH)
            ch_tsp.psd(nb_points=10000)

            # Take the maximum PSD value over all frequencies
            max_psd_per_channel.append(np.max(ch_tsp.Pxx))

        # Create channel position array
        channel_positions = np.arange(first_channel, last_channel + 1)

        # Plot scatter
        plt.figure(figsize=(10, 6))
        plt.scatter(channel_positions, max_psd_per_channel, color='dodgerblue')
        plt.xlabel("FO Channel")
        plt.ylabel("Max PSD value")
        plt.title(f"Max PSD per FO Channel - Event {event_id}")
        plt.grid(True)

        # Save plot to results folder
        plt.savefig(os.path.join(results_folder, f"max_psd_per_channel_event_{event_id}.png"), dpi=300)
        plt.close()

        # === SAVE TIME SERIES AND PSD TO CSV ===
        import pandas as pd
        import os

        if counter == 1:  # only on first loop, create folder
            csv_folder = os.path.join(results_folder, "csv_exports")
            os.makedirs(csv_folder, exist_ok=True)

        # Save synchronized time series
        min_len = len(aligned_fo)
        df_time_series = pd.DataFrame({
            "acc_time": absolute_time[:min_len],
            "acc_x": trace_x.signal[:min_len],
            "acc_y": trace_y.signal[:min_len],
            "acc_z": trace_z.signal[:min_len],
            "fo_time": timestamps[:min_len],
            "fo_signal": fibre_optics.signal[:min_len],
        })
        df_time_series.to_csv(os.path.join(csv_folder, f"time_series_event_{event_id}.csv"), index=False)

        # Save PSDs for this event
        df_psd = pd.DataFrame({
            "freq": freqs_shared,
            "psd_x": trace_x.Pxx,
            "psd_y": trace_y.Pxx,
            "psd_z": trace_z.Pxx,
            "psd_fo": fibre_optics.Pxx,
        })
        df_psd.to_csv(os.path.join(csv_folder, f"psd_event_{event_id}.csv"), index=False)

        traces = [trace_x.signal[:len(aligned_fo)], trace_y.signal[:len(aligned_fo)], trace_z.signal[:len(aligned_fo)],
                  fibre_optics.signal]

        counter += 1

        # Save the processed data to a file
        import pickle

        with open(os.path.join(results_folder, f"processed_data_event_{event_id}.pickle"), 'wb') as fo:
            pickle.dump({
                "PSD_x": [p.tolist() for p in psd_x_all],
                "PSD_y": [p.tolist() for p in psd_y_all],
                "PSD_z": [p.tolist() for p in psd_z_all],
                "PSD_fo": [p.tolist() for p in psd_fo_all],
                "trace_x": [trace_x.signal[:len(aligned_fo)].tolist()],
                "trace_y": [trace_y.signal[:len(aligned_fo)].tolist()],
                "trace_z": [trace_z.signal[:len(aligned_fo)].tolist()],
                "trace_fo": [p.tolist() for p in fibre_optics.signal[:min_len]],
                "freq": [p.tolist() for p in fibre_optics.frequency_Pxx],
                "time": timestamps,
            }, fo)

        # Plotting the results ########################################################

        # Plot the accelerometer vs FO data
        if PLOT_CONFIG["sig_acc_fo"]:
            plot_sig_acc_fo(save_dir=results_folder,
                            event_id=event_id,
                            accel_time=absolute_time,
                            trace_x=trace_x.signal[:len(aligned_fo)],
                            trace_y=trace_y.signal[:len(aligned_fo)],
                            trace_z=trace_z.signal[:len(aligned_fo)],
                            fo_time=timestamps,
                            fo_data=fo_data,
                            fo_channel=center_channel,
                            first_channel=first_channel,
                            save_interactive=interactive_plots)

        # Plot FO data before and after filtering
        if PLOT_CONFIG["sig_fo_raw_and_processed"]:
            plot_sig_fo_raw_and_processed(
                save_dir=results_folder,
                event_id=event_id,
                timestamps=timestamps,
                raw_signal_data=super_raw_data,
                processed_data=fo_data,
                fo_channel=center_channel,
                first_channel=first_channel,
                save_interactive=interactive_plots)

        # Accelerometer data and PSD's
        if PLOT_CONFIG["sig_psd_acc"]:
            plot_sig_psd_acc(event_id,
                             absolute_time,
                             trace_x,
                             trace_y,
                             trace_z,
                             fs=1000,
                             save_dir=results_folder,
                             freq_range=(0, 100),
                             fo_for_crop=aligned_fo)

        # Accelerometer data and fo with PSD's for 128/256/512
        if PLOT_CONFIG["sig_psd_acc_fo"]:
            plot_sig_psd_acc_fo(event_id=event_id,
                                save_dir=results_folder,
                                accel_time=absolute_time,
                                trace_x=trace_x.signal[:len(aligned_fo)],
                                trace_y=trace_y.signal[:len(aligned_fo)],
                                trace_z=trace_z.signal[:len(aligned_fo)],
                                fo_time=timestamps,
                                fo_trace=fo_data,
                                len_w=[128, 256, 512, 1024],
                                fo_channel=center_channel,
                                first_channel=first_channel,
                                fs_accel=1000,
                                fs_fo=sampling_frequency,
                                freq_range=(0, 100),
                                save_interactive=interactive_plots)

        if PLOT_CONFIG["sig_fft_acc_fo"]:
            plot_sig_fft_acc_fo(event_id=event_id,
                                save_dir=results_folder,
                                trace_x=trace_x,
                                trace_y=trace_y,
                                trace_z=trace_z,
                                accel_time=absolute_time,
                                fo_trace=fibre_optics,
                                fo_time=timestamps,
                                fo_channel=center_channel,
                                first_channel=first_channel,
                                fo_for_crop=aligned_fo)

        # IN ORDER TO CHECK ALLIGNMENT BETWEEN FO AND ACCELEROMETER DATA
        if PLOT_CONFIG["sig_acc_fo_align"]:
            plot_sig_acc_fo_align(event_id,
                                  timestamps,
                                  trace_x.signal[:len(aligned_fo)],
                                  trace_y.signal[:len(aligned_fo)],
                                  trace_z.signal[:len(aligned_fo)],
                                  aligned_fo,
                                  save_dir=results_folder, )

    # PLOT THE AGGREGATED PSD SUBPLOT
    if PLOT_CONFIG["psd_summary"]:
        plot_psd_summary(freqs_shared,
                         psds_x=psd_x_all,
                         psds_y=psd_y_all,
                         psds_z=psd_z_all,
                         psds_fo=psd_fo_all,
                         save_dir=results_folder)

    end_time = time.time()
    total_time = end_time - start_time

    logger.success(f"Finished processing all events in {total_time:.2f} seconds ({total_time / 60:.2f} minutes).")
