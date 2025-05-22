import numpy as np
from loguru import logger
import time

from datetime import datetime, timedelta
from DatabaseUtils import get_commands

from SignalProcessingTools.time_signal import TimeSignalProcessing, IntegrationRules, Windows

from src.erju.process_FO_base import BaseFOdata
from src.utils.file_utils import from_window_get_fo_file, compute_psd, bandpass, align_signals, \
    compute_cosine_similarity_windows, compute_psd_fixed, create_results_folder
from src.utils.plot_utils import plot_sig_acc_fo, plot_sig_fo_raw_and_processed, \
    plot_sig_psd_acc, plot_sig_psd_acc_fo, plot_sig_acc_raw_and_processed, \
    plot_sig_acc_fo_align, plot_cosine_sim_boxplot, plot_psd_summary, plot_fo_window_and_psd_grid

from src.utils.db_utils import fetch_accel_data, unpack_timeseries, estimate_sampling_frequency

# Define the paths ###########################################################################################
path_db = r"P:/11207352-stem/database/Wielrondheid_132887.db"
path_fo = r'E:\recording_2024-09-06T11_58_54Z_5kHzping_1kHzlog_1mCS_10mGL_6000channels'
path_plots = r"N:\Projects\11210000\11210064\B. Measurements and calculations\holten"

# Define the time and location of the data to be processed ####################################################
start_date = '2024-09-08 00:00:00'
end_date = '2024-09-09 00:00:00'
# Parameters for querying the database
locations = ['Meetjournal_MP8_Holten_zuid_4m_C']  # centre accelerometer
# locations = ['Meetjournal_MP7_Holten_zuid_4m_B']  # left accelerometer
# locations = ['Meetjournal_MP9_Holten_zuid_4m_D']  # right accelerometer
campaigns = None
traintype = "SPR(A)"  # ICM
track = "1"

# Define the Fibre Optics channels ############################################################################
first_channel = 1094
center_channel = 1194
last_channel = 1294

# Define other parameters ######################################################################################
window_size = 1024  # Size of the window for the PSD calculation

if __name__ == "__main__":
    # Create the dynamically named results folder
    results_folder = create_results_folder(base_path=path_plots,
                                           start_date=start_date,
                                           end_date=end_date,
                                           traintype=traintype,
                                           center_channel=center_channel,
                                           track=track)

    # Fetch the accelerometer data
    events, tim, mis = fetch_accel_data(db_path=path_db,
                                        start_date=start_date,
                                        end_date=end_date,
                                        locations=locations,
                                        campaigns=campaigns,
                                        traintype=traintype,
                                        track=track,
                                        get_timeseries=True)

    # Get the event time series dictionary
    event_series = list(tim.values())[0]

    # Initialize storage for similarities
    similarity_scores_x, similarity_scores_y, similarity_scores_z = [], [], []
    # Initialize storage for the PSDs
    psd_x, psd_y, psd_z = [], [], []

    start_time = time.time()
    logger.info(f"Starting processing of {len(events)} events...")
    counter = 1
    total_events = len(events)

    # Loop through each event and its corresponding time series
    for event, (event_id, data) in zip(events, event_series.items()):
        # Create a counter to keep track of the event number
        logger.info(f"Processing event {counter}/{total_events} → Event ID: {event_id}")

        # Unpack the accelerometers time series data
        absolute_time, trace_x_raw, trace_y_raw, trace_z_raw, time_window = unpack_timeseries(event, data)
        # The frequency of the accelerometer data is 1000 Hz.
        freq_accel = estimate_sampling_frequency(absolute_time)

        # For the event, create an instance of the FOdata class and extract the data
        fo = BaseFOdata(path_fo=path_fo, first_channel=first_channel,
                        last_channel=last_channel, center_channel=center_channel)

        # Extract the fo data that belongs to the event
        fo_files_in_event = from_window_get_fo_file(fo=fo, time_window=time_window)
