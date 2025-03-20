import os
from datetime import datetime, timedelta
from src.erju.process_FO_base import BaseFOdata
from DatabaseUtils import get_commands
from src.utils.file_utils import from_window_get_fo_file

# ---------------------- CONFIGURATION ----------------------
# Paths
path_database = r"P:/11207352-stem/database/Wielrondheid_132887.db"
path_fo_files = r'E:\recording_2024-09-06T11_58_54Z_5kHzping_1kHzlog_1mCS_10mGL_6000channels'
save_dir = r"N:/Projects/11210000/11210064/B. Measurements and calculations/holten/fo_vs_accel/"

# Ensure save directory exists
os.makedirs(save_dir, exist_ok=True)

# Time range for extracting events
start_date = '2024-09-07 00:00:00'
end_date = '2024-09-08 00:00:00'

# Parameters for querying the database
locations = ['Meetjournal_MP8_Holten_zuid_4m_C']
campaigns = None
traintype = "SPR(A)"
track = "2"

# Sensor channel configuration
center_channel = 1194
channel_range = 5
start_channel = max(0, center_channel - channel_range)  # Ensure start channel is not negative
end_channel = center_channel + channel_range
relative_center_channel = channel_range * 2 // 2  # Calculate the center in the new channel selection


# ---------------------- GET ACCELEROMETER DATA ----------------------

# Connect to the database and fetch event data for the specified time range
conn = get_commands.connect_to_db(path_database)
events, tim, mis = get_commands.get_events_between_dates(
    conn, start_date, end_date, locations, campaigns, get_timeseries=True, traintype=traintype, track=track
)
get_commands.close_connection(conn)

# Loop through each event and its associated accelerometer time series
for event_index, (event, (inner_key, data)) in enumerate(zip(events, list(tim.values())[0].items()), start=1):

    # Extract absolute event start time from database
    start_time_str = event[2]  # Assuming event[2] is a string
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")

    # Extract accelerometer time series data
    relative_time = data["TIME"]
    trace_x = data["TRACE_X"]
    trace_y = data["TRACE_Y"]
    trace_z = data["TRACE_Z"]

    # Convert relative times to absolute times
    absolute_time = [start_time + timedelta(seconds=t) for t in relative_time]

    # Define the time window for FO data extraction
    # THIS IS THE TIME WINDOW WE WILL USE TO EXTRACT THE FO DATA
    time_window = [absolute_time[0], absolute_time[-1]]

    # ---------------------- GET FO DATA ----------------------
    # Find the relevant FO files that match the time window; including a buffer of 1 file before and after
    fo_files_in_event = from_window_get_fo_file(path_fo_files, time_window)

    # Create a process_fo instance
    fo = BaseFOdata.create_instance()

