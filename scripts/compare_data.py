from datetime import datetime, timedelta

from DatabaseUtils import get_commands

from src.erju.process_FO_base import BaseFOdata
from src.utils.file_utils import get_files_list, from_window_get_fo_file


# What data we want to compare? Lets tackle it one by one:
# 1. There is the accelerometer data
# 2. There is the FO data


# 1. Accelerometer data ##########################################

# Function to fetch the data from the database based on some given conditions
def fetch_accel_data(db_path: str,
                     start_date: str,
                     end_date: str,
                     locations: list,
                     campaigns: list = None,
                     get_timeseries: bool = False,
                     traintype: str = None,
                     track: str = None):
    """
    Connects to SQLite STEM database and fetches the accelerometer data

    Args:
        db_path (str): Path to the SQLite database
        start_date (str): Start date in the format 'YYYY-MM-DD HH:MM:SS'
        end_date (str): End date in the format 'YYYY-MM-DD HH:MM:SS'
        locations (list): List of locations i.e. ['Meetjournal_MP8_Holten_zuid_4m_C', 'location2']
        campaigns (list): List of campaigns
        get_timeseries (bool): Whether to get the timeseries of the events, default is False
        traintype (str): Filter by traintype (optional)
        track (str): Filter by track (optional)

    Returns:
        events (list): List of events
        tim (dict): Dictionary of timeseries
        mis (list): List of missing files

    """
    # Connect to the database and fetch event data
    conn = get_commands.connect_to_db(db_path)
    # Fetch the events between the given dates and other conditions
    events, tim, mis = get_commands.get_events_between_dates(conn=conn,
                                                             start_date=start_date,
                                                             end_date=end_date,
                                                             locations=locations,
                                                             campaigns=campaigns,
                                                             get_timeseries=get_timeseries,
                                                             traintype=traintype,
                                                             track=track)
    # Close the connection
    get_commands.close_connection(conn)

    # Print the number of events fetched in a given time range
    print(f"Number of events fetched: {len(events)}")

    return events, tim, mis


# Function to extract the time series data from the fetched data
def unpack_timeseries(event: list, data: dict):
    """
    Unpacks accelerometer time series data and returns absolute time and XYZ traces.

    Args:
        event (list/tuple): Event record from the database (event[2] is the start time string)
        data (dict): Dictionary containing 'TIME', 'TRACE_X', 'TRACE_Y', 'TRACE_Z'

    Returns:
        start_time (datetime): Parsed start time of the event
        absolute_time (list of datetime): Absolute timestamps for each data point
        trace_x (np.array): Accelerometer trace in X
        trace_y (np.array): Accelerometer trace in Y
        trace_z (np.array): Accelerometer trace in Z
        time_window (list of datetime): [start, end] of the absolute time range
    """
    # Extract absolute start time of the event
    start_time_str = event[2]
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")

    # Extract time series traces
    relative_time = data["TIME"]
    trace_x = data["TRACE_X"]
    trace_y = data["TRACE_Y"]
    trace_z = data["TRACE_Z"]

    # Convert to absolute timestamps
    absolute_time = [start_time + timedelta(seconds=t) for t in relative_time]

    # Define time window
    time_window = [absolute_time[0], absolute_time[-1]]

    return absolute_time, trace_x, trace_y, trace_z, time_window


if __name__ == "__main__":
    # Define the paths
    path_db = r"P:/11207352-stem/database/Wielrondheid_132887.db"
    path_fo = r'E:\recording_2024-09-06T11_58_54Z_5kHzping_1kHzlog_1mCS_10mGL_6000channels'

    # Time range for extracting events
    start_date = '2024-09-07 00:00:00'
    end_date = '2024-09-08 00:00:00'
    # Parameters for querying the database
    locations = ['Meetjournal_MP8_Holten_zuid_4m_C']
    campaigns = None
    traintype = "SPR(A)"
    track = "2"
    # fo channels
    first_channel = 1190
    last_channel = 1200

    # Fetch the accelerometer data
    events, tim, mis = fetch_accel_data(db_path=path_db,
                                        start_date=start_date,
                                        end_date=end_date,
                                        locations=locations,
                                        campaigns=campaigns,
                                        traintype=traintype,
                                        track=track,
                                        get_timeseries=True)

    # Get the event time series dictionary (assuming one location)
    event_series = list(tim.values())[0]

    # Loop through each event and its corresponding time series
    for event, (event_id, data) in zip(events, event_series.items()):
        # Unpack the time series data
        absolute_time, trace_x, trace_y, trace_z, time_window = unpack_timeseries(event, data)

        # Plot absolute time in x vs accelerometer traceX in y
        # plt.figure()
        # plt.plot(absolute_time, trace_x)
        # plt.show()

        # 2 Lets look at the FO data ##########################################

        # Lets create an instance of the BaseFOdata class

        fo = BaseFOdata.create_instance(dir_path=path_fo,
                                        first_channel=first_channel,
                                        last_channel=last_channel,
                                        reader='optasense')
        # For each event, find the files in the time window
        fo_files_in_event = from_window_get_fo_file(path_fo, time_window)

        # fo.extract_properties_per_file(fo_files_in_event[0])
        # print(fo.properties['SamplingFrequency[Hz]'])

        # Now we loop through the fo files one by one and extract the data
        # First lets create a container to store the fo data
        fo_data = []
        # Now the loop
        for fo_file in fo_files_in_event:
            # Get properties from the first file
            if fo_file == fo_files_in_event[0]:
                fo.extract_properties_per_file(fo_file)
                print(fo.properties)
            # Try using the extract_data from the BaseFOdata class
            # This function already has a bandpass filter implemented, as well as a
            # conversion form optical phase to strain
            fo.extract_data(file_name=fo_file, first_channel=first_channel, last_channel=last_channel,
                            start_time=time_window[0], end_time=time_window[1])

            # Append the data to the list
            fo_data.append(fo.data)

            # Lets observe the data to check
            print(fo.data.shape)
