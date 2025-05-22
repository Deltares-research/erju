from DatabaseUtils import get_commands


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


def estimate_sampling_frequency(time_vector):
    """
    Estimate the sampling frequency [Hz] from a list of datetime timestamps.

    Args:
        time_vector (list of datetime): List of datetime objects representing time axis.

    Returns:
        float: Estimated sampling frequency in Hz.
    """
    if len(time_vector) < 2:
        raise ValueError("Need at least two timestamps to compute sampling frequency.")

    # Calculate time differences in seconds
    time_deltas = np.diff([t.timestamp() for t in time_vector])
    avg_delta = np.mean(time_deltas)

    return 1.0 / avg_delta
