from DatabaseUtils import get_commands
from datetime import datetime, timedelta
import numpy as np


def _safe_float_or_nan(value):
    """Convert value to float, returning NaN when missing/invalid."""
    if value is None:
        return np.nan
    if isinstance(value, str) and value.strip() == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


# Function to fetch the data from the database based on some given conditions
def fetch_accel_data(
    db_path: str,
    start_date: str,
    end_date: str,
    locations: list,
    campaigns: list = None,
    get_timeseries: bool = False,
    traintype: str = None,
    track: str = None,
):
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
    events, tim, mis = get_commands.get_events_between_dates(
        conn=conn,
        start_date=start_date,
        end_date=end_date,
        locations=locations,
        campaigns=campaigns,
        get_timeseries=get_timeseries,
        traintype=traintype,
        track=track,
    )
    # Close the connection
    get_commands.close_connection(conn)

    return events, tim, mis


def unpack_accel_data(
    path_db: str,
    start_date: str,
    end_date: str,
    location: str,
    campaign: str,
    traintype: str = None,
    track: str = None,
    get_timeseries: bool = True,
):
    """ """
    # Fetch the accelerometer data
    events, tim, mis = fetch_accel_data(
        db_path=path_db,
        start_date=start_date,
        end_date=end_date,
        locations=location,
        campaigns=campaign,
        traintype=traintype,
        track=track,
        get_timeseries=True,
    )

    # Get the event time series dictionary (assuming one location)
    event_series = list(tim.values())[0]

    records = []

    # Loop through each event and its corresponding time series
    for event, (event_id, data) in zip(events, event_series.items()):
        # Unpack the time series data
        absolute_time, trace_x, trace_y, trace_z, time_window = unpack_timeseries(
            event, data
        )

        record = {
            "event_id": event_id,
            "absolute_time": absolute_time,
            "trace_x": trace_x,
            "trace_y": trace_y,
            "trace_z": trace_z,
            "time_window": time_window,
        }
        records.append(record)

    return records


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


def fetch_multi_mp_accel_data(
    path_db: str,
    start_date: str,
    end_date: str,
    measurement_points: list,
    sensor_id_map: dict,
    campaigns: list = None,
    traintype: str = None,
    track: str = None,
):
    """
    Fetch accelerometer data from multiple measurement points and group by event.

    This function queries the database for each measurement point separately,
    then groups the results by event_id (same train passing creates same event_id
    across all measurement points).

    Args:
        path_db: Path to the SQLite database
        start_date: Start date in the format 'YYYY-MM-DD HH:MM:SS'
        end_date: End date in the format 'YYYY-MM-DD HH:MM:SS'
        measurement_points: List of measurement point names (e.g.,
                           ['Meetjournal_MP8_Holten_zuid_4m_C', ...])
        sensor_id_map: Dictionary mapping measurement point names to sensor IDs
                      (e.g., {'Meetjournal_MP8_Holten_zuid_4m_C': 'MP8', ...})
        campaigns: List of campaigns (optional)
        traintype: Filter by traintype (optional)
        track: Filter by track (optional)

    Returns:
        dict: Dictionary organized by event_id, with each event containing
              data from all measurement points:
              {
                  "event_id_1": {
                      "event_metadata": {...},  # Common event info
                      "measurement_points": {
                          "MP8": {time, x, y, z, fs_hz, ...},
                          "MP9": {time, x, y, z, fs_hz, ...},
                          ...
                      }
                  },
                  ...
              }
    """
    print(f"\nFetching data from {len(measurement_points)} measurement points...")

    # Dictionary to store data grouped by event_id
    events_by_id = {}

    # Fetch data for each measurement point
    for mp_name in measurement_points:
        print(f"  - {mp_name}...")

        # Fetch events and timeseries for this MP
        events, tim, mis = fetch_accel_data(
            db_path=path_db,
            start_date=start_date,
            end_date=end_date,
            locations=[mp_name],
            campaigns=campaigns,
            traintype=traintype,
            track=track,
            get_timeseries=True,
        )

        # Merge timeseries from ALL campaigns into a flat {event_id: data} dict.
        # When a location spans multiple campaigns (e.g. a sensor that was renamed
        # between measurement periods), get_events_between_dates returns one entry
        # per matching campaign table.  Taking only list(tim.values())[0] would
        # silently discard all events from every campaign beyond the first.
        event_series: dict = {}
        if tim:
            for campaign_data in tim.values():
                event_series.update(campaign_data)

        # Get sensor ID from mapping (no string parsing)
        if mp_name not in sensor_id_map:
            raise ValueError(
                f"Measurement point '{mp_name}' not found in sensor_id_map. "
                f"Available keys: {list(sensor_id_map.keys())}"
            )
        sensor_id = sensor_id_map[mp_name]

        # Process each event by looking up its timeseries by event_id (tijdsignaal,
        # column index 7).  Direct lookup avoids the positional zip truncation that
        # occurs when `events` contains rows from multiple campaigns but
        # `event_series` only held one campaign's data.
        for event in events:
            event_id = event[7]  # tijdsignaal column

            if event_id not in event_series:
                # Timeseries file was not found for this event; skip sensor.
                continue

            data = event_series[event_id]

            # Unpack timeseries
            absolute_time, trace_x, trace_y, trace_z, time_window = unpack_timeseries(
                event, data
            )

            # Estimate sampling frequency
            fs_hz = estimate_sampling_frequency(absolute_time)

            # If this is a new event, create entry with metadata
            if event_id not in events_by_id:
                events_by_id[event_id] = {
                    "event_metadata": {
                        "event_id": event_id,
                        "start_time": event[2],  # Event start time string
                        "traintype": event[3] if len(event) > 3 else None,
                        "track": event[4] if len(event) > 4 else None,
                        "speed": _safe_float_or_nan(
                            event[5] if len(event) > 5 else None
                        ),
                        "time_window": time_window,
                    },
                    "measurement_points": {},
                }

            # Add this MP's data to the event
            events_by_id[event_id]["measurement_points"][sensor_id] = {
                "mp_name": mp_name,
                "sensor_id": sensor_id,
                "absolute_time": absolute_time,
                "trace_x": np.array(trace_x),
                "trace_y": np.array(trace_y),
                "trace_z": np.array(trace_z),
                "fs_hz": fs_hz,
                "n_samples": len(absolute_time),
            }

    print(f"\nGrouped data into {len(events_by_id)} unique events")
    return events_by_id
