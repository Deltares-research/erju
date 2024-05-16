# Once the FO and the accelerometer signals match, we can extract
# the individual signals and save them for the to be input for the ML training...

class ExtractData:
    """
    This class will extract the data from the FO and the accelerometer signals
    in individual files. It will find an appropiate time window for a passing
    of a train, and extract the data from the FO and the accelerometer signals.

    """

    def __init__(self, fo_signal, acc_signal):
        """
        Initialize the class instance

        Args:
            fo_signal: The FO signal
            acc_signal: The accelerometer signal
        """


    # First, we use the accelerometer data to find the time window for the passing of the train.
    def find_events(self, location: str, thresholds: dict, distance: int, window_size: int):
        """
        Find the sections where an event is recorded in the signal. We do this by looking at the
        combined signal from all sensors at a given location. If the combined signal is above the
        threshold, we consider that an event. We use distance to ignore the signal for a given
        number of data points after the event. We use window_size to create a window around the
        event.

        Args:
            location (str): The location to analyze
            thresholds (float): The threshold for the combined signal
            distance (int): Minimum number of data points that should separate two events
            window_size (int): The number of samples before and after the event

        Returns:
            sensor_windows (dict): A dictionary containing the windows for each sensor at a given location
        """

        # Get the signal data for the chosen location
        data = self.grouped_df[location]

        # Get the threshold for this location
        threshold = thresholds.get(location, 0.02)  # Use a default value if the location is not in the dictionary

        # Overwrite the threshold for location 1, DELETE THIS if you want custom thresholds for each location
        threshold = thresholds.get('location_1')


        # Create an empty dictionary to store the windows for each sensor
        sensor_windows = {}

        # Combine the signal from all sensors at a given location
        # They are combined in order to get the exact same windows in all sensors, as some times
        # the signal is above the threshold in one sensor but not in the other...
        combined_signal = data.iloc[:, 1:].sum(axis=1)

        # Overwrite the combined signal for location 1, DELETE THIS if you want custom signals for each location
        combined_signal = self.grouped_df['location_1'].iloc[:, 1:].sum(axis=1)

        # Find the indices where the combined signal crosses the threshold
        cross_indices = np.where(np.diff(combined_signal > threshold))[0]

        # Split the indices into "events" where the signal is above the threshold
        # For this we look at the difference between the indices and split when the difference
        # is greater than the distance between values above the threshold
        events = np.split(cross_indices, np.where(np.diff(cross_indices) > distance)[0] + 1)

        # For each "event", find the start and end points
        for event in events:
            start_point = event[0]
            end_point = event[-1]

            # Create a window around the start and end points by adding the window_size
            # both before and after the start and end points. We also make sure that the
            # window does not go outside the signal data (that's why we use max and min)
            start_index = max(0, start_point - window_size)
            end_index = min(len(data) - 1, end_point + window_size)

            # Append the window to the list for each sensor
            for sensor in data.columns[1:]:
                sensor_windows.setdefault(sensor, []).append((start_index, end_index))

        return sensor_windows