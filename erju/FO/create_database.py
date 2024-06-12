import pandas as pd

from erju.Accel.create_data_windows import AccelDataTimeWindows
from utils.utils import get_files_in_dir

class CreateDatabase:
    """
    A class used to create a database of FO and accelerometer data.
    It will create a database with at least 4 elements inside: the first element is
    the start time of the event, the second element is the end time of the event, the third
    element is the FO signal and the fourth, fith and so on, will be all the accelerometer
    signals, with as many elements as sensors we have.
    """

    def __init__(self, fo_data_path: str, acc_data_path: str):
        """
        Initialize the class instance to perform the database creation.
        Key elements are the paths to the FO and accelerometer data files and
        using the AccelDataTimeWindows and BaseFindTrains classes for the operations.

        Args:
            fo_data_path (str): The path to the folder containing the FO data files.
            acc_data_path (str): The path to the folder containing the accelerometer data files.
        """
        self.fo_data_path = fo_data_path
        self.acc_data_path = acc_data_path
        self.window_indices = None
        self.window_times = None

    def from_accel_get_windows(self, window_buffer: int = 10, threshold: float = 0.02, nsta: float = 0.5,
                               nlta: float = 5.0, trigger_on: float = 7, trigger_off: float = 1):
        """
        Get the time windows from the accelerometer data using the STA/LTA method.

        Args:
            window_buffer (int): The buffer to add to the time window.
            threshold (float): The threshold to use for the time window.
            nsta (float): The length of the average short time average window.
            nlta (float): The length of the average long time average window.
            trigger_on (float): The threshold to trigger the event on.
            trigger_off (float): The threshold to trigger the event off.

        Returns:
            accel_data_df (pd.DataFrame): The dataframe with the accelerometer data.
        """
        # Create an instance of the AccelDataTimeWindows class
        accel_windows = AccelDataTimeWindows(accel_data_path=self.acc_data_path,
                                             window_buffer=window_buffer,
                                             threshold=threshold)

        # Get the list of files names in the folder
        file_names = get_files_in_dir(folder_path=self.acc_data_path, file_format='.asc', keep_extension=False)

        # Create a dataframe with the data from the first location, specifying the number of columns
        # (in this case 3, because we use the first 3 columns of data from the file) and get the data
        # from the first file in the list
        accel_data_df = accel_windows.extract_accel_data_from_file(file_name=file_names[0], no_cols=3)

        # Detect the events using the STA/LTA method
        nsta = int(nsta * 1000) # convert to seconds with a fz of 1000 Hz
        nlta = int(nlta * 1000) # convert to seconds with a fz of 1000 Hz
        self.window_indices, self.window_times = accel_windows.detect_events_with_sta_lta(accel_data=accel_data_df,
                                                                                nsta=nsta,
                                                                                nlta=nlta,
                                                                                trigger_on=trigger_on,
                                                                                trigger_off=trigger_off)

        # Return
        return self.window_indices, self.window_times

    def from_windows_get_database(self):
        """
        Using the time windows from the accelerometer data, read the accelerometer and FO data and
        extract the data for the time windows. Put all the extracted data in a dataframe.
        """

        # Create an empty dataframe to store the data
        database = pd.DataFrame()

        # Create 2 columns called 'start_time' and 'end_time' with the window times.
        # The window_times is a list of tuples with the start and end times of the events.
        database['start_time'] = [t[0] for t in self.window_times]
        database['end_time'] = [t[1] for t in self.window_times]

        return database


##### TEST THE CODE ###################################################################################################

# Create an instance of the CreateDatabase class
db = CreateDatabase(fo_data_path=r'D:\csv_files_das\Culemborg_09112020',
                    acc_data_path=r'D:\accel_trans\culemborg_data')

# Get the time windows from the accelerometer data
window_indices, window_times = db.from_accel_get_windows(window_buffer=10, threshold=0.02, nsta=0.5,
                                                         nlta=5.0, trigger_on=7, trigger_off=1)

# Get the database from the time windows
database = db.from_windows_get_database()

# Print the database
print(database.head())





