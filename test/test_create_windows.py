import unittest
from erju.Accel.create_data_windows import AccelDataTimeWindows
from utils.utils import get_files_in_dir
import pandas as pd


class TestAccelAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up the test environment
        cls.accel_data_path = r'D:\accel_trans\culemborg_data'
        cls.window_size_extension = 10  # seconds
        cls.event_separation_internal = 5  # seconds
        cls.threshold = 0.02
        cls.trigger_on = 7
        cls.trigger_off = 1

        # Create an instance of AccelDataTimeWindows
        cls.time_windows = AccelDataTimeWindows(accel_data_path=cls.accel_data_path,
                                                window_buffer=cls.window_size_extension,
                                                event_separation_internal=cls.event_separation_internal,
                                                threshold=cls.threshold)
        # Get the list of file names in the directory
        cls.file_names = get_files_in_dir(folder_path=cls.accel_data_path, file_format='.asc', keep_extension=False)

    def test_accel_analysis(self):
        # Test the functionality of the accelerometer analysis

        # Load accelerometer data
        accel_data_df = self.time_windows.extract_accel_data_from_file(self.file_names[0], no_cols=3)

        # Check if the loaded data is a DataFrame with 3 columns
        self.assertIsInstance(accel_data_df, pd.DataFrame)
        self.assertEqual(accel_data_df.shape[1], 3)

        # Create window indices and times using the threshold method
        windows_indices, windows_times = self.time_windows.create_windows_indices_and_times(accel_data_df)

        # Check if windows are created
        self.assertGreater(len(windows_indices), 0)
        self.assertGreater(len(windows_times), 0)

        # Detect events using the STA/LTA method
        nsta = int(0.5 * 1000)  # 0.5 seconds window for STA
        nlta = int(5 * 1000)  # 5 seconds window for LTA
        windows_indices_sta_lta, windows_times_sta_lta = self.time_windows.detect_events_with_sta_lta(accel_data_df,
                                                                                                      nsta, nlta,
                                                                                                      self.trigger_on,
                                                                                                      self.trigger_off)

        # Check if events are detected
        self.assertGreater(len(windows_indices_sta_lta), 0)
        self.assertGreater(len(windows_times_sta_lta), 0)


if __name__ == '__main__':
    # Run the unittests
    unittest.main()
