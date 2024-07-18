import unittest
from erju.create_accel_windows import AccelDataTimeWindows
from utils.utils import get_files_in_dir, join_fo_channel_per_day
import pandas as pd
import matplotlib.pyplot as plt


class TestAccelAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accel_data_path = r'D:\accel_trans\culemborg_data'
        cls.logbook_path = r'C:\Projects\erju\data\logbook_20201109_20201111.xlsx'
        cls.window_size_extension = 10  # seconds
        cls.event_separation_internal = 5  # seconds
        cls.threshold = 0.02
        cls.trigger_on = 1.5
        cls.trigger_off = 1

        cls.time_windows = AccelDataTimeWindows(accel_data_path=cls.accel_data_path,
                                                logbook_path=cls.logbook_path,
                                                window_buffer=cls.window_size_extension,
                                                event_separation_internal=cls.event_separation_internal,
                                                threshold=cls.threshold)
        cls.file_names = get_files_in_dir(folder_path=cls.accel_data_path, file_format='.asc', keep_extension=False)

    def test_accel_analysis(self):
        accel_data_df = self.time_windows.extract_accel_data_from_file(self.file_names[19], no_cols=3)
        self.assertIsInstance(accel_data_df, pd.DataFrame)
        self.assertEqual(accel_data_df.shape[1], 3)

        windows_indices, windows_times = self.time_windows.create_windows_indices_and_times(accel_data_df)
        self.assertGreater(len(windows_indices), 0)
        self.assertGreater(len(windows_times), 0)

        nsta = int(1 * 1000)
        nlta = int(8 * 1000)
        windows_indices_sta_lta, windows_times_sta_lta = self.time_windows.detect_events_with_sta_lta(accel_data_df,
                                                                                                      nsta, nlta,
                                                                                                      self.trigger_on,
                                                                                                      self.trigger_off)
        self.assertGreater(len(windows_indices_sta_lta), 0)
        self.assertGreater(len(windows_times_sta_lta), 0)

        # Plot the signal and STA/LTA ratio with detected events using threshold method
        self.time_windows.plot_accel_signal_and_windows(accel_data_df, windows_indices)

        # Plot the signal and STA/LTA ratio with detected events using STA/LTA method
        self.time_windows.plot_accel_signal_and_windows(accel_data_df, windows_indices_sta_lta,
                                                        nsta, nlta, trigger_on=self.trigger_on,
                                                        trigger_off=self.trigger_off)

        # Display the plots
        plt.show()

        filtered_window_indices, filtered_window_times = self.time_windows.filter_windows_with_logbook(time_buffer=15,
                                                      window_indices=windows_indices_sta_lta,
                                                      window_times=windows_times_sta_lta)

        # Plot the signal and STA/LTA ratio with detected events using STA/LTA method
        self.time_windows.plot_accel_signal_and_windows(accel_data_df, filtered_window_indices,
                                                        nsta, nlta, trigger_on=self.trigger_on,
                                                        trigger_off=self.trigger_off)

        # FO data path
        self.fo_data_path = r'D:\RAIL4EARTH_PROJECT\DAS_DATA'
        # join the FO channel data per day for the given channel number
        fo_data = join_fo_channel_per_day(folder_path=self.fo_data_path, channel_no=4270)
        print(fo_data)


if __name__ == '__main__':
    unittest.main()
