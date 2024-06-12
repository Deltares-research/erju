from erju.Accel.create_data_windows import AccelDataTimeWindows
from utils.utils import get_files_in_dir


# Define the path to the folder containing the accelerometer data
accel_data_path = r'D:\accel_trans\culemborg_data'
window_size_extension = 10  # seconds
event_separation_internal = 5  # seconds
threshold = 0.02
trigger_on = 7
trigger_off = 1

# Create an instance of the AccelDataTimeWindows class
time_windows = AccelDataTimeWindows(accel_data_path=accel_data_path,
                                    window_buffer=window_size_extension,
                                    event_separation_internal=event_separation_internal,
                                    threshold=threshold)

# Get the list of file names in the folder
file_names = get_files_in_dir(folder_path=accel_data_path, file_format='.asc', keep_extension=False)

# Create a dataframe with the data from the first location, specifying the number of columns
# (in this case 3, because we use the first 3 columns of data from the file) and get the data
# from the first file in the list
accel_data_df = time_windows.extract_accel_data_from_file(file_names[0], no_cols=3)

# Find the indices and times where the combined signal crosses the threshold
windows_indices, windows_times = time_windows.create_windows_indices_and_times(accel_data_df)

# Detect events using STA/LTA method
nsta = int(0.5 * 1000)  # 0.5 seconds window for STA
nlta = int(5 * 1000)  # 10 seconds window for LTA
windows_indices_sta_lta, windows_times_sta_lta = time_windows.detect_events_with_sta_lta(accel_data_df,
                                                                                         nsta,
                                                                                         nlta,
                                                                                         trigger_on=trigger_on,
                                                                                         trigger_off=trigger_off)
# Print the results with both methods
print('For the old method:')
print('number of windows:', len(windows_indices))
print('windows indices:', windows_indices)
print('windows times:', windows_times)
print('For the STA/LTA method:')
print('number of windows:', len(windows_indices_sta_lta))
print('windows indices:', windows_indices_sta_lta)
print('windows times:', windows_times_sta_lta)


# Plot the signal and STA/LTA ratio with detected events using threshold method
time_windows.plot_accel_signal_and_windows(accel_data_df, windows_indices)

# Plot the signal and STA/LTA ratio with detected events using STA/LTA method
time_windows.plot_accel_signal_and_windows(accel_data_df, windows_indices_sta_lta,
                                           nsta, nlta, trigger_on=trigger_on, trigger_off=trigger_off)

