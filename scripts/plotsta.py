import matplotlib.pyplot as plt
from src.erju import AccelDataTimeWindows
from src.utils.file_utils import get_files_in_dir

# Define file paths and parameters
accel_data_path = r'D:\accel_trans\culemborg_data'
logbook_path = r'C:\Projects\erju\data\logbook_20201109_20201111.xlsx'
window_size_extension = 10  # seconds
event_separation_internal = 5  # seconds
threshold = 0.02
trigger_on = 1.5
trigger_off = 1

# Initialize AccelDataTimeWindows object with the specified parameters
time_windows = AccelDataTimeWindows(accel_data_path=accel_data_path,
                                    logbook_path=logbook_path,
                                    window_buffer=window_size_extension,
                                    event_separation_internal=event_separation_internal,
                                    threshold=threshold)

# Get a list of accelerometer files from the directory
file_names = get_files_in_dir(folder_path=accel_data_path, file_format='.asc', keep_extension=False)

# Step 1: Extract accelerometer data from the 14th file (index 13)
file_name = file_names[13]  # Choose the 14th file
accel_data_df = time_windows.extract_accel_data_from_file(file_name, no_cols=3)  # Extract data with 3 columns

# Step 2: Create event windows using a threshold method
windows_indices, windows_times = time_windows.create_windows_indices_and_times_threshold(accel_data_df)

# Step 3: Detect events using the STA/LTA method
nsta = int(1 * 1000)  # STA window (1 second, converted to milliseconds)
nlta = int(8 * 1000)  # LTA window (8 seconds, converted to milliseconds)
windows_indices_sta_lta, windows_times_sta_lta = time_windows.detect_accel_events_sta_lta(
    accel_data_df, nsta, nlta, trigger_on, trigger_off
)


# Step 5: Plot signal and event windows using the STA/LTA method
time_windows.plot_accel_signal_and_windows(accel_data_df, windows_indices_sta_lta,
                                           nsta, nlta, trigger_on=trigger_on, trigger_off=trigger_off)

# Display the plots
plt.show()



# Display the final plots
plt.show()
