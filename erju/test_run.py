# Purpose: Test run of the ReadTDMS class
import time

from erju.find_trains import FindTrains
from erju.plot_data import PlotData

# Define the path to the TDMS file
dir_path = r'C:\Projects\erju\data'
dir_path = r'D:\FO_culemborg_22112020\subtest'
# Define the first and last channel to be extracted
first_channel = 0
last_channel = 500


# Start the timer
start_time = time.time()

# Initialize the FindTrains class instance
train_22_cul = FindTrains(dir_path, first_channel, last_channel)
# Extract the properties of the TDMS file
properties = train_22_cul.extract_properties()

# Get the average signal
signal_mean = train_22_cul.signal_averaging(plot=False)

# Stop the timer
stop_time = time.time()
print('Elapsed time: ', stop_time - start_time, 'seconds')

# Find the file names above the threshold
selected_files = train_22_cul.get_files_above_threshold(signal_mean, threshold=500)
# From the selected files, extract the data
all_data = train_22_cul.get_data(selected_files)


# Plot the data
# Create the plotting instance
train_22_cul_plots = PlotData(dir_path, selected_files[0], all_data)
train_22_cul_plots.plot_data(save_figure=True)



##################
# OLD TEST

#
# # Initialize the ReadTDMS class instance
# read_tdms = ReadTDMS(file_path, first_channel, last_channel)
# # Get the properties of the TDMS file
# properties = read_tdms.get_properties()
# # Get the selected data from the TDMS file
# data = read_tdms.get_data()
#
# # Initialize the ScanForTrains class instance with the ReadTDMS instance
# scan_for_trains = ScanForTrains(data, properties, first_channel, last_channel)
#
# # Grab the search parameters
# scan_channel, start_time, end_time = scan_for_trains.search_params()
#
# print('scan_channel: ', scan_channel)
# print('start_time: ', start_time)
# print('end_time: ', end_time)
#
# # Get the average signal
# mean_signal = scan_for_trains.signal_averaging()
# print('mean_signal: ', mean_signal)
#
# # Plot the selected data
# #read_tdms.plot_data(save_figure=True)
