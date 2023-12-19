# Purpose: Test run of the ReadTDMS class
import time

from erju.find_trains_silixa import FindTrains
from erju.plot_data import PlotData

# Define the path to the TDMS file
dir_path = r'C:\Projects\erju\data'
dir_path = r'D:\FO_culemborg_22112020\subtest'
# Define the first and last channel to be extracted
first_channel = 0
last_channel = 8000

#########################################################################
# Start the timer
start_time = time.time()

# Initialize the FindTrains class instance
file_cul = FindTrains(dir_path, first_channel, last_channel)
# Extract the properties of the TDMS file
properties = file_cul.extract_properties()

# Get the average signal
signal_mean = file_cul.signal_averaging(plot=True)

# Find the file names above the threshold
selected_files = file_cul.get_files_above_threshold(signal_mean, threshold=500)
print(selected_files)
# From the selected files, extract the data
all_data = file_cul.get_data(selected_files)

#################################################################################

# Plot the data
# Create the plotting instance
file_cul_plots = PlotData(dir_path, selected_files[0], all_data)
file_cul_plots.plot_array_channels(save_figure=True)
file_cul_plots.plot_single_channel(channel_index=3800, save_figure=True)


# Stop the timer
stop_time = time.time()
print('Elapsed time: ', stop_time - start_time, 'seconds')