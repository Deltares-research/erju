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

#########################################################################
# Start the timer
start_time = time.time()

# Initialize the FindTrains class instance
train_22_cul = FindTrains(dir_path, first_channel, last_channel)
# Extract the properties of the TDMS file
properties = train_22_cul.extract_properties()

# Get the average signal
signal_mean = train_22_cul.signal_averaging(plot=True)

# Find the file names above the threshold
selected_files = train_22_cul.get_files_above_threshold(signal_mean, threshold=500)
# From the selected files, extract the data
all_data = train_22_cul.get_data(selected_files)

#################################################################################

# Plot the data
# Create the plotting instance
train_22_cul_plots = PlotData(dir_path, selected_files[0], all_data)
train_22_cul_plots.plot_data(save_figure=True)

# Stop the timer
stop_time = time.time()
print('Elapsed time: ', stop_time - start_time, 'seconds')
