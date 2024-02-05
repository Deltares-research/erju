# Purpose: Test run of the ReadTDMS class
import time

from tqdm import tqdm

from erju.find_trains_silixa import FindTrains
from erju.plot_data import PlotData

# Define the path to the TDMS file
dir_path = r'C:\Projects\erju\data'
dir_path = r'D:\FO_culemborg_20112020\subset'
# Define the first and last channel to be extracted
first_channel = 0
last_channel = 8000
threshold = 500

#########################################################################
# Start the timer
start_timer = time.time()

# Initialize the FindTrains class instance
file_cul = FindTrains(dir_path, first_channel, last_channel)
# Extract the properties of the TDMS file
properties = file_cul.extract_properties()

# Get the average signal
signal_mean = file_cul.signal_averaging(plot=False)

# Find the file names above the threshold
selected_files = file_cul.get_files_above_threshold(signal_mean, threshold=threshold)
# From the selected files, extract the data
all_data = file_cul.get_data(selected_files)

pbar = tqdm(total=len(selected_files))
# Plot the data for all the files inside the selected files
for file_name in selected_files:
    # Create the plotting instance
    train_22_cul_plots = PlotData(dir_path, file_name, all_data)
    # Plot the data
    train_22_cul_plots.plot_array_channels(save_figure=True)
    pbar.update(1)

pbar.close()
#################################################################################

# Plot a single channel
# Choose a channel index
file_index = 1
channel_index =3500

# Create the instance for a given file index
single_ch_plot = PlotData(dir_path, selected_files[file_index], all_data)

# plot an array of channels
single_ch_plot.plot_array_channels(save_figure=True)
# Plot the data
single_ch_plot.plot_single_channel(channel_index=channel_index, save_figure=True)


# Stop the timer
stop_timer = time.time()
print('Elapsed time: ', stop_timer - start_timer, 'seconds')