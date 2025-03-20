import time

from src.utils.plot_FO_data import PlotData
from src.erju.process_FO_base import BaseFOdata

from src.utils.file_utils import get_files_in_dir

##### USER INPUT #######################################################################################################

# Define the path to the TDMS file
dir_path = r'C:\Projects\erju\data\holten\recording_2024-08-29T08_01_16Z_5kHzping_1kHzlog_1mCS_10mGL_3000channels'

# Define the path to save the figures
save_to_path = r'C:\Projects\erju\outputs\holten'

# Define the first and last channel to be extracted
first_channel = 0
last_channel = 3000  # to note, the maximum number of channels in the current iDAS files is 7808

# Define the threshold for the signal detection
threshold = 1.5E-8

# Choose the reader type between 'silixa' / 'nptdms' / 'optasense'
reader_type = 'optasense'

channel = 1500
########################################################################################################################

# Start the timer
start_timer = time.time()

# Initialize the FindTrains class instance
file_cul_instance = BaseFOdata.create_instance(dir_path, first_channel, last_channel, reader_type)

# Get a list with all the names of the TDMS files
if reader_type == 'optasense':
    file_format = '.h5'
else:
    file_format = '.tdms'

file_names = get_files_in_dir(folder_path=dir_path, file_format=file_format)
print('File names: ', file_names)

# Extract the properties of the TDMS file
if reader_type == 'optasense':
    properties = file_cul_instance.extract_properties_per_file(file_names[0])
else:
    properties = file_cul_instance.extract_properties()

print('Properties: ', properties)

# Get the average signal
signal_mean = file_cul_instance.signal_averaging(file_type=file_format, plot=True, save_to_path=save_to_path, threshold=threshold)

# Find the file names above the threshold
selected_files = file_cul_instance.get_files_above_threshold(file_type=file_format, signal=signal_mean, threshold=threshold)

print('Selected files: ', selected_files)

# Save the name of the files with trains in a txt
file_cul_instance.save_txt_with_file_names(save_to_path=save_to_path, selected_files=selected_files,
                                           file_names=file_names, include_indexes=True)

# From the selected files, extract the data
all_data = file_cul_instance.get_data_per_file(file_names)

# Stop the timer
stop_timer = time.time()
print('Elapsed time: ', stop_timer - start_timer, 'seconds')


########################################################################################################################
# PLOTTING THE DATA


# Plot the data file by file in the 30 seconds window
for file_name in selected_files:
    # Create the plotting instance
    train_22_cul_plots = PlotData(file_name, all_data)
    # Plot the data
    train_22_cul_plots.plot_array_channels(start_time=properties['FileStartTime'],
                                           end_time=properties['FileEndTime'],
                                           save_to_path=save_to_path,
                                           save_figure=True,
                                           guide_line=channel)

"""
data = file_cul_instance.get_data_with_window(selected_files[0],
                                              window_before=30,
                                              window_after=30,
                                              resample=True,
                                              new_sampling_frequency=100)


extended_plot = PlotData(selected_files[0], all_data)
extended_plot.plot_2d_buffer(save_to_path=save_to_path, save_figure=True, data=data)


# Plot files together to follow a given train
file_cul_instance.plot_array_channels(file_to_plot=selected_files[0],
                                      window_before=30,
                                      window_after=30,
                                      resample=True,
                                      new_sampling_frequency=50,
                                      save_to_path=save_to_path,
                                      save_figure=True)
"""

# Plot a single channel
file_index = 1          # File index to plot (from the selected_files list)
channel_index = channel    # Channel from the file to plot

# Create the instance for a given file index
single_ch_plot = PlotData(selected_files[file_index], all_data)


# Plot the data
single_ch_plot.plot_single_channel(channel_index=channel_index,
                                   start_time=properties['FileStartTime'],
                                   end_time=properties['FileEndTime'],
                                   save_to_path=save_to_path,
                                   save_figure=True)


"""

# plot an array of channels
single_ch_plot.plot_array_channels(start_time=properties['FileStartTime'],
                                   end_time=properties['FileEndTime'],
                                   save_to_path=save_to_path,
                                   save_figure=True)

"""