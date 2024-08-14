import time

from erju.process_FO_base import BaseFOdata
from utils.plot_FO_data import PlotData

from utils.utils import get_files_in_dir

##### USER INPUT #######################################################################################################
# Define the path to save the figures
save_to_path = r'C:\Projects\erju\test\test_output'

# Define the first and last channel to be extracted
first_channel = 0
last_channel = 5000  # to note, the maximum number of channels in the current iDAS files is 7808

# Choose the reader type between 'silixa' / 'nptdms' / 'optasense'
reader_type = 'silixa'

if reader_type == 'silixa':
    file_format = '.tdms'
    file_path = r'D:\FO_culemborg_20112020\subset'
elif reader_type == 'nptdms':
    file_format = '.tdms'
    file_path = r'D:\FO_culemborg_20112020\subset'
elif reader_type == 'optasense':
    file_format = '.h5'
    file_path = r'C:\Projects\erju\data\optasense'
else:
    print('The reader type is not valid. Please choose between "silixa", "nptdms" or "optasense".')


########################################################################################################################

# Start the timer
start_timer = time.time()

# Initialize the FindTrains class instance
FOdata = BaseFOdata.create_instance(file_path, first_channel, last_channel, reader_type)

# Get a list with all the names of the TDMS files
file_names = get_files_in_dir(folder_path=file_path, file_format=file_format)
print('File names: ', file_names)

# Extract the properties of FO file
properties = FOdata.extract_properties_per_file(file_names[0])
print('Properties: ', properties)

# Get the data
data = FOdata.get_data_per_file(file_names)

# Stop the timer
stop_timer = time.time()
print('Elapsed time: ', stop_timer - start_timer, 'seconds')


########################################################################################################################
"""
# Plot the data file by file in the 30 seconds window
for file in file_names:
    # Create the plotting instance
    train_22_cul_plots = PlotData(file_name=file, all_data=data)
    # Plot the data
    train_22_cul_plots.plot_array_channels(start_time=properties['FileStartTime'],
                                           end_time=properties['FileEndTime'],
                                           save_to_path=save_to_path,
                                           save_figure=True)
"""

# Plot a single channel
file_index = 15          # File index to plot
channel_index = 2300    # Channel from the file to plot


# Create the instance for a given file index
single_ch_plot = PlotData(file_name=file_names[file_index], all_data=data)

# Plot the data
single_ch_plot.plot_single_channel(channel_index=channel_index,
                                   start_time=properties['FileStartTime'],
                                   end_time=properties['FileEndTime'],
                                   save_to_path=save_to_path,
                                   save_figure=True)