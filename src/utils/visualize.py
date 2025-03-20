# Simple code to open a FOAS file and plot the data:
import time
from src.erju.process_FO_base import BaseFOdata  # Import from erju/process_FO_base.py
from file_utils import get_files_in_dir  # Import from utils/file_utils.py
import matplotlib.pyplot as plt




# User inputs:
# Define the path to the file
file_path = r'C:\fo_samples\holten_simple'
#file_path = r'C:\Projects\erju\data\optasense'
#file_path = r'C:\Projects\erju\data\silixa'
# Define the path to save the figures
save_to_path = r'N:\Projects\11210000\11210064\B. Measurements and calculations\holten\visualize'

# Define the first and last channel to be extracted
first_channel = 0
last_channel = 3000

# Define the reader type
reader_type = 'optasense'

########################################################################################################################
if reader_type == 'silixa':
    file_format = '.tdms'
elif reader_type == 'nptdms':
    file_format = '.tdms'
elif reader_type == 'optasense':
    file_format = '.h5'
else:
    print('The reader type is not valid. Please choose between "silixa", "nptdms" or "optasense".')


# Start the timer
start_timer = time.time()

# Initialize the FindTrains class instance
FOdata = BaseFOdata.create_instance(file_path, first_channel, last_channel, reader_type)
# Get a list with all the names of the TDMS files
file_names = get_files_in_dir(folder_path=file_path, file_format=file_format)
print('File names: ', file_names)

# Extract the properties of the TDMS file
properties = FOdata.extract_properties_per_file(file_names[0])
print('Properties: ', properties)

# Get the data
data = FOdata.get_data_per_file(file_names)

# Plot with imshow the first element in the data dictionary
fig, ax = plt.subplots()
im = ax.imshow(data[file_names[0]], aspect='auto')
plt.colorbar(im)
plt.show()

# Stop the timer
stop_timer = time.time()
print('Elapsed time: ', stop_timer - start_timer, 'seconds')

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