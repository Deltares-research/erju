# Purpose: Test run of the ReadTDMS class
from erju.read_tdms import ReadTDMS
from erju.scan_for_trains import ScanForTrains

# Define the path to the TDMS file
file_path = r'C:\Projects\erju\data\iDAS_continous_measurements_30s_UTC_20201121_101949.913.tdms'
#file_path = r'D:\FO_culemborg_22112020'
# Define the first and last channel to be extracted
first_channel = 100
last_channel = 200


# Initialize the ReadTDMS class instance
read_tdms = ReadTDMS(file_path, first_channel, last_channel)
# Get the properties of the TDMS file
properties = read_tdms.get_properties()
# Get the selected data from the TDMS file
data = read_tdms.get_data()

# Initialize the ScanForTrains class instance with the ReadTDMS instance
scan_for_trains = ScanForTrains(data, properties, first_channel, last_channel)

# Grab the search parameters
scan_channel, start_time, end_time = scan_for_trains.search_params()

print('scan_channel: ', scan_channel)
print('start_time: ', start_time)
print('end_time: ', end_time)

# Get the average signal
mean_signal = scan_for_trains.signal_averaging()
print('mean_signal: ', mean_signal)

# Plot the selected data
#read_tdms.plot_data(save_figure=True)