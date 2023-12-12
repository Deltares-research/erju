# Purpose: Test run of the ReadTDMS class
from erju.read_tdms import ReadTDMS
from erju.scan_for_trains import ScanForTrains
from erju.find_trains import FindTrains

# Define the path to the TDMS file
dir_path = r'C:\Projects\erju\data'
# Define the first and last channel to be extracted
first_channel = 0
last_channel = 500



##############################

trains = FindTrains(dir_path, first_channel, last_channel)
trains.extract_properties()
print(trains.search_params())
print(trains.signal_averaging())


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
