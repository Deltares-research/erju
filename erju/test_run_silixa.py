# Purpose: Test run of the ReadTDMS class
import time
import os
import glob

from utils.TDMS_Read import TdmsReader

from erju.find_trains_silixa import FindTrains
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

# Get a list of all TDMS files in the directory
dir_path = r'D:\FO_culemborg_22112020\subtest'
extension = '*.tdms'
all_files = glob.glob(os.path.join(dir_path, extension))

file_cul = FindTrains(dir_path, first_channel, last_channel)
# Extract the properties of the TDMS file
properties = file_cul.extract_properties()
print(properties)


# Stop the timer
stop_time = time.time()
print('Elapsed time: ', stop_time - start_time, 'seconds')

