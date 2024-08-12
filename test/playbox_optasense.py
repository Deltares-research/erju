import h5py
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp




def convert_microseconds_to_datetime(microseconds):
    """
    Convert a timestamp in microseconds since epoch to a UTC datetime.

    Parameters:
    - microseconds (int): Timestamp in microseconds.

    Returns:
    - datetime: Corresponding UTC datetime.
    """
    # Convert microseconds to seconds
    seconds = microseconds * 1e-6
    # Return the corresponding datetime
    return datetime.utcfromtimestamp(seconds)


# Path to the .h5 file
optasense_path = r'C:\Projects\erju\data\optasense\Short_GL_Data_2022-07-06T054723Z 1.h5'

# Open the file
with h5py.File(optasense_path, 'r') as file:
    # Print the keys to see what datasets and groups are available
    print("Keys:", list(file.keys()))

with h5py.File(optasense_path, 'r') as file:
    # List all top-level groups and datasets
    print("All levels are as follows.........:")
    def printname(name):
        print(name)
    file.visit(printname)

print("...................................................................................")

# Open the file for reading
fp = h5py.File(optasense_path,'r')

# Print the keys for the top-level object in the file
print("Keys:", list(fp.keys()))

# Print the attributes for "Acquisition"
print("Attributes inside acquisition: ", list(fp['Acquisition'].attrs.keys()))

# Print the values associated with some important attributes
print('\nThe gauge length is ',fp['Acquisition'].attrs['GaugeLength'],'m.')
print('The pulse rate is ',fp['Acquisition'].attrs['PulseRate'],'Hz.')
print('The spatial sampling interval is ',fp['Acquisition'].attrs['SpatialSamplingInterval'],'m.')

# Inside the "Acquisition" group, there are 2 subgroups: "Custom" and "Raw[0]"
# Lets explore the subgroups
print("Inside the Custom group: ", list(fp['Acquisition']['Custom'].attrs.keys()))
print("Inside the Raw[0] group: ", list(fp['Acquisition']['Raw[0]'].attrs.keys()))
print("Inside the Raw[0]/Custom group: ", list(fp['Acquisition']['Raw[0]']['Custom'].attrs.keys()))
print("Inside the Raw[0]/RawData group: ", list(fp['Acquisition']['Raw[0]']['RawData'].attrs.keys()))
print("Inside the Raw[0]/RawDataTime group: ", list(fp['Acquisition']['Raw[0]']['RawDataTime'].attrs.keys()))

# Custom does not have more subgroups, but Raw[0] has sevel sub-subgroups.
# "Custom" (not the same as above), "RawDataTime", and "RawData".
# "Custom" is generally empty.
# "RawDataTime", is a dataset containing the UTC timestamp for each sample in the file in units of microseconds.
# "RawData", is the actual data.

# Create a new variable for the "RawDataTime" h5py dataset so we don't have to type so much
rawDataTime = fp['Acquisition']['Raw[0]']['RawDataTime']

# Get the first entry in "RawDataTime"
print('The first value in \"RawDataTime\" is',rawDataTime[0],\
      'which is the timestamp of the first sample in microseconds.')

# Convert from microseconds to a datetime object
file_start_time = convert_microseconds_to_datetime(rawDataTime[0])
print('The first timestamp in the file is',file_start_time)
# Get also the last time
file_end_time = convert_microseconds_to_datetime(rawDataTime[-1])
print('The last timestamp in the file is',file_end_time)
# The total duration of the recording is the difference between the first and last timestamps
duration = file_end_time - file_start_time
print('The total duration of the recording is',duration)
# In seconds
duration_seconds = duration.total_seconds()
print('The total duration of the recording is',duration_seconds,'seconds')

# Let's now look at the ACTUAL SIGNAL DATA.................................
# This is inside fp['Acquisition']['Raw[0]']['RawData']
# It is a 2D dataset (channel/distance vs. time)
# Create a new variable for the "RawData" h5py dataset
rawData = fp['Acquisition']['Raw[0]']['RawData']
# Explore the shape of the data
print('The shape of the data is',rawData.shape)
# This order is determined by a recording software user switch allowing to store the data in trace-order or channel-order.
# Generally the first axis is distance, and the second axis is time, for fast time-domin processing, and is normally constant throughout the reocrding

# Check what attributes are inside the dataset
print('The attributes inside the dataset are',list(rawData.attrs.keys()))
# Check the Dimensions, PartEndTime, PartStartTime, and StartIndex attributes
print('The dimensions are',rawData.attrs['Dimensions'])
#print('The PartEndTime is',rawData.attrs['PartEndTime'])
#print('The PartStartTime is',rawData.attrs['PartStartTime'])
#print('The StartIndex is',rawData.attrs['StartIndex'])

# Extract a single time slice (assume time is the first dimension)
trace = rawData[:, 2000]  # Assuming you want to inspect data for location index 5000

# Calculate the time interval and sampling rate
dt = (rawDataTime[1] - rawDataTime[0]) * 1e-6  # Convert from microseconds to seconds
fs = 1.0 / dt  # Sampling frequency
print('The CALCULATED time interval between samples is', dt, 's')
print('Sampling rate derived from timestamps:', fs, 'Hz')

# Generate the time axis based on the time dimension
time = np.arange(len(trace)) * dt
print('Length of time array:', len(time))

# Plot the data
plt.figure(figsize=(10, 4))
plt.plot(time,trace,'k')
plt.xlabel('Time (s)')
plt.ylabel('Raw Amplitude')
plt.show()
plt.close()

# DO NOTE, THE AMPLITUDE VALUES ARE IN RAW UNITS, AND NEED TO BE CONVERTED TO PHYSICAL (STRAIN) UNITS USING THE CALIBRATION FACTOR
# Check the formulation from https://github.com/ethanfwilliams/OOI_RCA_DAS_notebook/blob/main/OOI_RCA_OptaSense_DAS_intro.ipynb

# Print the units
print('The raw data unit is',fp['Acquisition']['Raw[0]'].attrs['RawDataUnit'])

# Remove the mean from the trace
trace = trace.astype(float)
trace -= np.mean(trace)

# Convert to units of radians
trace *= (2*np.pi)/2**16

# Convert to units of strain
n = fp['Acquisition']['Custom'].attrs['Fibre Refractive Index']
L = fp['Acquisition'].attrs['GaugeLength']
trace *= (1550.12 * 1e-9)/(0.78 * 4 * np.pi * n * L)

# Now plot again in strain units
plt.figure(figsize=(8,3))
plt.plot(time,trace,'k')
plt.xlabel('Time (s)')
plt.ylabel('Strain Amplitude')
plt.show()
plt.close()



# Assemble the metadata we will need
fs = fp['Acquisition']['Raw[0]'].attrs['OutputDataRate'] # sampling rate in Hz
dx = fp['Acquisition'].attrs['SpatialSamplingInterval'] # channel spacing in m
nx = fp['Acquisition']['Raw[0]'].attrs['NumberOfLoci'] # number of channels
ns = fp['Acquisition']['Raw[0]']['RawDataTime'].attrs['Count'] # number of samples
L = fp['Acquisition'].attrs['GaugeLength'] # gauge length in m
n = fp['Acquisition']['Custom'].attrs['Fibre Refractive Index'] # refractive index
scale_factor = (2*np.pi)/2**16 * (1550.12 * 1e-9)/(0.78 * 4 * np.pi * n * L)

# Create some useful arrays
t = np.arange(ns)/fs # time in s
x = np.arange(nx)*dx # distance in m
frq = np.fft.rfftfreq(ns,d=1./fs) # frequency in Hz

# Calculate and plot the spectrum for a few channels
chans = [100,1000,2000,4500]
fig,ax = plt.subplots(1,2,figsize=(14,6))
for chan in chans:
    tr = rawData[:, chan].astype(float) # get the data at each channel
    tr -= np.mean(tr) # remove the mean
    tr *= scale_factor # convert to strain
    ftr = 20 * np.log10( (2/ns) * abs(np.fft.rfft(tr * np.hamming(ns))) ) # calculate the PSD
    ax[0].plot(t,tr)
    ax[1].plot(frq,ftr,label='%.1f km' % (chan*dx*1e-3))
ax[1].set_xscale('log')
ax[1].set_xlim([1e-2,1e2])
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Amplitude')
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('PSD')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

# Plot the 2D data
plt.figure(figsize=(8,6))
plt.imshow(rawData,aspect='auto',extent=[0,t[-1],0,x[-1]],cmap='gray')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.show()
plt.close()



# CLOSE THE FILE
fp.close()



