
#%%
import sys
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from obspy import Trace
# adding modules to the system path
sys.path.insert(0, r'C:\Projects\erju')
from utils import Array_to_ObsPy,filtering, getShift, stalta, getON, plotONOF

#%%##############################--- LOADING SIGNALS ----###################################

# Reading csv file with pair of signals (e.g. DAS and accelerometer)
df = pd.read_csv(r'D:\csv_file_test\test_data.csv')
df.head()

# Extracting sampling frequency and signals
fs = df['fs'].iloc[0]
data = np.array(df[['sensor1', 'sensor2']])

#%%##############################--- FILTERING SIGNALS ----###################################
fmin = 1.0
fmax = 30.0
filt_data = filtering(data, fs, fmin, fmax, filter_type='bandpass')
#%%##############################--- TIME SHIFT CORRECTION USING STA_LTA Method ----###################################

# Defning input parameters
ymin = 0.5       # Lowest threshold of characteristic function cft
ymax = 7.0       # Highiest threshold of characteristic function cft
sel_time = 2500  # This time contains the first signal or train passage
sta = 0.5        # Lenght of the average short time average window
lta = 5.0        # Lenght of the average long time average window

signal1 = filt_data[:,0][:sel_time]
signal2 = filt_data[:,1][:sel_time]

on_of1,cft1 = getON(signal1,fs,ymin,ymax,sta,lta)
on_of2,cft2 = getON(signal2,fs,ymin,ymax,sta,lta)

print(f'First break in signal 1: {on_of1}')
print(f'First break in signal 2: {on_of2}')
print(f'First ratio in signal 1: {cft1}' )
print(f'First ratio in signal 2: {cft2}' )


# %% Ploting characteristic functions together with first break per signal

# Signal 1
plotONOF(signal1,cft1, on_of1)

# Signal 2
plotONOF(signal2,cft2, on_of2)

# %%
