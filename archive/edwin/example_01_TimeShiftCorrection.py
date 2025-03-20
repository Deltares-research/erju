
#%%
import sys
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from obspy import Trace
# adding modules to the system path
sys.path.insert(0, r'/')
from utils import Array_to_ObsPy,filtering, getShift, stalta, getON, plotONOF

#%%##############################---LOADING SIGNALS----###################################

# Reading csv file with pair of signals (e.g. DAS and accelerometer)
df = pd.read_csv(r'D:\csv_file_test\test_data.csv')
df.head()


# Extracting sampling frequency and signals
# Extracting sampling frequency and signals
fs = df['fs'].iloc[0]
data = np.array(df[['sensor1','sensor2']])

#%%##############################---FILTERING SIGNALS----###################################
fmin = 1.0
fmax = 30.0
filt_data = filtering(data, fs, fmin, fmax, filter_type='bandpass')

# Extracting filtered signals from obspy object
filt_data01 = filt_data[:,0]
filt_data02 = filt_data[:,1]

#%%##############################---TIME SHIFT CORRECTION USING STA_LTA Method----###################################

# Defning input parameters
ymin = 0.5       # Lowest threshold of characteristic function cft
ymax = 7.0       # Highiest threshold of characteristic function cft
sel_time = 2500  # This time contains the first signal or train passage
sta = 0.5        # Lenght of the average short time average window
lta = 5.0        # Lenght of the average long time average window

# correcting time shift of DAS and accelerometers signals
data_corr1,data_corr2 = getShift(filt_data01,filt_data02,sel_time,fs,ymax,ymin,sta,lta)

# Plotting shifted traces respect to original traces
fig,(ax0,ax1) = plt.subplots(2,1,figsize=(6,4))
ax0.plot(filt_data01)
ax0.plot(filt_data02)
ax1.plot(data_corr1)
ax1.plot(data_corr2)

ax0.set_title("Unshifted signals")
ax1.set_title("Shifted signals")
ax0.set_xlabel('Samples')
ax1.set_xlabel('Samples')
fig.tight_layout()
# %% save shifted signals as csv file
plt.show()

df_results = pd.DataFrame({'sensor1':data_corr1,'sensor2':data_corr2,'fs':fs})
df_results.head(5)

#df_results.to_csv(r'C:\Users\obandohe\OneDrive - Stichting Deltares\Documents\DELTARES_PROJECTS\2024\12_RAIL4EARTH\csv_file_test\signals_shifted.csv',index=False)

# %%
