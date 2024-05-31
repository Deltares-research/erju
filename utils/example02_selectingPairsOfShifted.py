#%%
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# adding modules to the system path
sys.path.insert(0, r'C:\Users\obandohe\OneDrive - Stichting Deltares\GIT_PROJECTS\erju')
from utils.utils import Array_to_ObsPy, filtering, getShift, stalta, getON, plotONOF

#%%
# Reading csv file with pair of signals (e.g. DAS and accelerometer)
df = pd.read_csv(r'C:\Users\obandohe\OneDrive - Stichting Deltares\Documents\DELTARES_PROJECTS\2024\12_RAIL4EARTH\csv_file_test\signals_shifted.csv')
df.head()

signal = df['sensor1']  # Extracting signal
fs = df['fs'].iloc[0]    # Extracting sampling frequency


# Defning input parameters
ymin = 0.5       # Lowest threshold of characteristic function cft
ymax = 7.0       # Highiest threshold of characteristic function cft
sta = 0.5        # Lenght of the average short time average window
lta = 5.0        # Lenght of the average long time average window

# Computing characteristic function and time limits
on_of,cft = getON(signal,fs,ymin,ymax,sta,lta)

# Plotting characteristic function
plotONOF(signal,cft, on_of)


# %%#############----Extracting individual train passages-----######################

t_init = on_of[:,0]
window = 12*fs # in seconds
signal_arr1 = np.zeros((window,len(t_init)))
signal_arr2 = np.zeros((window,len(t_init)))

for index, t_val in enumerate(t_init):
    signal_arr1[:,index] = np.array(df['sensor1'])[int(t_val-window/2):int(t_val+window/2)]
    signal_arr2[:,index] = np.array(df['sensor2'])[int(t_val-window/2):int(t_val+window/2)]



# %%


for i in range(len(t_init)):

    fig,axs = plt.subplots()

    axs.plot(signal_arr1[:,i])
    axs.plot(signal_arr2[:,i])

# %%
