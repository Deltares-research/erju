import os
import glob
import matplotlib.pyplot as plt
import numpy as np

file_path = r'D:\FO_culemborg_22112020'
os.chdir(file_path)

# Get a list of all .tdms files in the directory
all_files = glob.glob('*.tdms')


def search_energy(self, data_type='1Hz', plot=True):
    """
    Function to search for train in the selected data
    It also plots the data if plot=True

    @param data_type: type of data to be loaded
    @param plot: boolean to plot the data
    @return: mean amplitude of the selected data
    """

    half_way = round(self._n_traces * 0.5)
    first_channel = self._reference_channel + half_way  # 0 position of 90 m references receivers spread
    n_traces_to_search = 1  # Taking 1 trace at the middle of the 90 m spread length

    # Time is set a bit shorther to spot train energy passing in the middle of 30 second window.
    time_window = np.array([round(self._record_length * 0.2), round(self._record_length * 0.8)])
    start_time = time_window[0]
    end_time = time_window[1]
    # iDASFiles = iDASFiles[0:240] # Train energy is selected during 1 hour of measurements...........
    all_mean_amp = np.zeros((len(self._DATAFiles), 1))
    for iiz in range(0, len(self._DATAFiles)):

        if data_type == 'iDAS':

            iDAS_data, fs = ProRail_Viz_load_iDAS_Data(self._DATAFiles[iiz], start_time, end_time, first_channel,
                                                       n_traces_to_search)
            data = iDAS_data

        elif data_type == '4_5Hz':

            Shotgather_1Hz, offset_1Hz, Shotgather_4_5Hz, offset_4_5Hz, timestamp, time_range = ProRail_Viz_load_GEODE_Data(
                self._DATAFiles[iiz])
            data = Shotgather_4_5Hz

        elif data_type == '1Hz':

            Shotgather_1Hz, offset_1Hz, Shotgather_4_5Hz, offset_4_5Hz, timestamp, time_range = ProRail_Viz_load_GEODE_Data(
                self._DATAFiles[iiz])
            data = Shotgather_1Hz

        else:
            print('Please, select an appropiate data type')

        all_mean_amp[iiz, :] = np.mean(np.abs(data))

        if plot == True:

            import matplotlib
            matplotlib.rc('xtick', labelsize=14)
            matplotlib.rc('ytick', labelsize=14)
            fig, ax = plt.subplots(figsize=(15, 6))
            rec_num = np.arange(0, len(all_mean_amp))
            plt.plot(rec_num, all_mean_amp)
            plt.plot(np.arange(0, 500), np.ones((500)) * np.mean(all_mean_amp), '--', color='red')
            plt.xlabel('Record Number', fontsize=14)
            plt.ylabel('Mean amplitude', fontsize=14)
            # plt.xlim(0,max(rec_num))
            plt.grid()
            plt.show()

        else:

            print("Please, plot all computed rms values per record")

        return (all_mean_amp)