import numpy as np
from nptdms import TdmsFile as td
import matplotlib.pyplot as plt
import matplotlib
# Path to the folder containing the TDMS file
path_to_file = r'C:\Projects\erju\data\iDAS_continous_measurements_30s_UTC_20201121_101949.913.tdms'

# TDMS file reader
#tdms_file = td.read(path_to_file)
#print(tdms_file.properties)

#n_channels = len(tdms_file.groups()[0].channels())

# Define the starting channel and the number of traces
starting_channel = 4200
n_traces = 100
start_time = 0
end_time = 30

# Initialize an empty list to store the selected data
selected_data = []


with td.open(path_to_file) as tdms_file:

    for group in tdms_file.groups():
        group_name = group.name
        for channel in group.channels():
            channel_number = int(channel.name)
            # Check if the channel is within the range of selected channels
            if starting_channel <= channel_number < starting_channel + n_traces:
                # Access dictionary of properties:
                properties = channel.properties
                # Access numpy array of data for channel:
                data = channel[:]
                selected_data.append(data)

# Convert the list of selected data to a numpy array and transpose it
selected_data = np.array(selected_data)


print(selected_data.shape)


def plot_imshow(data, start_channel, n_traces, start_time, end_time, save_figure=True):
    """
    Function for plotting DAS traces

    INPUT:

        data    :       Extracted iDAS traces
        rec_name:       name of selected tdms record
        first_channel:  First channel assigned to extract traces
        start_time  : user selectable initial time
        end_time    : user seletable end time of selected file
        save_figure : string for saving plotted figure. If True figure is saved.

    """


    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    fig, ax = plt.subplots(figsize=(15, 8))
    Z = np.abs(data)
    plt.imshow((Z), interpolation='kaiser', aspect='auto', cmap='jet',
               extent=[start_channel, start_channel + n_traces, end_time, start_time], vmax=Z.max() * 0.30)
    plt.xlabel('Distance[m]', fontsize=14)
    plt.ylabel('Time [s]', fontsize=14)
    plt.show(block=False)

    if save_figure == True:

        plt.savefig('mifigurita.jpg', dpi=300)

    elif save_figure == False:

        pass


plot_imshow(selected_data, starting_channel, n_traces, start_time, end_time, save_figure=True)