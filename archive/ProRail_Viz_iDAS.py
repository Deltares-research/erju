# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:40:20 2020

@author: deltares
"""

######################################################################MAIN FUNCTIONS##################################################
def ProRail_Viz_init_iDAS_Data(path):
    # All imports to perform operation. Import must be installed in python using pip install name of import
    import os
    import glob
    os.chdir(path)    
    # Search for all tdms files contained in the directory.
    Allfiles=glob.glob('*.tdms')
    
    from tdms_reader import TdmsReader 
    import numpy as np 
    tdms = TdmsReader(Allfiles[0])  
    props = tdms.get_properties() 
    
        
    # Extract all attributed from iDAS data
    '''
    INPUT: 
        
        path:               Directory of folder of tdms collected records (e.g.: D:\\iDAS_ProRail_09112020_continuous_measurements_30s)
    
    RETURN:
        
        props:              Properties of the record
        fs:                 Sampling frequency in Hz
        n_samples:          maximum number of samples for each record
        distance_vector     Distance vector along the fiber in meters
        dx:                 Channel spacing assigned by iDAS.
    
    '''
    ##----Extract important properties----##
    zero_offset = props.get('Zero Offset (m)') 
    dx = props.get('SpatialResolution[m]') * props.get('Fibre Length Multiplier')
    n_channels = tdms.fileinfo['n_channels']
    distance = zero_offset + np.arange(n_channels+1) * dx 
    fs = props.get('SamplingFrequency[Hz]')
    n_samples=tdms.channel_length 


    return(Allfiles,props,fs,n_samples,distance,dx)


def ProRail_Viz_load_iDAS_Data(Allfiles,start_time,end_time,first_channel,n_traces):

    from tdms_reader import TdmsReader
    import numpy as np
    
    """
    Function to extract iDAS data from tdms files. 
    
    INPUT:
        
        Allfiles    : tdms files contained in the working directory
        start_time  : user selectable initial time
        end_time    : user seletable end time of selected file
        first_channel: User seleted first channel
        n_traces     : Number of traces to be analized
        
    RETURN:
        
        data: 2D array containing selected traces.
        fs  : sampling frequency of extracted traces in Hz.
    
    """ 
    
    
    tdms = TdmsReader(Allfiles)
    props = tdms.get_properties()
    zero_offset = props.get('Zero Offset (m)') 
    dx = props.get('SpatialResolution[m]') * props.get('Fibre Length Multiplier')
    n_channels = tdms.fileinfo['n_channels']
    distance = zero_offset + np.arange(n_channels) * dx 
    fs = props.get('SamplingFrequency[Hz]')    
    first_channel_index= (np.abs(distance-first_channel)).argmin()
    last_channel_index = (first_channel_index) + n_traces
    start_time=start_time*fs
    end_time=end_time*fs
    
    data = tdms.get_data(int(first_channel_index), int(last_channel_index), int(start_time), int(end_time))
    all_data = tdms.get_data()
    print(all_data.shape)

    data_4200 = tdms.get_data(4200, 4300, 0, 30)
    print(data_4200.shape)

    return(data,fs)

     
def plot_imshow(data,rec_name,first_channel,n_traces,start_time,end_time,save_figure=True):
    
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
    
    
    import matplotlib.pyplot as plt
    import numpy as np 
    import matplotlib 
    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14)
    fig, ax = plt.subplots(figsize=(15,8))
    Z=np.abs(data)
    plt.imshow((Z),interpolation='kaiser', aspect='auto',cmap='jet',extent=[first_channel,first_channel+n_traces,end_time,start_time],vmax=Z.max()*0.30)
    plt.xlabel('Distance[m]',fontsize=14)
    plt.ylabel('Time [s]',fontsize=14)
    plt.title(rec_name,fontsize=14)
    plt.show(block=False)
    
    if save_figure==True:
    
        plt.savefig(rec_name + '.jpg',dpi=150)
        
    elif save_figure==False:
        
        pass
        

#% #######################################################END OF MAIN FUNCTIONS#########################################################################################3

if __name__ == '__main__':
    import os
    path = r'C:\Projects\erju\data' # directory should be replaced by the correct user’s directory
    os.chdir(path)
    Allfiles,props,fs,n_samples,distance,dx=ProRail_Viz_init_iDAS_Data(path)
    
    start_time = 0              # Starting time in seconds
    end_time = 30               # End time in seconds
    first_channel = 4200        # First channel at the north side of the test site
    n_traces = 100              # Number of traces to upload
    rec_name = Allfiles[0]      # Selected record to be ploted
    
    data_iDAS,fs=ProRail_Viz_load_iDAS_Data(rec_name,start_time,end_time,first_channel,n_traces)
    

    plot_imshow(data_iDAS,rec_name,first_channel,n_traces,start_time,end_time,save_figure=True)
    














    

