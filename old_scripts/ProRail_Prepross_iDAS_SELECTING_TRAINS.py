# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:40:20 2020

@author: obandohe
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
    
    if len(Allfiles)>=100:
        
        print("Please, consider selecting a less number of records")
        
    elif len(Allfiles)>=500:
        
        print("You cannot use more than 500 records. The computation will be too slow")
        
    # Extract all attributed from iDAS data
    '''
    INPUT: 
        
        path:               Directory of folder of tdms collected records (e.g.: D:\\iDAS_ProRail_09112020_continuous_measurements_30s)
    
    OUTPUT:
        
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

    return(data,fs)


def ProRail_Viz_load_GEODE_Data(GEODEFiles):
    from obspy import read
    # Search for all tdms files contained in the directory.    
    st=read(GEODEFiles)   # Obspy function to open SEG2 data format
    size=np.shape(st)
    ShotGather_ref=np.zeros((size[1], size[0]))
    for i in range(0, size[0]):
        tr=np.reshape(st[i],(1,size[1]))
        ShotGather_ref[:, i] = tr
    Shotgather_norm=ShotGather_ref
    fs = st[0].stats.sampling_rate
    timestamp = st[0].stats.starttime.datetime
    index = np.array([0,5,10,15,20,47,45,42,40,35,30,25]) # Indixed according to the field geometry
    index2 = index+48
    All_traces_1Hz = Shotgather_norm
    offset_1Hz = np.array([0,10,20,30,40,45,50,55,60,70,80,90])
    Shotgather_1Hz = (All_traces_1Hz[:,index2])
    Shotgather_4_5Hz=np.zeros((15000,48))
    Shotgather_4_5Hz[:,0:24]=Shotgather_norm[:,24:48]
    Shotgather_4_5Hz[:,24:48]=np.fliplr(Shotgather_norm[:,0:24])
    Shotgather_4_5Hz = np.fliplr(Shotgather_4_5Hz)
    offset_4_5Hz = np.arange(0,48)+23
    time_range = np.arange(0,len(Shotgather_4_5Hz[:,0]))/fs
    return(Shotgather_1Hz,offset_1Hz,Shotgather_4_5Hz,offset_4_5Hz,timestamp,time_range)


def ProRail_Viz_Array_to_ObsPy(data,fs,fs_new,resampling):

    import numpy as np
    from obspy import Trace, Stream,UTCDateTime
    
    sg_norm = data
    m,n=np.shape(sg_norm)
    st = Stream()
        
    for ii in range(0,n):
    
        trace = Trace(header={'station': 'DAS', 'channel': 'Z'+str(ii)})
        trace.data = sg_norm[:,ii]
        #--Default values
        year=2020
        month = 10
        day=14
        hours=11
        minutes=5
        seconds=5
        
        trace.stats.starttime=UTCDateTime(year, month, day, hours, minutes, seconds, 000000)
        trace.stats.npts=len(trace.data)
        trace.stats.sampling_rate=fs
        trace.stats.station=str(ii)
        trace.stats.channel='Z'+str(ii)
        trace.stats.distance = ii
        trace.stats.network = 'DAS'
        
        st.append(trace)
    
    st.detrend('constant')
    st.taper(0.01, type='cosine')       
    st.filter("lowpass", freq=35)
    
    if resampling =='False':
        st = st   
    elif resampling== 'True':
        fs = fs_new
        st.resample(fs)
        
    return(st)


def Array_to_ObsPy(data,fs):

    import numpy as np
    from obspy import Trace, Stream,UTCDateTime
    size = np.shape(data)
    st = Stream();
        
    for ii in range(0,size[1]):
    
        trace = Trace(header={'station': 'DAS', 'channel': 'Z'+str(ii)});
        trace.data = data[:,ii];
        
        #--Default values
        year=2020
        month = 11
        day=9
        hours=13
        minutes=35
        seconds=5
        trace.stats.starttime=UTCDateTime(year, month, day, hours, minutes, seconds, 000000);
        trace.stats.npts=len(trace.data);
        trace.stats.sampling_rate=fs;
        trace.stats.station=str(ii);
        trace.stats.channel='Z'+str(ii);
        trace.stats.distance = ii;
        trace.stats.network = 'DAS';
        
        st.append(trace);
       
    return(st)


def resampled_data(data,fs,fs_new):
    st = Array_to_ObsPy(data,fs)
    st.resample(fs_new)
    resampled_array = Obspy_to_Array(st)
    return(resampled_array)


def Obspy_to_Array(st):
    import numpy as np
    size = np.shape(st)
    Array = np.zeros((size[1], size[0]))
    
    for ii in range(0,size[0]):
       
        Array[:,ii] = st[ii].data
        
    return(Array)


def Convert_2d_to_3d_array(data_1hour,m_iter):

    import numpy as np
    import math
    m,n = np.shape(data_1hour)
    tot_samples = m*n
    #m_iter = 45000 # m_iter can be 30,000,40,000,45,000,50,000,60,000
    n_col = n # Fix value
    n_slices = math.floor(tot_samples/(n_col*m_iter))
    m_new = m_iter*n_slices
    data_1hour_new = data_1hour[0:m_new,:]
    interval = np.append(np.arange(0,m_new,m_iter),m_new)
    Data_3D_new=np.zeros((m_iter,n,n_slices))
    
    for j in range(0,len(interval)-1):
        
        slide = data_1hour_new[interval[j]:interval[j+1],:]
    
        Data_3D_new[:,:,j] = slide

    return (Data_3D_new)


def Convert_3d_to_2d_array(data_3D):
    data_2D = data_3D.transpose(2,0,1).reshape(-1,data_3D.shape[1])
    return(data_2D)
    


def feed_sql_database(database_name,csv_file2load,path):

    import pandas as pd
    import sqlite3
    import os
    os.chdir(path)    
    dataset=pd.read_csv(csv_file2load)
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    save2name = str(csv_file2load)
    dataset.to_sql(save2name[0:len(save2name)-4], con=conn,if_exists='replace') # Adding datt to database
    cursor.close()
    conn.close()



class RecordSelection:

    def __init__(self,DATAFiles,fs,reference_channel,n_traces,record_length):
    
        self._DATAFiles = DATAFiles
        self._fs = fs
        self._reference_channel = reference_channel
        self._n_traces = n_traces
        self._record_length = record_length
    
    def search_energy(self,data_type='1Hz',plot=True):
    
        import matplotlib.pyplot as plt
        import numpy as np
        half_way = round(self._n_traces*0.5)
        first_channel = self._reference_channel+half_way # 0 position of 90 m references receivers spread
        n_traces_to_search = 1            # Taking 1 trace at the middle of the 90 m spread length
        # Time is set a bit shorther to spot train energy passing in the middle of 30 second window.
        time_window =np.array([round(self._record_length*0.2),round(self._record_length*0.8)])
        start_time= time_window[0]
        end_time= time_window[1]
        #iDASFiles = iDASFiles[0:240] # Train energy is selected during 1 hour of measurements...........
        all_mean_amp = np.zeros((len(self._DATAFiles),1))
        for iiz in range(0,len(self._DATAFiles)):
            
            
            if data_type == 'iDAS':
            
                iDAS_data,fs= ProRail_Viz_load_iDAS_Data(self._DATAFiles[iiz],start_time,end_time,first_channel,n_traces_to_search)
                data = iDAS_data
                
            elif data_type == '4_5Hz':
            
                Shotgather_1Hz,offset_1Hz,Shotgather_4_5Hz,offset_4_5Hz,timestamp,time_range=ProRail_Viz_load_GEODE_Data(self._DATAFiles[iiz])
                data = Shotgather_4_5Hz
                
            elif data_type == '1Hz':
            
                Shotgather_1Hz,offset_1Hz,Shotgather_4_5Hz,offset_4_5Hz,timestamp,time_range=ProRail_Viz_load_GEODE_Data(self._DATAFiles[iiz])
                data = Shotgather_1Hz
                
            else:
                print('Please, select an appropiate data type')
            
            all_mean_amp[iiz,:] = np.mean(np.abs(data))
            print(all_mean_amp)
    
    
        if plot == True:
    
            import matplotlib 
            matplotlib.rc('xtick', labelsize=14) 
            matplotlib.rc('ytick', labelsize=14)
            fig, ax = plt.subplots(figsize=(15,6))
            rec_num = np.arange(0,len(all_mean_amp))
            plt.plot(rec_num, all_mean_amp)
            #plt.plot(np.arange(0,500),np.ones((500))*np.mean(all_mean_amp),'--',color='red')
            plt.xlabel('Record/file Number',fontsize=14)
            plt.ylabel('Mean amplitude',fontsize=14)
            #plt.xlim(0,max(rec_num))
            plt.grid()
            plt.show()
    
        else:
            
            print("Please, plot all computed rms values per record")
    
        return(all_mean_amp)
    
    
    class getData:
    
        def __init__(self,all_mean_amp,set_limit):
            
            self._set_limit = set_limit
            self._all_mean_amp = all_mean_amp
            self._DATAFiles = DATAFiles
            self._fs = fs
            self._reference_channel = reference_channel
            self._n_traces = n_traces
            self._record_length = record_length
        
        def get_energetic_iDAS_records(self):
            import numpy as np
            index_train = np.array(np.where(self._all_mean_amp[:,0]>self._set_limit))[0,:]
            start_time = 0
            end_time = self._record_length
            rec_names_selected = {}
            selected_data = np.zeros((int(self._record_length*self._fs)+1,self._n_traces+1,len(index_train)))
            for ii in range(0,len(index_train)):
                
                data,fs= ProRail_Viz_load_iDAS_Data(self._DATAFiles[index_train[ii]],start_time,end_time,reference_channel,n_traces) 
                
                selected_data[:,:,ii] = data
                rec_names_selected[ii] = self._DATAFiles[index_train[ii]]
            return(rec_names_selected,selected_data)
        
        def get_energetic_GEODE_records(self):
            
            import numpy as np
            index_train = np.array(np.where(self._all_mean_amp[:,0]>self._set_limit))[0,:]
            rec_names_selected = {}
            selected_data = np.zeros((int(self._record_length*self._fs),self._n_traces,len(index_train)))
            for ii in range(0,len(index_train)):
                
                Shotgather_1Hz,offset_1Hz,Shotgather_4_5Hz,offset_4_5Hz,timestamp,time_range=ProRail_Viz_load_GEODE_Data(self._DATAFiles[index_train[ii]])
                
                selected_data[:,:,ii] = Shotgather_1Hz
                rec_names_selected[ii] = self._DATAFiles[index_train[ii]]
            return(rec_names_selected,selected_data)
    
       
        
#%% #######################################################END OF MAIN FUNCTIONS#########################################################################################3

import numpy as np
import os
path = r'D:\FO_culemborg_22112020'
#path = r'C:\Projects\erju\data'
#path = r'C:\Users\obandohe\OneDrive - Stichting Deltares\Documents\DELTARES PROJECTS_2020\06_PRORAIL_PROJECT\01_FIELD_MEASUREMENTS\FIELD_MEASUREMENTS_PRORAIL_09112020\ProRail-09112020-signal-test'
os.chdir(path)
import glob
#Allfiles=glob.glob('*.dat')
Allfiles=glob.glob('*.tdms')


#%%
Allfiles = Allfiles[0:400]  # 1--0:1500; 2 -- 2548:4110;  3 -- 5428:6988

#%%
DATAFiles = Allfiles
reference_channel = 100
n_traces = 100
fs = 1000
record_length = 30
rs = RecordSelection(DATAFiles,fs,reference_channel,n_traces,record_length)
All_mean = rs.search_energy(data_type='iDAS',plot=True)
#%%
set_limit = 400 
data = rs.getData(All_mean,set_limit)

rec_names_selected,selected_data = data.get_energetic_iDAS_records()

print('rec_names_selected')
print(rec_names_selected)
print('selected_data')
print(selected_data)

# #from segypy import wiggle
#
# #wiggle(selected_data[:,:,12])
#
#
# #%%
#
# data_2d = Convert_3d_to_2d_array(selected_data)
#
#
#
# #%np.save('Trains_Only_4200.npy',selected_data)
#
# import pandas
# df = pandas.DataFrame(data_2d,columns=['iDAS'+ str(i) for i in range(0,201)])
#
# df_names = pandas.DataFrame(rec_names_selected,index=['filenames'])
#
# df_names = df_names.transpose()
#
# df_mean = pandas.DataFrame(All_mean,columns=['mean'])
#
#
# df.to_csv(r'C:\Projects\erju\test\iDAS_4900_5200_22112020_1.csv')
# df_names.to_csv(r'C:\Projects\erju\test\iDAS_4900_5200_22112020_1_filenames.csv')
# df_mean.to_csv(r'C:\Projects\erju\test\iDAS_4900_5200_22112020_1_mean.csv')
#
#
#
#
#
#
#
#
#
#
#
#










    

