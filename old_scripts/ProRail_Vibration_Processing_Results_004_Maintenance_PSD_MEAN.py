# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:15:24 2021

@author: obandohe
"""
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

def cal_fact(sel_chan):

    from io import  StringIO
    import numpy as np
    import pandas as pd
    
    """
    INPUT:
        
        ACCFiles:*.ASC files to be loaded
        fs:sampling frequency of loaded record
        sel_chan: Channels names to be loaded.
        See description below:
            
            locatie	sensornr	meetrichting	meetkanaal
            1	               V1	v	ch 0
            1	               V2	hl	ch 1
            2	               V3	v	ch 2
            2	               V4	hl	ch 3
            5	               V5	v	ch 4
            5	               V6	hl	ch 5
            6	               V7	v	ch 6
            6	               V9	hl	ch 7
            9	              V10	v	ch 8
            9	              V11	hl	ch 9
            3	              tc1	v	ch 10
            3	              tc1	hl	ch 11
            3	              tc1	he	ch 12
            4	              tc3	v	ch 13
            4	              tc3	hl	ch 14
            4	              tc3	he	ch 15
            7	              tc4	v	ch 16
            7	              tc4	hl	ch 17
            7	              tc4	he	ch 18
            8	              tc5	v	ch 19
            8	              tc5	hl	ch 20
            8	              tc5	he	ch 21
    
    
    
    RETURN:
        
        data_acc: Extracted traces from desired channels
        time_acc: time vector for extractec traces.
    
    """
    # channrl names and indixes used to select calibration factors

    


    chan_name = sel_chan
    
    if chan_name=='V1':
        index = 0
    elif chan_name=='V2':
        index = 1
    elif chan_name=='V3':
        index = 2
    elif chan_name=='V4':
        index = 3
    elif chan_name=='V5':
        index = 4
    elif chan_name=='V6':
        index = 5
    elif chan_name=='V7':
        index = 6
    elif chan_name=='V9':
        index = 7
    elif chan_name=='V10':
        index = 8
    elif chan_name=='V11':
        index = 9
    elif chan_name=='tc1_v':
        index = 10
    elif chan_name=='tc1_hl':
        index = 11
    elif chan_name=='tc1_he':
        index = 12
    elif chan_name=='tc3_v':
        index = 13
    elif chan_name=='tc3_hl':
        index = 14
    elif chan_name=='tc3_he':
        index = 15
    elif chan_name=='tc4_v':
        index = 16
    elif chan_name=='tc4_hl':
        index = 17
    elif chan_name=='tc4_he':
        index = 18
    elif chan_name=='tc5_v':
        index = 19
    elif chan_name=='tc5_hl':
        index = 20
    elif chan_name=='tc5_he':
        index = 21
    
    
    # Calibration factors to convert volts to mg

    cal_factor = np.array([-203.874, -206.249, -207.018,-202.265,-203.190,-202.429,-197.336,-194.420,-198.748,-202.573,-88.917,-93.756,-94.011,-90.921,-88.480,-92.473,-203.046,-94.424,-95.007,-199.900,-203.479,-89.206,1.0,1.0])


    cf = cal_factor[index] # convert volts to g (acceleration are divided by 1000 to convert to g's)
    

    return cf


def acc2vel(acc,fs): 
    
    """
    INPUT: 
        
        acc : 1-D Time history acceleration signal.
        fs = sampling frequency in Hz
        
    RETURN:
        
        Vel: velocity time history
    
    """
    
    from scipy import integrate
    dt = 1/fs
    vel = integrate.cumtrapz(acc,dx=dt,initial=0) # using Trapezoid method
    return(vel)


import numpy as np


def ProRail_filter(data,fs,fmin,fmax,detrend='true',filter_signal='true'):

    
    """
    
    INPUT: 
        
        data: 1-D array
        fs: sanmpling frequency in Hz
        fmin: minimum frequency
        fmax: maximum frequency
        detrend: string set as "True by default"
        filter_signal = string if True filtering is applied, if False signal is not filtered
    
    
    RETURN:
        
        tr.data: 1-D array 
    
    """
    

    from obspy import Trace
    tr = Trace(header={'station': 'DAS', 'channel': 'Z'})
    tr.data = data
    tr.stats.npts=len(data)
    tr.stats.sampling_rate=fs
    tr.stats.station='TestSite'
    tr.stats.channel='Z'
    tr.stats.network = 'DAS'
    
    if detrend == 'true':
        tr.detrend('constant')
    elif detrend =='false':
        tr = tr 
    else:
        
        print('Please, indicate True or False to pass the argument')
            
    if filter_signal == 'true':
        
        tr.taper(0.05, type='cosine')
        tr.filter("bandpass", freqmin=fmin,freqmax=fmax,corners=4, zerophase=True)
        
    elif filter_signal == 'false':
        tr = tr
        
    else:
        print('Please, indicate True or False to pass the argument')
        
    return (tr.data)

def manual_picking(shotgather):
    
    """
        Need to produce a wiggle plot using iDAS_pre_processing.wiggle_plot before introducing manual picking. \n
        Double (left) click: pick individual point. Note: Must wait until the picked point is shown on the plot before proceeding to the next pick. \n
        Right click: finish picking and store picked points. The function will not complete execution until a right click has been given.\n
   
        Returns: every second element of the picked points (they are saved twice). 
    """
    
    import matplotlib.pyplot as plt
    import numpy as np

    counter = 0 
    n_point = 1
    xy = np.zeros([100, 2])
    
    while n_point==1: 
        
        picked_coords = plt.ginput(n_point, timeout=-1, show_clicks=True, mouse_add=1, mouse_pop=2, mouse_stop=3)
        
        counter += 1
        
        if picked_coords == []:
            
            break
        
        else: 
        
            xy[counter,:] = np.array(picked_coords)
            index_plot = np.transpose(np.array(np.where(np.abs(xy[:,0])>0)))
            saved_coords = np.unique(xy[index_plot[:,0],:],axis=0)   
              
    return(saved_coords)



def Power_Spectral_Density_function(Signal,fs,win_len):
    
    from scipy import signal
    
    ff, pxxa = signal.welch(Signal, fs,nperseg=win_len, nfft=win_len,scaling='density',return_onesided=True)

    return(ff, pxxa)


def load_accData_component(Allfiles,Alltime,sel_chan,fs,fmin,fmax):
    
    import numpy as np
    import pandas as pd
    
    full_segments = np.zeros((30000,1))

    for iik in range(0,12):

        filename = Allfiles[iik]
        df = pd.read_csv(filename)
        array = df[sel_chan]
        
        acc_data = array.to_numpy()
        
        filename = Alltime[iik]
        df = pd.read_csv(filename)
        array = df.to_numpy()
        time_seg = array[:,1:]
        
        time_seg = time_seg.astype(int)
        
        All_segments = np.zeros((30000,len(time_seg[0,:])))  
        
        for ix in range(0,len(time_seg[0,:])):
            
            All_segments[:,ix] =  acc_data[time_seg[:,ix]]
        
        full_segments = np.hstack((All_segments,full_segments))
    
    mon,non = np.shape(full_segments) 
    
    full_segments = full_segments[:,0:non-1]

    return full_segments



def ouput_component(full_segments,fs,win_len,sel_chan):
    m,n = np.shape(full_segments)
    vel = np.zeros((30000,n))
    acc = np.zeros((30000,n))
    psd = np.zeros(((int(win_len/2)+1),n))
    cf = cal_fact(sel_chan)
    for ik in range(0,n):
        acc_filt = ProRail_filter((full_segments[:,ik]*cf/1000)*9.81,fs,fmin,fmax,detrend='true',filter_signal='true')
        vel0 = acc2vel(acc_filt,fs)
        freq,amp = Power_Spectral_Density_function(vel0,fs,win_len)
        psd[:,ik] = amp
        vel[:,ik] = vel0
        acc[:,ik] = acc_filt
    return(vel,acc,psd,freq)



def plot_component(rec,rec_name,comp_name,units):
    
    fig,ax = plt.subplots(figsize=[15,6])
    
    time = np.arange(0,len(rec))/fs
    max_val = np.max(np.abs(rec))
    loc_x = np.array(np.where(max_val==np.abs(rec)))
    loc_x = loc_x[0]
    print(loc_x)
    plt.plot(time,rec)
    plt.xlim(min(time),max(time))
    plt.xlabel('Time[s]',fontsize=14)
    plt.ylabel(units,fontsize=14)
    plt.plot(loc_x/fs,rec[loc_x],'o',markersize=8,color='red')
    plt.text(loc_x/fs,rec[loc_x],str(np.round(max_val,5)))
    plt.title(comp_name)
    plt.grid()
    plt.savefig(rec_name,dpi=150)



def get_all_psd(df,psd1,train_type):
    
    
    direction1 = 'u'
    direction2 = 'db'
    
    df_1 = []
    
    df_1=df
    
    df21 = df_1[df_1.type == train_type]
    
    df31 = df21[df21.direction == direction1]
    index_r1 = df31['Index']
    df32 = df21[df21.direction == direction2]
    index_r2 = df32['Index']
    index_sel1 = index_r1.to_numpy()
    index_sel2 = index_r2.to_numpy()
    All_PSD1 = psd1[:,index_sel1]
    All_PSD2 = psd1[:,index_sel2]
    
    mean_psd1 = np.mean(All_PSD1,axis=0)
    std_psd1 = np.std(mean_psd1)
    str_err1 = std_psd1/np.sqrt(len(mean_psd1))
    
    mean_psd2 = np.mean(All_PSD2,axis=0)
    std_psd2 = np.std(mean_psd2)
    str_err2 = std_psd2/np.sqrt(len(mean_psd2))
    

    all_mean = np.array([np.mean(mean_psd1),np.mean(mean_psd2)])
    all_std = np.array([str_err1,str_err2])

    return(All_PSD1,All_PSD2,all_mean,all_std)


def plot_results(all_mean1,all_std1,all_mean2,all_std2,comp_name):

    fig,(ax0,ax1) = plt.subplots(ncols=2,figsize=(10,6))
    
    dir_train=['u','db']
    
    ax0.bar(dir_train,all_mean1,yerr=all_std1,width=0.4,linewidth=2,capsize=8)
    ax0.set_xlabel('Travelling direction',fontsize=14)
    ax0.set_ylabel('Mean PSD [m/s]^2/Hz',fontsize=14)
    ax0.set_ylim(0,4e-11)
    ax0.grid()
    ax0.set_title(comp_name + '_09112020',fontsize=14)
    
    ax1.bar(dir_train,all_mean2,yerr=all_std2,width=0.4,linewidth=2,capsize=8)
    ax1.set_xlabel('Travelling direction',fontsize=14)
    ax1.set_ylabel('Mean PSD [m/s]^2/Hz',fontsize=14)
    ax1.set_ylim(0,4e-11)
    ax1.grid()
    ax1.set_title(comp_name + '_11112020',fontsize=14)


def plot_psd(All_PSD1,All_PSD2,freq1,comp_name):


    fig1,(ax0,ax1) = plt.subplots(nrows=2,figsize=(8,6))
    ax0.plot(freq1,All_PSD1,color='black',linewidth=1.0)
    ax0.set_xlabel('Frequency[Hz]',fontsize=14)
    ax0.set_ylabel('PSD [m/s]^2/Hz',fontsize=14)
    ax0.set_title(comp_name+'--'+train_type + ' -- u',fontsize=14)
    ax0.set_xlim(fmin,fmax)
    ax0.grid()
    ax1.plot(freq1,All_PSD2,color='black',linewidth=1.0)
    ax1.set_xlabel('Frequency[Hz]',fontsize=14)
    ax1.set_ylabel('PSD [m/s]^2/Hz',fontsize=14)
    ax1.set_title(comp_name+'--'+train_type + ' -- db',fontsize=14)
    ax1.set_xlim(fmin,fmax)
    ax1.grid()
    fig1.tight_layout()

#%---------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import os
import matplotlib.pylab as plt
import glob


comp_name = ['N05V','N05H','N09V','N09H','N24V','N24H','S09V','S09H','S24V','S24H','N13V','N13H','N13P','N18V','N18H','N18P','S13V','S13H','S13P','S18V','S18H','S18P']

All_comp = ['V1','V2','V3','V4','V5','V6','V7','V9','V10','V11','tc1_v','tc1_hl','tc1_he','tc3_v','tc3_hl','tc3_he','tc4_v','tc4_hl','tc4_he','tc5_v','tc5_hl','tc5_he']

fs = 1000
fmin = 1.0
fmax = 100.0
win_len = 1024


#------------------------------
import matplotlib 
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)

train_type = 'd'

mtyp = ['o','^']
lw = ['--o','--^']
colr = ['black','gray']


for ikk in range(0,len(comp_name)):

    sel_chan = All_comp[ikk]

    
    path1 = r'F:\PRORAIL_CSV_FILES\VIBRATIONS\csv_files_accelerometers\Culemborg_09112020'
    os.chdir(path1)
    Allfiles1=glob.glob('*.csv')
    Alltime1=glob.glob('Time_segments/*.csv')
    full_segments1 = load_accData_component(Allfiles1,Alltime1,sel_chan,fs,fmin,fmax)
    vel1,acc1,psd1,freq1 = ouput_component(full_segments1,fs,win_len,sel_chan)
    #Index	type	direction	totaal aantal bakken	starttime	end time	direction	dt[s]	dx_A_B [m]	speed [km/h]	Total time difference [m]	length [m]	filename
    df_r1 = pd.read_excel(r'F:\PRORAIL_CSV_FILES\VIBRATIONS\csv_files_accelerometers\INDEX.xlsx','Trains_09112020')
     
    path2 = r'F:\PRORAIL_CSV_FILES\VIBRATIONS\csv_files_accelerometers\Culemborg_11112020'
    os.chdir(path2)
    Allfiles2=glob.glob('*.csv')
    Alltime2=glob.glob('Time_segments/*.csv')
    full_segments2 = load_accData_component(Allfiles2,Alltime2,sel_chan,fs,fmin,fmax)
    vel2,acc2,psd2,freq2 = ouput_component(full_segments2,fs,win_len,sel_chan)
    #Index	type	direction	totaal aantal bakken	starttime	end time	direction	dt[s]	dx_A_B [m]	speed [km/h]	Total time difference [m]	length [m]	filename
    df_r2 = pd.read_excel(r'F:\PRORAIL_CSV_FILES\VIBRATIONS\csv_files_accelerometers\INDEX.xlsx','Trains_11112020')
     
    All_PSD_11,All_PSD_21,all_mean_1,all_std_1 = get_all_psd(df_r1,psd1,train_type)
    All_PSD_12,All_PSD_22,all_mean_2,all_std_2 = get_all_psd(df_r2,psd2,train_type)
    
    plot_results(all_mean_1,all_std_1,all_mean_2,all_std_2,comp_name[ikk])


#%%