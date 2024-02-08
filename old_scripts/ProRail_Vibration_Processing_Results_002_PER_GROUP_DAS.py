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




def Power_Spectral_Density_function(Signal,fs,win_len):
    
    from scipy import signal
    
    ff, pxxa = signal.welch(Signal, fs,nperseg=win_len, nfft=win_len,scaling='density',return_onesided=True)

    return(ff, pxxa)


def integrate_DAS(signal_fo,fs): 
    
    from scipy import integrate
    dt = 1/fs
    Dt = integrate.cumtrapz(signal_fo,dx=dt,initial=0)
    return(Dt)
    

def ouput_component_DAS(full_segments,fs,win_len,vel_str,fmin,fmax):
    m,n = np.shape(full_segments)
    vel = np.zeros((30000,n))
    psd = np.zeros(((int(win_len/2)+1),n))
    for ik in range(0,n):
        trace_filt = ProRail_filter((full_segments[:,ik]),fs,fmin,fmax,detrend='true',filter_signal='true')
        
        strain = integrate_DAS(trace_filt,fs)*11.6
    
        vel_das = (strain*1e-9)*vel_str
        
        freq,amp = Power_Spectral_Density_function(vel_das,fs,win_len)
        psd[:,ik] = amp
        vel[:,ik] = vel_das
        
    return(vel,psd,freq)




def plot_vel_per_group(full_segments,df,train_type,win_len,vel_str,fmin,fmax):
    
    
    vel,psd,freq = ouput_component_DAS(full_segments,fs,win_len,vel_str,fmin,fmax)
    
    #%--------------------------------------------------------------SELECTING FILTER---------------------------------------------------------
    import matplotlib 
    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14)
    
    
    if train_type == 'd':
        tab = np.array([4,6,8,10,12])
    elif train_type == 's': 
        tab = np.array([4,6,10])
    else:
        
        print('Please, choose between type d and s')
    
    #----------------------------------------------------------------------------
    
    fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(10,8))
    
    
    df21 = df_1[df.type == train_type]
    df31 = df21[df21.direction == 'u']
    vel_mean_comp1 = np.zeros((len(tab)))
    speed_mean_comp1 = np.zeros((len(tab)))
    
    df32 = df21[df21.direction == 'db']
    vel_mean_comp2 = np.zeros((len(tab)))
    speed_mean_comp2 = np.zeros((len(tab)))
    
    
    for iz in range(0,len(tab)):
        
        df41 = df31[df31.tab == tab[iz]]
        index_r1 = df41['Index']
        index_sel1 = index_r1.to_numpy()
        vel_comp1 = np.max(np.abs(vel[:,index_sel1]),axis=0)
        vel_mean_comp1[iz] = np.mean(vel_comp1)
        
        
        df42 = df32[df32.tab == tab[iz]]
        index_r2 = df42['Index']
        index_sel2 = index_r2.to_numpy()
        vel_comp2 = np.max(np.abs(vel[:,index_sel2]),axis=0)
        vel_mean_comp2[iz] = np.mean(vel_comp2)
        
        
        index_vel = df41['speed_km_h']
        index_vel1 = index_vel.to_numpy()
        speed_mean_comp1[iz] = np.mean(index_vel1)
        
        
        index_vel = df42['speed_km_h']
        index_vel2 = index_vel.to_numpy()
        speed_mean_comp2[iz] = np.mean(index_vel2)
    
        
        ax0.scatter(np.ones((len(index_vel1)))*tab[iz],index_vel1,s=70,color='black',marker='o')
        ax0.scatter(np.ones((len(index_vel2)))*tab[iz],index_vel2,s=70,color='gray',marker='^')
        ax0.grid()
        ax0.set_ylim(50,115)
        ax0.set_xlabel('Group',fontsize=14)
        ax0.set_ylabel('Speed of Train [km/h]',fontsize=14)
        ax0.set_title(comp_name[ikk])
        ax0.grid()
    
        
        ax1.scatter(np.ones((len(vel_comp1)))*tab[iz],vel_comp1,s =70,color='black',marker='o')
        ax1.scatter(np.ones((len(vel_comp2)))*tab[iz],vel_comp2,s =70,color='gray',marker='^')
        ax1.grid()
        ax1.set_xlabel('Group',fontsize=14)
        ax1.set_ylabel('Particle Velocity [m/s]',fontsize=14)
        ax1.grid()
    

    ax1.plot(tab,vel_mean_comp1,':o',label='u',color='black',linewidth=2,markersize=10)
    ax1.plot(tab,vel_mean_comp2,'--^',label='db',color='gray',linewidth=2,markersize=10)
    ax1.legend(fontsize=14,loc='upper right')
    ax0.grid()
    ax1.grid()
    fig.tight_layout()
    
    path = r'E:\PRORAIL_CSV_FILES\VIBRATIONS\FINAL_RESULTS\DAS\PER_GROUP'
    os.chdir(path)

    plt.savefig('vel_per_group_'+sel_chan,dpi=150)
    
    
def plot_psd_per_group(full_segments,df,train_type,win_len,vel_str,fmin,fmax):


    vel,psd,freq = ouput_component_DAS(full_segments,fs,win_len,vel_str,fmin,fmax)

    #%--------------------------------------------------------------SELECTING FILTER---------------------------------------------------------
    import matplotlib 
    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14)
    
    
    lt = ['-','--',':','-.','-']
    lw = np.array([2,2,2,1,1])
    
    if train_type == 'd':
        tab = np.array([4,6,8,10,12])
    elif train_type == 's': 
        tab = np.array([4,6,10])
    else:
        
        print('Please, choose between type d and s')
    
    #----------------------------------------------------------------------------
    
    fig,(ax0,ax1,ax2) = plt.subplots(nrows=3,figsize=(10,12))
    
    
    df21 = df_1[df.type == train_type]
    df31 = df21[df21.direction == direction1]
    psd_mean_comp1 = np.zeros((len(tab)))

    df32 = df21[df21.direction == direction2]
    psd_mean_comp2 = np.zeros((len(tab)))


    for iz in range(0,len(tab)):
        
        df41 = df31[df31.tab == tab[iz]]
        index_r1 = df41['Index']
        index_sel1 = index_r1.to_numpy()
        psd_comp1_scat = np.mean(psd[:,index_sel1],axis=0)
        psd_comp1 = np.mean(psd[:,index_sel1],axis=1)
        psd_mean_comp1[iz] = np.mean(psd_comp1)
        
        
        df42 = df32[df32.tab == tab[iz]]
        index_r2 = df42['Index']
        index_sel2 = index_r2.to_numpy()
        psd_comp2_scat = np.mean(psd[:,index_sel2],axis=0)
        psd_comp2 = np.mean(psd[:,index_sel2],axis=1)
        psd_mean_comp2[iz] = np.mean(psd_comp2)
        
           
        ax0.plot(freq,psd_comp1,lt[iz],label='Group'+str(tab[iz]),linewidth=lw[iz],color='black')
        ax0.set_xlabel('Frequency[Hz]',fontsize=14)
        ax0.set_ylabel('PSD [m/s]^2/Hz',fontsize=14)
        ax0.set_title(comp_name[ikk] + ' -- u')
        ax0.set_xlim(1,100)
    
        
        ax1.plot(freq,psd_comp2,lt[iz],linewidth=lw[iz],color='black')
        ax1.set_xlabel('Frequency[Hz]',fontsize=14)
        ax1.set_ylabel('PSD [m/s]^2/Hz',fontsize=14)
        ax1.set_title(comp_name[ikk] + ' -- db')
        ax1.set_xlim(1,100)
    
    
        ax2.scatter(np.ones((len(psd_comp1_scat)))*tab[iz],psd_comp1_scat,s =70,color='black',marker='o')
        ax2.scatter(np.ones((len(psd_comp2_scat)))*tab[iz],psd_comp2_scat,s =70,color='gray',marker='^')
        ax2.set_xlabel('Group',fontsize=14)
        ax2.set_ylabel('Average PSD -- [m/s]^2/Hz ',fontsize=14)
    

    ax2.plot(tab,psd_mean_comp1,':o',label='u',color='black',linewidth=2,markersize=10)
    ax2.plot(tab,psd_mean_comp2,'--^',label='db',color='gray',linewidth=2,markersize=10)
    ax0.legend(fontsize=14)
    ax2.legend(fontsize=14,loc='upper right')
    ax0.grid()
    ax1.grid()
    ax2.grid()
    fig.tight_layout()
    
    path = r'E:\PRORAIL_CSV_FILES\VIBRATIONS\FINAL_RESULTS\DAS\PER_GROUP'
    os.chdir(path)

    plt.savefig('psd_per_group_'+sel_chan,dpi=150)

    
#%%---------------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import os
import matplotlib.pylab as plt


path = r'E:\PRORAIL_CSV_FILES\VIBRATIONS\FINAL_RESULTS\VELOCITY\DAS'
os.chdir(path)

comp_name = ['Channel39','Channel44','Channel49','Channel54','Channel59']
All_comp = ['FO40','FO45','FO50','F055','FO60']

fs = 1000
fmin = 1.0
fmax = 100.0
win_len = 1024
vel_str = 300
train_type = 's'
#------------------------------
direction1 = 'u'
direction2 = 'db'

All_segments1 = np.load('All_DAS_11112020.npy')



for ikk in range(0,len(comp_name)):

    sel_chan = All_comp[ikk]
    full_segments1 = All_segments1[:,:,ikk]
    
    df_1 = pd.read_excel(r'E:\PRORAIL_CSV_FILES\VIBRATIONS\csv_files_das\INDEX.xlsx','Trains_11112020') 
    
    plot_vel_per_group(full_segments1,df_1,train_type,win_len,vel_str,fmin,fmax)
    
    plot_psd_per_group(full_segments1,df_1,train_type,win_len,vel_str,fmin,fmax)

 
