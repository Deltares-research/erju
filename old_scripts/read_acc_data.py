# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:00:35 2022

@author: obandohe
"""

import numpy as np
from io import  StringIO
import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
import glob

def ProRail_Viz_load_ACC_Data(ACCFiles,fs,sel_chan):
    
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
    #sel_chan = ['V1','V2','V3']

    chan_index = np.zeros((len(sel_chan)),dtype=int)
    
    for ix in range(0,len(sel_chan)):
    
        chan_name = sel_chan[ix]
        
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
        
        chan_index[ix]=index
    


    cal_fact = np.array([-203.874, -206.249, -207.018,-202.265,-203.190,-202.429,-197.336,-194.420,-198.748,-202.573,-88.917,-93.756,-94.011,-90.921,-88.480,-92.473,-203.046,-94.424,-95.007,-199.900,-203.479,-89.206])


    f = open(ACCFiles, 'r') # 'r' = read
    lines = f.readlines()
    f.close()
    
    rec_len = 30
    
    time = fs*60*rec_len
    
    data_acc=np.zeros((time,len(chan_index)))
    
    time_acc=np.zeros((time))
    
    for k in range(0,time):
        tmp=np.loadtxt(StringIO(lines[k+1]))
        time_acc[k] = tmp[0]/fs
        data_acc[k,:]=tmp[chan_index+1]

    data_acc = ((data_acc/1000)*cal_fact[chan_index]) # convert volts to g

    return(data_acc,time_acc)

#%%

path = r'G:\PRORAIL_CSV_FILES\ProRail_AccelerometerData\culemborg'
os.chdir(path)
fname = glob.glob(path)

ACCFiles = glob.glob('*.asc')
ACCFiles = ACCFiles[0]
print(ACCFiles)
sel_chan = ['tc1_v','tc4_v']
fs = 1000
data_acc,time = ProRail_Viz_load_ACC_Data(ACCFiles,fs,sel_chan)

#%%

plt.plot(time,data_acc[:,0])









