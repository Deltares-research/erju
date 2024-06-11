# -*- coding: utf-8 -*-
"""
Created on Mon May  3 10:14:10 2021

@author: obandohe
"""


import pandas as pd
import os
import matplotlib.pylab as plt
import glob
import numpy as np


import matplotlib 
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)

comp_name = ['N05V','N05H','N09V','N09H','N24V','N24H','S09V','S09H','S24V','S24H','N13V','N13H','N13P','N18V','N18H','N18P','S13V','S13H','S13P','S18V','S18H','S18P']


width = 0.30

shift = np.array([-0.5*width,0.5*width]) 

for iiz in range(0,len(comp_name)):

    name_chan = comp_name[iiz] 
    
    color = ['red','blue']
    
    date_sel = ['09112020','11112020']
    
    path1 = r'F:\PRORAIL_CSV_FILES\VIBRATIONS\FINAL_RESULTS\ACCELEROMETERS\09112020'
    path2 = r'F:\PRORAIL_CSV_FILES\VIBRATIONS\FINAL_RESULTS\ACCELEROMETERS\11112020'
    
    path = [path1,path2]
    
    fig,ax = plt.subplots(ncols=3,nrows=2,figsize=(12,6))
    
    
    for iir in range(0,2):
    
        os.chdir(path[iir])
        Allfiles=glob.glob('*.csv')
        
    
        sel_chan = name_chan
        
        filename = 'All_mean_d.csv'
        df = pd.read_csv(filename)
        array = df[sel_chan]
        array1 = df.to_numpy()
        
        filename = 'All_std_d_error.csv'
        df = pd.read_csv(filename)
        array = df[sel_chan]
        array2 = df.to_numpy()
        
        
        psd1 = np.load('All_PSD_d_u.npy')
        freq = np.load('freq.npy')
        psd1 = psd1[:,:,iiz]
        mean_psd1 = np.mean(psd1,axis=1)
        std_psd1 = np.std(psd1,axis=1)
        psd_min1 = mean_psd1-std_psd1
        psd_max1 = mean_psd1+std_psd1
        
        
        psd2 = np.load('All_PSD_d_db.npy')
        freq = np.load('freq.npy')
        psd2 = psd2[:,:,iiz]
        mean_psd2 = np.mean(psd2,axis=1)
        std_psd2 = np.std(psd2,axis=1)
        psd_min2 = mean_psd2-std_psd2
        psd_max2 = mean_psd2+std_psd2
        
        
        #------------------------------Sprinter------------------------------
        filename = 'All_mean_s.csv'
        df = pd.read_csv(filename)
        array = df[sel_chan]
        array3 = df.to_numpy()
        
        filename = 'All_std_s_error.csv'
        df = pd.read_csv(filename)
        array = df[sel_chan]
        array4 = df.to_numpy()
        
        
        psd3 = np.load('All_PSD_s_u.npy')
        freq = np.load('freq.npy')
        psd3 = psd3[:,:,iiz]
        mean_psd3 = np.mean(psd3,axis=1)
        std_psd3 = np.std(psd3,axis=1)
        psd_min3 = mean_psd3-std_psd3
        psd_max3 = mean_psd3+std_psd3
        
        
        psd4 = np.load('All_PSD_s_db.npy')
        freq = np.load('freq.npy')
        psd4 = psd4[:,:,iiz]
        mean_psd4 = np.mean(psd4,axis=1)
        std_psd4 = np.std(psd4,axis=1)
        psd_min4 = mean_psd4-std_psd4
        psd_max4 = mean_psd4+std_psd4
        
     #---------------------------------------Plotting resdults-------------------------------------------------   
        
        ax[0,0].plot(freq,mean_psd1,color=color[iir],label=date_sel[iir])
        ax[0,0].set_xlim(1,100)
        ax[0,0].set_ylim(0,1.2e-9)
        ax[0,0].set_xlabel('Frequency [Hz]',fontsize=14)
        ax[0,0].set_ylabel('PSD [m/s]^2/Hz',fontsize=14)
        ax[0,0].set_title(name_chan + '-- d to u',fontsize=14)
        ax[0,0].grid('on')
        ax[0,0].legend()
    
    
        ax[0,1].plot(freq,mean_psd2,color=color[iir])
        ax[0,1].set_xlabel('Frequency [Hz]',fontsize=14)
        ax[0,1].set_ylabel('PSD [m/s]^2/Hz',fontsize=14)
        ax[0,1].set_title(name_chan + '-- d to db',fontsize=14)
        ax[0,1].set_xlim(1,100)
        ax[0,1].set_ylim(0,1.2e-9)
        ax[0,1].grid('on')
        
        
        dir_train = ['u','db']
        
        all_mean2 = array1[:,iiz+1]
        all_std2 = array2[:,iiz+1]
        
        x_ref = np.arange(len(dir_train))
    
    
        ax[0,2].bar(x_ref+shift[iir],all_mean2,yerr=all_std2,width=width,linewidth=2,capsize=8,color=color[iir])
        ax[0,2].set_xlabel('Travelling direction',fontsize=14)
        ax[0,2].set_ylabel('Mean PSD [m/s]^2/Hz',fontsize=14)
        ax[0,2].set_xticks(x_ref)
        ax[0,2].set_xticklabels(dir_train)
        ax[0,2].set_title('Train type -- d',fontsize=14)
        ax[0,2].grid('on')
        
        #---------------------------Sprinter------------------------------------------
    
        ax[1,0].plot(freq,mean_psd3,color=color[iir])
        ax[1,0].set_xlim(1,100)
        ax[1,0].set_ylim(0,1.2e-9)
        ax[1,0].set_xlabel('Frequency [Hz]',fontsize=14)
        ax[1,0].set_ylabel('PSD [m/s]^2/Hz',fontsize=14)
        ax[1,0].set_title(name_chan + '-- s to u',fontsize=14)
        ax[1,0].grid('on')
    
    
        ax[1,1].plot(freq,mean_psd4,color=color[iir])
        ax[1,1].set_xlabel('Frequency [Hz]',fontsize=14)
        ax[1,1].set_ylabel('PSD [m/s]^2/Hz',fontsize=14)
        ax[1,1].set_title(name_chan + '-- s to db',fontsize=14)
        ax[1,1].set_xlim(1,100)
        ax[1,1].set_ylim(0,1.2e-9)
        ax[1,1].grid('on')
     
        
        dir_train = ['u','db']
        
        all_mean4 = array3[:,iiz+1]
        all_std4 = array4[:,iiz+1]
        
        ax[1,2].bar(x_ref+shift[iir],all_mean4,yerr=all_std4,width=width,linewidth=2,capsize=8,color=color[iir])
        ax[1,2].set_xlabel('Travelling direction',fontsize=14)
        ax[1,2].set_ylabel('Mean PSD [m/s]^2/Hz',fontsize=14)
        ax[1,2].set_xticks(x_ref)
        ax[1,2].set_xticklabels(dir_train)
        ax[1,2].set_title('Train Type -- s',fontsize=14)
        ax[1,2].grid('on')
    
        fig.tight_layout()
    


    path = r'F:\PRORAIL_CSV_FILES\VIBRATIONS\FINAL_RESULTS\ACCELEROMETERS'
    os.chdir(path)

    plt.savefig(name_chan,dpi=150)
    

#%%--------------------------Amplitude attenuation----------------------------------------------

index_N = np.array([0,2,10,13,4],dtype='int') 

index_S = np.array([6,16],dtype='int') 

    
n_names = ['N05V','N09V','N13V','N18V','N24V']
    
amp1_u1 = array1[0,index_N+1]
amp1_db1 = array1[1,index_N+1]
amp1_u2 = array3[0,index_N+1]
amp1_db2 = array3[1,index_N+1]

fig,ax = plt.subplots(figsize=(8,6))

plt.subplot(211)
plt.plot(n_names,amp1_u1,'--',label='u',linewidth=2,color='black') 
plt.plot(n_names,amp1_db1,'-',label='db',linewidth=2,color='gray') 
plt.ylim(0,3.5e-11) 
plt.xlabel('Transducer component',fontsize=14)
plt.ylabel('Mean PSD [m/s]^2/Hz',fontsize=14)
plt.title('09112020 - d',fontsize=14)
plt.grid() 
plt.legend(fontsize=14)  
    
plt.subplot(212)
plt.plot(n_names,amp1_u2,'--',label='u',linewidth=2,color='black') 
plt.plot(n_names,amp1_db2,'-',label='db',linewidth=2,color='gray')  
plt.ylim(0,3.5e-11)
plt.xlabel('Transducer component',fontsize=14)
plt.ylabel('Mean PSD [m/s]^2/Hz',fontsize=14)
plt.title('09112020 - s',fontsize=14)
plt.grid() 
plt.legend(fontsize=14)  
plt.tight_layout() 












