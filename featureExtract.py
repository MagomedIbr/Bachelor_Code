from pathlib import Path
import scipy as sp
from scipy.io import wavfile
import soundfile as sf
import time


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import rc
from scipy import signal
import scipy.io as sio
import os 
import itertools
import re
from matplotlib.patches import Polygon
import scipy.io as sio
import math
import csv
import random
from scipy.fftpack import fftn, ifftn, fft, ifft

start_time = time.time()
temp_folder = "/share/temp/students/magomed"
df = pd.read_csv(temp_folder+'/'+'allEMGData_Modes_no_Audbile.csv')
# df = pd.read_csv(temp_folder+'/'+'allEmgData.csv')
# df = pd.read_csv("p1Sessions.csv")
# df[df.columns.difference(['sessionID','userID'])] = df[df.columns.difference(['movementID','userID'])].apply(pd.to_numeric)    
print(df)

#WL 
def calc_waveform_length(arr):
    if len(arr) == 1:
        return abs(array.iloc[0])
    else:
        wl = 0.0
        for index in range(len(arr)-1):
            wl += abs(arr.iloc[index+1]-arr.iloc[index])
        return wl


#Integral of the EMG:
def calc_iemg(arr):
    if len(arr) == 0:
        return 0
    else:
        iemg = 0.0
        for index in range (len(arr)):
            iemg += abs(arr.iloc[index])
        return iemg
    
# Zero Crossings
def calc_zc(arr):
    if len(arr) < 2:
        return 0
    else:
        zc = 0
        for index in range(len(arr)-1):
            if ((arr.iloc[index+1]>0 and arr.iloc[index]<0) or (arr.iloc[index+1]<0 and arr.iloc[index]>0)):
                zc += 1
        return zc

# Slope Sign Change
def calc_ssc(arr):
    if len(arr) < 3:
        return 0
    else:
        ssc = 0
        for index in range(len(arr)-2):
            if ((arr.iloc[index+1]>arr.iloc[index] and arr.iloc[index+1]>arr.iloc[index+2]) or (arr.iloc[index+1]<arr.iloc[index] and arr.iloc[index+1]<arr.iloc[index+2])):
                ssc += 1
        return ssc

#number of times that the signal amplitude exceeds a predefined threshold
def calc_wamp(arr):
    if len(arr) == 0:
        return 0
    else:
        t_wamp = 0.3
        wamp=0
        for index in range(len(arr)-1):
            if((arr.iloc[index]-arr.iloc[index+1]) > t_wamp):
                wamp += 1
            else:
                wamp += 0
        return wamp
        
	
def calc_medf(freqs,df_fft_pos):
    frequency_strength_sum = np.sum(df_fft_pos)
    frequency_strength_sum_half = frequency_strength_sum/2.0
    frequency_strength_add = 0.0
    for index in range(len(df_fft_pos)):
        frequency_strength_add += df_fft_pos[index]
        if frequency_strength_add >= frequency_strength_sum_half:
            return freqs[index]

def calc_meanf(freqs,df_fft_pos):
    frequency_strength_sum = np.sum(df_fft_pos)
    weighted_frequency_sum = 0.0
    for index in range(len(df_fft_pos)):
        weighted_frequency_sum += freqs[index]*df_fft_pos[index]
    return weighted_frequency_sum/frequency_strength_sum
    
def construct_new_features(window_df):
    column_names = []
    rows = []
    new_features_vals = []
    for index in range(0,6):
        # print(index)
        channel_str = 'channel_'+str(index+1)
        column_vals = window_df.iloc[:,index]
        column_cals_len = len(column_vals)
        ### Time Domain Features
        # MAV (Mean Absolute Value)
        mav_str = channel_str + "_mav"
        column_names.append(mav_str)
        new_features_vals.append((np.abs(column_vals)).mean())
        # RMS (Root Mean Square)
        rms_str = channel_str + "_rms"
        column_names.append(rms_str)
        new_features_vals.append(math.sqrt(sum(np.square(column_vals))/len(column_vals)))
        # IAV
        iav_str = channel_str + "_iav"
        column_names.append(iav_str)
        new_features_vals.append(sum(np.abs(column_vals)))
	# VAR
        var_str = channel_str + "_var"
        column_names.append(var_str)
        new_features_vals.append(np.var(column_vals))
        # SSI (Simple square integral)
        ssi_str = channel_str + "_ssi"
        column_names.append(ssi_str)
        new_features_vals.append(sum(np.square(column_vals)))
        # Variance
        var_str = channel_str + "_var"
        column_names.append(var_str)
        new_features_vals.append(np.var(column_vals))
        # Waveform Length (WL)
        wl_str = channel_str + "_wl"
        column_names.append(wl_str)
        new_features_vals.append(calc_waveform_length(column_vals))
	# Waveform Length (WL)
        iemg_str = channel_str + "_iemg"
        column_names.append(iemg_str)
        new_features_vals.append(calc_iemg(column_vals))
        # Average Amplitude Change
        aac_str = channel_str + "_aac"
        column_names.append(aac_str)
        new_features_vals.append(calc_waveform_length(column_vals)/(len(column_vals)))
        # Zero Crossings
        zc_str = channel_str + "_zc"
        column_names.append(zc_str)
        new_features_vals.append(calc_zc(column_vals))
        # Slope Sign Change
        ssc_str = channel_str + "_ssc"
        column_names.append(ssc_str)
        new_features_vals.append(calc_ssc(column_vals))
	# Slope Sign Change
        wamp_str = channel_str + "_wamp"
        column_names.append(wamp_str)
        new_features_vals.append(calc_wamp(column_vals))
        
        ### Frequency Domain Features
        N = 0.2*column_cals_len # array size
        T = 1.0/600 # inverse of the sampling rate
        freqs = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
        df_fft = fft(column_vals)
        df_fft_pos = (2/N) * np.abs(df_fft[0:np.int(N/2)]) # positive freqs only
        # Median Frequency
        medf_str = channel_str + "_medf"
        column_names.append(medf_str)
        new_features_vals.append(calc_medf(freqs,df_fft_pos))
        # Weighted Mean Frequency
        meanf_str = channel_str + "_meanf"
        column_names.append(meanf_str)
        new_features_vals.append(calc_meanf(freqs,df_fft_pos))
        
    rows.append(new_features_vals)
    window_new_features_df = pd.DataFrame(data = rows, columns = column_names)
    return window_new_features_df
    
def extractFeatures():
    df_fe = pd.DataFrame()
    user_ids = np.unique(df.userID)
    counter = 1
    for user_id in user_ids:
        print(user_id)
        user_df = df[(df.userID == user_id)]
        # print(phase1_df)
        # df['UserID'] = user 
        session_ids = np.unique(user_df.sessionID)
        sess_length = len(session_ids)
        for s_id in session_ids:
            print("feature extracting for session: "+ str(s_id))
            session_df = user_df[(user_df.sessionID == s_id)]
            mode_ids = np.unique(session_df.modeID)	    
            print ("Program running for", time.time() - start_time, "s")
            for mode_id in mode_ids:
                mode_df = session_df[(session_df.modeID == mode_id)]
                utt_ids = np.unique(mode_df.uttID)
                for utt_id in utt_ids:
                        # print ("feature extracting for utterance: "+ str(utt_id))
                        utterance_df = mode_df[(mode_df.uttID == utt_id)]
                        utterance_df_len = len(utterance_df.index)
                        res = pd.DataFrame()  
                        window_df = utterance_df.iloc[0:(utterance_df_len),:]
                        window_new_features_df = construct_new_features(window_df)
                        window_new_features_df['userID'] = user_id
                        window_new_features_df['sessionID'] = s_id
                        window_new_features_df['uttID'] = utt_id
                        window_new_features_df['modeID'] = mode_id
                        res = res.append(window_new_features_df)
                        # sliding window of size 1200 and with 87.5% overlap
                        # for pos in range(0,len(phase1_session_df.index),150):
                        # print(pos)
                            # if (pos+1200) >= len(phase1_session_df.index):
                                # window_df = phase1_session_df.iloc[pos:,:]
                            # else:
                                # window_df = phase1_session_df.iloc[pos:(pos+1200),:]
                                # window_new_features_df = construct_new_features(window_df)
                                # window_new_features_df['sessionID'] = s_id
                                # window_new_features_df['userID'] = user_id
                                # window_new_features_df['concID'] = c_id
                                # res = res.append(window_new_features_df)
                        df_fe = df_fe.append(res)
                        print("Utterance number: " + str(counter) +"  utt_id: " + str(utt_id))
                        counter = counter+1
    print("Done!")
    return df_fe

#currently 3s per utterance
df_fe = extractFeatures()
print(df_fe)
df_fe.to_csv('Features_all_no_audible.csv',index=False)
print ("featureExtract took", time.time() - start_time, "s to run")