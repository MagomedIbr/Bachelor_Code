from pathlib import Path
import scipy as sp
from scipy.io import wavfile
import soundfile as sf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import rc
import scipy.io as sio
import os 
import itertools
import re
from matplotlib.patches import Polygon
import scipy.io as sio
import math
import csv
import random
import time

def speach_mode(session_length,utt_length,utt_num):
    mode_ids_temp = []
    if(session_length == 70):
        mode_temp = 'audible'
        mode_ids_temp = [mode_temp]*utt_length
    elif(session_length == 50):
        mode_temp = 'audible'
        mode_ids_temp = [mode_temp]*utt_length
    elif(session_length == 150):
        if(utt_num <= 49):
            mode_temp = 'audible' 
            mode_ids_temp = [mode_temp]*utt_length
        elif(utt_num <= 99 and utt_num>49):
            mode_temp = 'whisper' 
            mode_ids_temp = [mode_temp]*utt_length
        elif(utt_num > 99):
            mode_temp = 'silent' 
            mode_ids_temp = [mode_temp]*utt_length
    elif(session_length == 500):
        mode_temp = 'audible'
        mode_ids_temp = [mode_temp]*utt_length
    else:
        mode_temp = 'audible'
        mode_ids_temp = [mode_temp]*utt_length
    return mode_ids_temp

start_time = time.time()
emg_folder = Path('emg/002/001')
base_path = "/share/data/EMG/UKA_Full_Reconstructed_2018/emg"
temp_folder = "/share/temp/students/magomed"
# base_path = Path('emg')
p8_folder = base_path + '/' + '001'
b = os.listdir(p8_folder)
b_length = len(b)
# path_temps = str(p8_folder) + '/' + b[0]
# aa = os.listdir(path_temps)
# test = path_temps + '/' +aa[0]
# emg_test,samplerate = sf.read(test, format='RAW', subtype='PCM_16', samplerate=600,channels=6)
# print(emg_test)
# print(b)

people = sorted(os.listdir(base_path))
# print(people)
emg_data = np.empty([1,6])   
user = []
sessions = []
utt = []
mode = []
# for session in b:
    # print (session)
    # path_temp = str(p8_folder) + '/' + session
    # c = sorted(os.listdir(path_temp))
    # sess_len = len(c)
    # for file in c:
        # if ".adc" in file:
            # print(file)
            # pp_temp = path_temp + '/' + file
            # emg_temp,samplerate = sf.read(pp_temp, format='RAW', subtype='PCM_16', samplerate=600,channels=6)
            # emg_data = np.concatenate([emg_data,emg_temp],axis=0)
            # emg_length = emg_temp.shape[0]
            # user_temp = file[4:7]
            # session_temp = user_temp +file[8:11]
            # utt_temp = session_temp +file[12:16]
            # utt_temp = [utt_temp]*emg_length
            # user_temp = [user_temp]*emg_length
            # session_temp = [session_temp]*emg_length
            # mode_temp = speach_mode(sess_len, emg_length,c.index(file))
            # mode.extend(mode_temp)
            # user.extend(user_temp)
            # sessions.extend(session_temp)
            # utt.extend(utt_temp)

	

for person in people:
    print ("userID: " +str(person))
    path_temp_p = str(base_path) + '/' + person
    b = sorted(os.listdir(path_temp_p))
    for session in b:
        print("sessionID: "+ str(session))
        path_temp_s = str(path_temp_p) + '/' + session
        c = sorted(os.listdir(path_temp_s))
        sess_len = len(c)
        for file in c:
            if ".adc" in file:
                pp_temp = path_temp_s + '/' + file
                emg_temp,samplerate = sf.read(pp_temp, format='RAW', subtype='PCM_16', samplerate=600,channels=6)
                emg_data = np.concatenate([emg_data,emg_temp],axis=0)
                emg_length = emg_temp.shape[0]
                user_temp = file[4:7]
                session_temp = user_temp +file[8:11]
                utt_temp = session_temp +file[12:16]
                mode_temp = speach_mode(sess_len, emg_length,c.index(file))
                utt_temp = [utt_temp]*emg_length
                user_temp = [user_temp]*emg_length
                session_temp = [session_temp]*emg_length
                user.extend(user_temp)
                sessions.extend(session_temp)
                utt.extend(utt_temp)
                mode.extend(mode_temp)
        print(1.0-(len(user)/19630069.0))
# print (emg_data)
print ("All imports took ", time.time() - start_time, "s to run")
print(len(mode))
emg_data = np.delete(emg_data, (0), axis=0)
chanels = [1,2,3,4,5,6,7,8,9,10] 
# feature_list = []
# emg_data,samplerate = sf.read(emg_folder/'e07_002_001_0100.adc', format='RAW', subtype='PCM_16', samplerate=600,channels=6)
# emg_length = emg_data.shape[0]
# user = [1]*emg_length
# sessions = [0]*emg_length
emg_data_all = np.c_[emg_data,user]
emg_data_all = np.c_[emg_data_all,sessions]
emg_data_all = np.c_[emg_data_all,utt]
emg_data_all = np.c_[emg_data_all,mode]
print(emg_data_all.shape)
print(len(mode))
df = pd.DataFrame(emg_data_all,columns=chanels)
df.rename(columns={7:'userID'},inplace=True)
df.rename(columns={8:'sessionID'},inplace=True)
df.rename(columns={9:'uttID'},inplace=True)
df.rename(columns={10:'modeID'},inplace=True)
df[df.columns.difference(['sessionID','userID','uttID','modeID'])] = df[df.columns.difference(['sessionID','userID','uttID','modeID'])].apply(pd.to_numeric)    
print (df)
df.to_csv(temp_folder+'/'+'allEMGData_Modes.csv',index=False)

print ("My program took", time.time() - start_time, "to run")
