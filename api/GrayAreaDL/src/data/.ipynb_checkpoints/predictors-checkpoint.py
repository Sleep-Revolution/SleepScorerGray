import time
import os
import pandas as pd
import numpy as np
import mne
import re
import pickle
from datetime import datetime, timedelta
from ipywidgets import IntProgress
from IPython.display import display
import sys
# sys.path.insert(0, os.path.abspath("/main/home/gabrielj@sleep.ru.is/GrayAreaDL/"))
from src.data.partsignal import *
from typing import Union, List, Dict, Any, Optional, TypeVar
from src.preprocessing.resample import *
import glob

def fmt(ticks):
    if all(isinstance(h, str) for h in ticks):
        return "%s"
    return ("%.d" if all(float(h).is_integer() for h in ticks) else
            "%.2f")

class Predictors:
    def __init__(self,path_edf_data,convuV=False,verbose=1,times_stamps=0.005,epoch_time=30,signalsNames=['C4-A1'],maxHours=10,type_study="PSG"):
        self.path_edf_data = path_edf_data
        # self.allEdf = os.listdir(self.path_edf_data)
        # self.allEdf.sort()
        # self.edf_partsort = [int(s) for file in self.allEdf for s in re.findall(r'\d+', file)]
        self.convuV = convuV
        self.verbose = verbose
        self.times_stamps = times_stamps
        self.epoch_time= epoch_time
        self.signalsNames=signalsNames
        self.maxHours = maxHours
        self.type_study = type_study
        
        
        list_dirs = [[root,file] for root, dirs, files in os.walk(self.path_edf_data) for file in files if file.endswith(".edf")]
        self.filename= [i[1] for i in list_dirs]
        self.filedir= [i[0] for i in list_dirs]
        self.filedirsplit = [i.split("/")[-1] for i in self.filedir]
        if (len(list(set(self.filename))) != len(self.filename))&(len(list(set(self.filedir))) == len(self.filedir))&(len(set(self.filedirsplit)) == len(self.filedir)):
            self.allEdf = self.filedirsplit
        else:
            if (len(list(set(self.filename))) == len(self.filename)):
                self.allEdf = [i.split(".")[0] for i in self.filename]
            else:
                self.allEdf = [self.type_study+str(i) for i in range(len(self.filedir))]


        self.exclude = [['1','1 Impedance','1-2','1-F','2','2 Impedance','2-F','Abdomen CaL','Abdomen Fast','Abdomen','Activity','Light','Audio Volume','Audio Volume dB','C3 Impedance','C4 Impedance','cRIP Flow','cRIP Sum','E1 Impedance','E2 Impedance','ECG','ECG Impedance','EDA','Elevation','F Impedance','F3 Impedance','F4 Impedance','Flow','Flow Limitation','Heart Rate','Inductance Abdom','Inductance Thora','K','Left Leg','Left Leg Impedan','M1 Impedance','M1M2','M2 Impedance','Nasal Pressure','O1 Impedance','O2 Impedance','Pulse Waveform','PosAngle','PTT','Pulse','PWA','Resp Rate','Right Leg','Right Leg Impeda','RIP Flow','RIP Phase','RIP Sum','Snore','Saturation','SpO2 B-B','Thorax Fast','Chest','Voltage (battery','Voltage bluetoo','Voltage (core)','X Axis','Y Axis','Z Axis'], 
                ['Ambient Light A1', 'EKG Impedance', 'Pulse Wave (Plet', 'Set Pressure', 'Position', 'SpO2', 'Mask Pressure', 'PTT', 'Thorax', 'Heart Rate-0', 'Heart Rate-1','Abdomen CaL', 'Abdomen Fast', 'Abdomen', 'Activity', 'AF3 Impedance', 'AF4 Impedance', 'AF7 Impedance',  'AF8 Impedance', 'AFZ Impedance', 'Light', 'Audio', 'Audio Volume', 'Audio Volume dB', 'cRIP Flow', 'cRIP Sum', 'E1 Impedance', 'E1-E4 (Imp)','E2 Impedance', 'E2-AFZ (Imp)', 'E2-E3 (Imp)', 'E3 Impedance', 'E3-AFZ (Imp)', 'E4 Impedance', 'ECG','ECG Impedance', 'ECG LA', 'ECG LA Impedance', 'ECG LF', 'ECG LF Impedance', 'ECG RA', 'ECG RA Impedance', 'Elevation', 'EMG.Frontalis-Le', 'EMG.Frontalis-Ri', 'Flow','Flow Limitation', 'Inductance Abdom', 'Inductance Thora', 'K', 'LA-RA', 'Left Leg', 'Left Leg Impedan', 'LF-LA', 'LF-RA', 'Nasal Pressure', 'Pulse Waveform', 'PosAngle', 'Pulse','PWA', 'Resp Rate', 'Right Leg', 'Right Leg Impeda', 'RIP Flow', 'RIP Phase', 'RIP Sum', 'Snore', 'Saturation', 'SpO2 B-B', 'Thorax Fast', 'Chest', 'Voltage (battery', 'Voltage (bluetoo','Voltage (core)', 'X Axis', 'Y Axis', 'Z Axis']]

        self.channel_names_sas = np.array(['E1', 'E3', 'E2', 'E4', 'AF3', 'AF4', 'AF7', 'AF8', 'AFZ'])
        self.channel_category_sas = np.array(['eog', 'eog', 'eog', 'eog', 'eeg', 'eeg', 'eeg' ,'eeg', 'eog'])
        self.ref_channels_sas = [['E3','E4'],['eeg']]
        self.rename_sas = {'AF4' : 'AF4-E3E4','AF3' : 'AF3-E3E4','AF7' : 'AF7-E3E4','AF8' : 'AF8-E3E4'}
        self.anode_sas = np.array(['E3', 'E2', 'E1', 'E2'])
        self.cathode_sas = np.array(['AFZ', 'AFZ','E4', 'E3'])
        self.rename_sas = np.array(['E1-E4', 'E2-E3', 'E3-AFZ', 'E2-AFZ'])
        self.rename_category_sas = np.array(['eog','eog','eog','eog'])


        self.channel_names_psg = np.array(['F4', 'F3', 'C4', 'C3', 'O1', 'O2', 'E1', 'E2',"M1","M2"])
        self.channel_category_psg = np.array(['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg' ,'eog', 'eog','eeg','eeg'])
        self.anode_psg = np.array(['F4', 'F3', 'C4', 'C3', 'O1', 'O2',"E1","E2"])
        self.cathode_psg = np.array(["M1","M2","M1","M2","M2","M1","M2","M1"])
        self.rename_psg = np.array(['F4-M1','F3-M2','C4-M1','C3-M2','O1-M2','O2-M1','E1-M2','E2-M1'])
        self.rename_category_psg =  np.array(['eeg','eeg','eeg','eeg','eeg','eog','eog'])

        
        
        
    def Load(self,partID: int,signalsNames=None,**kwargs):
        if not (signalsNames is None):
            self.signalsNames=signalsNames
            
        if self.convuV:
            TouV = 1e6
        else:
            TouV = 1
        i = partID-1
        fileEDF = os.path.join(self.filedir[i],self.filename[i])
        if self.verbose>0:
            print("Load Part : %s, input name: %s, ouput name: %s" % (partID,self.filename[i],self.allEdf[i]))

        tmpfile = self.filename[i]
        
        if self.type_study == "SAS":
            channel_names = self.channel_names_sas
            channel_category = self.channel_category_sas
            exclude = self.exclude[1]
            ref_channels = self.ref_channels_sas
            rename = self.rename_sas
            anode = self.anode_sas
            cathode = self.cathode_sas
        else:
            channel_names = self.channel_names_psg
            channel_category = self.channel_category_psg
            exclude = self.exclude[0]
            rename = self.rename_psg
            anode = self.anode_psg
            cathode = self.cathode_psg
            
        raw = mne.io.read_raw_edf(fileEDF,verbose=self.verbose,exclude=exclude)
         #channels to use in re-referencing (deriving) the conventional SAS channels

        signalsName_tmp = []
        anode_tmp = []
        catode_tmp = []
        for i in self.signalsNames:
            i = i.split("-")
            assert len(i) > 1 , "Signal names need to be derivations"
            anode_tmp.append(i[0])
            catode_tmp.append(i[1])
            signalsName_tmp = signalsName_tmp+[i[0]]+[i[1]]

        if "E3E4" in catode_tmp:
            signalsName_tmp = signalsName_tmp+["E3"]+["E4"]
        
        ind = [i for i in range(len(channel_names)) if channel_names[i] in signalsName_tmp]
        
        edf=raw.pick_channels(channel_names[ind].tolist())
        edf.set_channel_types(dict(zip(channel_names[ind], channel_category[ind])))
        edf.load_data(verbose=True)

        if "E3E4" in catode_tmp:
            subchannels = channel_names[ind]
            rename_tmp = {i:i+"-E3E4" for i in subchannels[channel_category=="eeg"]}
            edf.set_eeg_reference(ref_channels=ref_channels[0], ch_type=ref_channels[1])
            edf.rename_channels(rename_tmp)

        ind = [i for i in range(len(anode)) if (anode[i] in anode_tmp)&(cathode[i] in catode_tmp)]
        if len(ind)>0:
            edf = mne.set_bipolar_reference(edf, anode=anode[ind].tolist(), cathode=cathode[ind].tolist())

        ind = [i for i in range(len(rename)) if rename[i] in anode_tmp]
        if len(ind)>0:
            edf.set_channel_types(dict(zip(rename[ind], rename_category[ind])))

        indcha = [i for i in range(len(edf.ch_names)) if edf.ch_names[i] in self.signalsNames]
        signals = []
        
        for i in range(len(indcha)):
            if i==0:
                signals_tmp, times = edf[edf.ch_names[indcha[i]]]
            else:
                signals_tmp = edf[edf.ch_names[indcha[i]]][0]
            
            signals.append(signals_tmp)
        
        # print(edf.info['meas_date']+timedelta(seconds=times[0]),edf.info['meas_date']+timedelta(seconds=times[-1]))
        ########### Sanity checks #############
        if self.times_stamps != times[1]:
            print(f"WARNING: data resampled from {1/times[1]} to {1/self.times_stamps} HZ")
            rate = (1/times[1])/(1/self.times_stamps)
            resamp = Resample(prev_hz=1/times[1], next_hz=1/self.times_stamps,filtering=True,up=1, down=rate)
            signals = [resamp.transform(i) for i in signals]
            times = (resamp.transform(times.reshape(1,len(times)))[0,:]).round(3)
            assert times[1]==self.times_stamps
            
        
       
        
        # if (signals[0].shape[1]/(3600/self.times_stamps))>(self.maxHours):
        #     print(f"WARNING: signal length of {signals[0].shape[1]/(3600/self.times_stamps)}H >{self.maxHours}H")
        #     cut = int(self.maxHours*(3600/self.times_stamps))
        #     signals = [i[:,:cut] for i in signals]
        #     times = times[:cut]
           
        # print(edf.info['meas_date']+timedelta(seconds=times[0]),edf.info['meas_date']+timedelta(seconds=times[-1]))
        today = datetime.today()
        s = edf.info['meas_date'].strftime("%H:%M:%S")
        edfstart = datetime.combine(edf.info['meas_date'], datetime.strptime(s, '%H:%M:%S').time())
        edfend= edfstart+timedelta(seconds=signals[0].shape[1]*self.times_stamps)
        
        ############################################################
        metadata = {}
        metadata["edfend"] = edfend
        metadata["edfstart"] = edfstart
        metadata["Measure date"] = edf.info['meas_date']
        metadata["TimeFromStart"] = times
        metadata["FileName"] = self.allEdf[i]
        metadata["FilePath"] = self.filedir[i]
        metadata["SignalName"] = np.array(edf.ch_names)[np.array(indcha)]
        
        _loaded_signal = PartSignal({"Signal": signals}, partID, s, metadata)
        
        return _loaded_signal
        
    
    def LoadSignals(self, partIDs: List[int], signalsNames=None):
        return [self.Load(partID,signalsNames) for partID in partIDs]


    def FindIntInStr(self,my_list):
        return list(map(int, re.findall(r'\d+', my_list)))
    
    def getallpart(self):
        return np.arange(1,len(self.allEdf)+1)

   
    def saveData(self,name):
        if os.path.isdir(os.path.join('save_data')):
            path_save = os.path.join('save_data')
        else:
            path_save = os.mkdir('save_data')
            path_save = os.path.join('save_data')
        
        path_name = os.path.join(path_save,name+'.pickle')
        i=0
        while os.path.exists(path_name):
            path_name = os.path.join(path_save,name+str(i)+'.pickle')
            i+=1
        with open(path_name, 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def loadDataPickle(self,path_name):
        with open(path_name, 'rb') as file:
            self.data = pickle.load(file)

    


if __name__ == "__main__":
    test = Predictors('/datasets/10x100/psg/edf_recordings/')
    one = test.Load(1)
    twothree = test.LoadSignals([2,3])
    # preprocessed_signal = iqr_standardize(cheby2_highpass_filtfilt(resample_2(signal), 64, 0.3)) # preprocessing steps
