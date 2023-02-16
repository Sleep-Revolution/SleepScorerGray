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
    
def fmt(ticks):
    if all(isinstance(h, str) for h in ticks):
        return "%s"
    return ("%.d" if all(float(h).is_integer() for h in ticks) else
            "%.2f")

class Predictors:
    def __init__(self,path_edf_data,convuV=False,verbose=1,times_stamps=0.005,epoch_time=30,signalsNames=['C4-A1', 'C4-M1', 'AF3-E3E4' ,'ROC-LOC', '1-2', 'E2-AFZ'],maxHours=10):
        self.path_edf_data = path_edf_data
        self.allEdf = os.listdir(self.path_edf_data)
        self.allEdf.sort()
        self.edf_partsort = [int(s) for file in self.allEdf for s in re.findall(r'\d+', file)]
        self.convuV = convuV
        self.verbose = verbose
        self.times_stamps = times_stamps
        self.epoch_time= epoch_time
        self.signalsNames=signalsNames
        self.maxHours = maxHours
    def Load(self,partID: int,signalsNames=None,**kwargs):
        if not (signalsNames is None):
            self.signalsNames=signalsNames
            
        if self.convuV:
            TouV = 1e6
        else:
            TouV = 1
        i = partID-1
        fileEDF = os.path.join(self.path_edf_data,self.allEdf[i])
        if self.verbose>0:
            print("Load Part : %s, file EDF: %s" % (partID,self.allEdf[i]))
            
        tmpfile = [i for i in os.listdir(fileEDF) if i.split(".")[-1] == "edf"][0]
        edf = mne.io.read_raw_edf(os.path.join(fileEDF,tmpfile),verbose=self.verbose)
        
        
        
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
            
        
       
        
        if (signals[0].shape[1]/(3600/self.times_stamps))>(self.maxHours):
            print(f"WARNING: signal length of {signals[0].shape[1]/(3600/self.times_stamps)}H >{self.maxHours}H")
            cut = int(self.maxHours*(3600/self.times_stamps))
            signals = [i[:,:cut] for i in signals]
            times = times[:cut]
           
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
