from typing import Union, List, Dict, Any, Optional, TypeVar
import numpy as np
import pandas as pd 
import os
import re
from ipywidgets import IntProgress
from IPython.display import display

class Prediction:
    def __init__(self,path_data_hypno,times_stamps = 0.005,epoch_time = 30,verbose=1,predictionName="HYPNOGRAMS"):
        self.SCORE_DICT = {
            'Wake': 0.,
            'N1': 1.,
            'N2': 2.,
            'N3': 3.,
            'REM': 4.}
        self.path_data_hypno=path_data_hypno
        self.allHypno = os.listdir(self.path_data_hypno)
        self.allHypno.sort(key=self.FindIntInStr)
        self.allIndHypno = [i for i in range(len(self.allHypno)) 
                            if len(list(map(int, re.findall(r'\d+',self.allHypno[i]))))==0]
        if len(self.allIndHypno) > 0:
            for i in self.allIndHypno:
                del self.allHypno[i]
        self.allPart = list(range(1,len(self.allHypno)+1))
        self.vecNumToLabel = np.vectorize(self.NumToLabel)
        self.times_stamps=times_stamps
        self.epoch_time=epoch_time
        self.verbose=verbose
        self.predictionName=predictionName
        
        
    
    def Load(self,partID:int,**kwargs):
        i = partID-1
        fileHYPNO = os.path.join(self.path_data_hypno,self.allHypno[i])
        if self.verbose>0:
            print("Load Part : %s, file HYPNO: %s" % (partID,self.allHypno[i]))
        data_scorer = pd.read_csv(fileHYPNO,sep=';')
        
        return {"Data":data_scorer,"partID":partID}
        
    
    def LoadParts(self, partIDs: List[int],**kwargs):
        alldata = []
        if self.verbose>0:
            f = IntProgress(min=0, max=len(self.allHypno),description="File: ")
            display(f)
            for i in range(len(partIDs)):
                f.value += 1
                f.description="File: "+self.allHypno[partIDs[i]]
                alldata.append(self.Load(partIDs[i],**kwargs))
        else:
            for i in range(len(partIDs)):
                alldata.append(self.Load(partIDs[i],**kwargs))
        return alldata
    
    def FindIntInStr(self,my_list):
        return list(map(int, re.findall(r'\d+', my_list)))

    def NumToLabel(self,a):
        return(self.hypnoInfo[self.hypnoInfo[1]==a][0].iloc[0])
   
    def numerize_labels(self,df):
        # turn event grid into both numerized list and 1 hot encoded matrix, the list is "label"
        scorings = []
        one_hot_scorings = []
        for r in df.itertuples():
            if r.Event in SCORE_DICT:
                # dirty method to skip header
                phase = r.Event
                dur = int(r.Duration)
                epochs = dur/30
                scorings += [SCORE_DICT[phase]]*int(epochs)

        # create 1 hot encoded matrix, this is "y"
        one_hot_scorings = []
        for i in range(len(scorings)):
            to_append = np.zeros(5)
            to_append[int(scorings[i])] = 1
            one_hot_scorings.append(to_append)
        # following actions perform label = label' and y = cell2mat(y') MATLAB functions
        # making one_hot_scoring the shape of size x 1 instead of 1 x size
        one_hot_scorings = np.concatenate(np.array(one_hot_scorings)[np.newaxis])
        # same for scorings
        scorings = np.transpose(np.array(scorings)[np.newaxis])

        print("shape of scorings, should be num x 1:", np.shape(scorings))
        print("shape of one_hot_scorings, should be num x 5:", np.shape(one_hot_scorings))
        return scorings, one_hot_scorings

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

    def getallpart(self):
        return np.array(self.allPart)


    


if __name__ == "__main__":
    test = Prediction("/datasets/10x100/psg/padded_hypnograms/")
    one = test.Load(1)
    twothree = test.LoadParts([2,3])
    # preprocessed_signal = iqr_standardize(cheby2_highpass_filtfilt(resample_2(signal), 64, 0.3)) # preprocessing steps
