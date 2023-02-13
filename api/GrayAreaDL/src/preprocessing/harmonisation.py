from datetime import datetime, timedelta
import numpy as np
import sys
import os
import pickle
sys.path.insert(0, os.path.abspath("/main/home/gabrielj@sleep.ru.is/GrayAreaDL/"))

############# PACKAGE FOR TEST #############
from sklearn.pipeline import Pipeline
from src.data.prediction import *
from src.data.predictors import *
from src.preprocessing.iqrstd import *
from src.preprocessing.chebyhpfilter import *
from src.preprocessing.resample import *
from src.preprocessing.bpfilter import *

class Harmonisation:
    def __init__(self,Prediction_,Predictors_, pathtoSave, windowEpoch, pipeline_prepro, clearTmpData=True):
        self.times_stamps = Predictors_.times_stamps
        self.epoch_time = Predictors_.epoch_time
        self.pathtoSave = pathtoSave
        self.windowEpoch = windowEpoch
        self.pipeline_prepro = pipeline_prepro
        self.clearTmpData = clearTmpData
        self.Prediction_ = Prediction_
        self.Predictors_ = Predictors_
        self.transfered_index = 0
        self.signalsNames = self.Predictors_.signalsNames
        self.List_NbEpoch = []
        self.MaxEpoch = 0
    
    def LoadTransformSave(self, indexPart):

        
        prediction_loaded = self.Prediction_.Load(indexPart)
        predictors_loaded_signal = self.Predictors_.Load(indexPart)
        ############# TIME HARMONISATION ################
        assert prediction_loaded["partID"] == predictors_loaded_signal.partID
        
        TimeFromStart =  predictors_loaded_signal.metadata['TimeFromStart']

        scostart = prediction_loaded["Data"].iloc[0,1]
        scoend= prediction_loaded["Data"].iloc[prediction_loaded["Data"].shape[0]-1,1]
        
        s = predictors_loaded_signal.date_measure
        edfstart = predictors_loaded_signal.metadata["edfstart"]
        edfend= predictors_loaded_signal.metadata["edfend"]
        
        TimeFromStart = np.array(list(map(lambda x:(predictors_loaded_signal.metadata['Measure date']+timedelta(seconds=x)).strftime("%H:%M:%S") ,TimeFromStart)))
        
        datetime_obj = datetime.strptime(scostart, "%H:%M:%S")
        edfstart = datetime_obj.replace(year=edfend.year, month=edfend.month, day=edfend.day)

        datetime_obj = datetime.strptime(scoend, "%H:%M:%S")
        scoend = datetime_obj.replace(year=edfend.year, month=edfend.month, day=edfend.day)
        datetime_obj = scoend+timedelta(seconds=self.epoch_time)
        
        ############################################################
        
        if (datetime_obj).strftime("%H:%M:%S") !=  edfend.strftime("%H:%M:%S"):
            scoend = scoend-timedelta(seconds=self.epoch_time)
            edfend = scoend+timedelta(seconds=self.epoch_time)
        while edfend.strftime("%H:%M:%S") not in TimeFromStart:
            scoend = scoend-timedelta(seconds=self.epoch_time)
            edfend = scoend+timedelta(seconds=self.epoch_time)

        indToStart = np.where(TimeFromStart==edfstart.strftime("%H:%M:%S"))[0][0]
        indToEnd = np.where(TimeFromStart==edfend.strftime("%H:%M:%S"))[0][0]

        TimeFromStart = TimeFromStart[indToStart:indToEnd]
        
        Nsignals = len(predictors_loaded_signal.signal_dict["Signal"])
        predictors_loaded_signal.signal_dict["Signal"] = [predictors_loaded_signal.signal_dict["Signal"][i][:,indToStart:indToEnd] for i in range(Nsignals)]

        ind = prediction_loaded["Data"]["Epoch starttime"]==edfstart.strftime("%H:%M:%S")
        indToStart = prediction_loaded["Data"]["Epoch starttime"][ind].index[0]
        
        ind = prediction_loaded["Data"]["Epoch starttime"]==scoend.strftime("%H:%M:%S")
        indToEnd = prediction_loaded["Data"]["Epoch starttime"][ind].index[0]

        prediction_loaded["Data"] = prediction_loaded["Data"].iloc[indToStart:(indToEnd+1),:]
        NbEpoch = prediction_loaded["Data"].shape[0]
        Timearray = np.arange(0,(NbEpoch)*self.epoch_time,self.times_stamps)
        
        TimeFromStart_re = TimeFromStart.reshape(NbEpoch,int(self.epoch_time/self.times_stamps))
        assert prediction_loaded["Data"]["Epoch starttime"].iloc[0] == TimeFromStart_re[0,0]
        assert prediction_loaded["Data"]["Epoch starttime"].iloc[NbEpoch-1] == TimeFromStart_re[NbEpoch-1,0]
        assert prediction_loaded["Data"]["Epoch starttime"].shape[0] == TimeFromStart_re.shape[0]
        
        predictors_loaded_signal.metadata["TimeFromStart"] = TimeFromStart
        
        ##################################################################
        # print("Harmo",predictors_loaded_signal.signal_dict["Signal"][0].shape)
        
        
        flag = 0
        for i in range(Nsignals):
            if i != 0:
                flag=1
            signal = predictors_loaded_signal.signal_dict["Signal"][i]
            signalName = predictors_loaded_signal.metadata["SignalName"][i]
            with open(os.path.join(self.pathtoSave,f'{signalName}_p{indexPart}.pickle'), 'wb') as f:
                        pickle.dump(signal, f, protocol=pickle.HIGHEST_PROTOCOL)
            if flag==0:
                with open(os.path.join(self.pathtoSave,f'{self.Prediction_.predictionName}_p{indexPart}.pickle'), 'wb') as f:
                    pickle.dump(prediction_loaded["Data"], f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return NbEpoch

    def clearTmpDataDir(self):
        listFile = os.listdir(self.pathtoSave)
        if len(listFile)==0:
                print("Empty Dir")
        else:
            for f in listFile:
                try:
                    os.remove(os.path.join(self.pathtoSave,f))
                except Exception as e: 
                        print(f"WARNING: {e}")
        
    def ListLoaTraSav(self, listPart):
        if self.clearTmpData:
            self.clearTmpDataDir()
        List_NbEpochs = []
        for i in listPart:
            List_NbEpochs.append(self.LoadTransformSave(i))
        self.MaxEpoch = min(List_NbEpochs)
        self.tmpfiles = os.listdir(self.pathtoSave)
        self.signalsNames = list(set([k for k in self.signalsNames for j in self.tmpfiles if k == j.split("_")[0]]))
        for i in listPart:
            ################################################
            print(i)
            
            filePrediction = os.path.join(self.pathtoSave,f"{self.Prediction_.predictionName}_p{i}.pickle")
            with open(filePrediction,"rb") as f:
                    prediction_loaded = pickle.load(f)
                    
            NbEpoch = prediction_loaded.shape[0]   
            col = prediction_loaded.shape[1]
            ############# SIGNAL WINDOW EPOCHS RESHAPING AND PIPELINE TRANSFORM ################
            if (self.MaxEpoch<self.windowEpoch):
                self.windowEpoch = self.MaxEpoch
                print(f"WARNING: windowEpoch > MaxEpoch = {self.MaxEpoch}, windowEpoch set to MaxEpoch")
            oldNbEpoch = NbEpoch
            if (NbEpoch % self.windowEpoch) != 0:
                EpochToRemove = NbEpoch % self.windowEpoch
                prediction_loaded = prediction_loaded.iloc[:(NbEpoch-EpochToRemove),:]
                NbEpoch = NbEpoch-EpochToRemove
            
            assert (NbEpoch % self.windowEpoch) == 0
            if (self.MaxEpoch==self.windowEpoch):
                newN=1
                prediction_loaded = prediction_loaded.iloc[:self.windowEpoch,:]
            else:
                newN = int(NbEpoch/self.windowEpoch)
            newP = int(self.windowEpoch*self.epoch_time/self.times_stamps)
            epochSize = int(self.epoch_time/self.times_stamps)
            
            ###############################################
            # print(NbEpoch,newN,prediction_loaded.shape,self.windowEpoch)
            
            prediction_loaded = prediction_loaded.to_numpy().reshape(newN,self.windowEpoch,col)
                
            # flag to save hypnograms
            flag = 0
        
        
            for k in self.signalsNames:
                fileSignal = os.path.join(self.pathtoSave,f"{k}_p{i}.pickle")
                with open(fileSignal,"rb") as f:
                    signal = pickle.load(f)
                

                if (oldNbEpoch % self.windowEpoch) != 0:
                    maxNepoch = int(NbEpoch*self.epoch_time/self.times_stamps)
                    signal = signal[:maxNepoch]
                
                if (self.MaxEpoch==self.windowEpoch):
                    signal = signal[:,:newP]
                    
                ###############################################
                # print(NbEpoch,self.windowEpoch,NbEpoch/self.windowEpoch)
                # print(newN,newP,signal.shape)
                
            
                newsignals = signal.reshape(newN,newP)
                assert prediction_loaded.shape[0] == newsignals.shape[0]

                ###############################################
                # print(newsignals.shape,newN,newP,self.windowEpoch)

                if not(self.pipeline_prepro is None):
                    newsignals = self.pipeline_prepro.transform(newsignals)

                ############# SAVE EACH INDIVDU AS SPECIFIC FILE FOR DATAGENERATOR ################
                
                for j in range(newN):
                    fileName = f'{k}_p{i}_{self.transfered_index+j}.pickle'
                    with open(os.path.join(self.pathtoSave,fileName), 'wb') as f:
                        pickle.dump(newsignals[j,:], f, protocol=pickle.HIGHEST_PROTOCOL)
                    if flag==0:
                        fileName = f'{self.Prediction_.predictionName}_p{i}_{self.transfered_index+j}.pickle'
                        with open(os.path.join(self.pathtoSave,fileName), 'wb') as f:
                            pickle.dump(prediction_loaded[j,:], f, protocol=pickle.HIGHEST_PROTOCOL)
                # print(k)
                os.remove(fileSignal)
                flag = 1
            
            os.remove(filePrediction)
            self.transfered_index += newN

        self.transfered_index=0
        
        self.files = os.listdir(self.pathtoSave)
        
        for k in range(len(self.signalsNames)):
            self.List_ID = ["_".join(i.split("_")[-2:]) for i in self.files if i.split("_")[0]==self.signalsNames[k]]
            if len(self.List_ID)>0:
                break
        self.List_ID.sort()
        print("DataSet successfully transformed & saved")
            
            
        
        


if __name__ == "__main__":
    Signals = Predictors('/datasets/10x100/psg/edf_recordings/')
    Hypno = Prediction("/datasets/10x100/psg/padded_hypnograms/")
    
    
    params = {"Preprocessing":[]}
    params["Preprocessing"] = [Resample(filtering=True),ChebyHPFilter(64, 0.3),IQRstd()]
    
    pipeline_prepro = Pipeline([(str(estimator), estimator) for estimator in params["Preprocessing"]])
    pathTosave = "/main/home/gabrielj@sleep.ru.is/GrayAreaDL/TmpData/"
    HM = Harmonisation(Hypno,Signals,pathTosave,windowEpoch=2000,pipeline_prepro=pipeline_prepro)

    #Test list of data
    HM.ListLoaTraSav(list(range(1,51)))
    print(os.listdir(pathTosave))
    
