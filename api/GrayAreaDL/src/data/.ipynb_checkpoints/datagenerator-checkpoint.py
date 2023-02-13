
from tensorflow import keras
import pickle
import numpy as np
import os
from src.data.predictors import *
#### Package for test

####################


class DataGenerator(keras.utils.Sequence):
    """
    Class object creating a keras generator able to load and preprocess dataset object.
    """
    def __init__(self, dataIndex, pathData, batch_size, signalNames, predictionName, Numscorer,pipeline=None ,shuffle=False, random_state=17, pred=False):
        self.dataIndex = dataIndex
        self.pathData = pathData
        self.batch_size = batch_size
        self.signalNames = signalNames
        self.predictionName = predictionName
        self.Numscorer = Numscorer
        self.shuffle = shuffle
        self.random_state = random_state
        self.files = os.listdir(self.pathData)
        self.signalNames = [k for k in self.signalNames for j in self.files if k == j.split("_")[0]]
        
        if len(self.signalNames)==0:
            raise ValueError("Signal names don't appear in pathData")
        
        for k in range(len(self.signalNames)):
            self.List_ID = ["_".join(i.split("_")[-2:]) for i in self.files if i.split("_")[0]==self.signalNames[k]]
            if len(self.List_ID)>0:
                break
        self.List_ID.sort()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataIndex) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.seed(self.random_state)
        self.indexes = np.arange(0,len(self.List_ID))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        

        flag = 0
        # Generate data
        X = []

        
        for k in self.signalNames:
            for i, ID in enumerate(indexes):
                with open(os.path.join(self.pathData,f"{k}_{self.List_ID[self.dataIndex[i]]}"),"rb") as f:
                    tmp_X = pickle.load(f)
                if k == self.signalNames[0]:
                    with open(os.path.join(self.pathData,f"{self.predictionName}_{self.List_ID[self.dataIndex[i]]}"),"rb") as f:
                        tmp_Y = pickle.load(f)

                if flag==0:
                    X_epochssignal = tmp_X.shape[0]
                    win_epochs = tmp_Y.shape[0]
                    # Initialization
                    X_ = np.empty((self.batch_size,X_epochssignal,))
                    Y = np.empty((self.batch_size,win_epochs), dtype=int)
                    flag=1
                    
                # Store class
                
                X_[i,:] = tmp_X
                Y[i,:] = tmp_Y[:,self.Numscorer+1]
                
            X.append(X_)
        self.n_classes = len(set(Y[:,0]))
        return X, Y
    
class DataGeneratorPred(keras.utils.Sequence):
    """
    Class object creating a keras generator able to load and preprocess dataset object.
    """
    def __init__(self, pathEDF, signalNames,pipeline=None,shuffle=False, random_state=17):
        self.pathEDF = pathEDF
        self.signalNames = signalNames
        self.pipeline = pipeline
        self.shuffle = shuffle
        self.random_state = random_state
        self.Predictors_ = Predictors(pathEDF,signalsNames=signalNames)
        self.list_id = self.Predictors_.getallpart()
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_id)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.seed(self.random_state)
        self.indexes = np.arange(0,len(self.list_id))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        X = self.Predictors_.Load(index).signal_dict["Signal"]
        if not self.pipeline is None:
            print(X[0].shape)
            X = [self.pipeline.transform(x) for x in X]
            print(X[0].shape)
        return X

if __name__ == "__main__":
    pathToData = "/main/home/gabrielj@sleep.ru.is/GrayAreaDL/TmpData/"
    signalNames = ['C4-A1', 'C4-M1', 'AF3-E3E4' ,'ROC-LOC', '1-2', 'E2-AFZ']
    batch_size = 32
    predictionName = "HYPNOGRAMS"
    Numscorer = 1
    DG = DataGenerator(pathToData, batch_size, signalNames, predictionName, Numscorer, shuffle=False, random_state=17)
    
    print(DG.__getitem__(0))