
from tensorflow import keras
import pickle
import numpy as np
import os
from src.data.predictors import *
import scipy.signal as scipy_signal
#### Package & Functions for test
def resample_2(s):
    """resample signal from fs_orig to fs_after hz."""
    return scipy_signal.resample_poly(s, 8, 25)

def iqr_standardize(s):
    """IQR standardization for the signal."""
    q75, q25 = np.percentile(s, [75 ,25])
    iqr = q75 - q25
    return (s - np.median(s)) / iqr

def cheby2_highpass_filtfilt(s, fs, cutoff, order=5, rs=40.0):
    """Chebyshev type1 highpass filtering.
    
    Args:
        s: the signal
        fs: sampling freq in Hz
        cutoff: cutoff freq in Hz
    Returns:
        the filtered signal
    """
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    sos = scipy_signal.cheby2(order, rs, norm_cutoff, btype='highpass', output='sos')
    return scipy_signal.sosfiltfilt(sos, s)
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
    def __init__(self, pathEDF, signalNames,pipeline=None,shuffle=False, random_state=17,ensemble=False,type_study="PSG"):
        self.pathEDF = pathEDF
        self.signalNames = signalNames
        self.pipeline = pipeline
        self.shuffle = shuffle
        self.random_state = random_state
        self.Predictors_ = Predictors(pathEDF,signalsNames=signalNames,type_study=type_study)
        self.list_id = self.Predictors_.getallpart()
        # self.list_id = [1,5,7]
        self.ensemble=ensemble
        # self.list_id = [1]
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_id)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.seed(self.random_state)
        self.indexes = np.arange(1,len(self.list_id)+1)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        self.currentSignal = self.Predictors_.Load(index)
        X = self.currentSignal.signal_dict["Signal"]
        if not self.pipeline is None:
            X = [self.pipeline.transform(x) for x in X]
        # if self.ensemble:
        X = np.array(X)
        X = np.swapaxes(X,2,1)
        # print("Before predict:", X.shape,(X.shape[1]*(1/64))/self.Predictors_.epoch_time)
        return X

#################################      ONLY FOR VALIDATION          ###########################################################
# class MatiasGeneratorPred(keras.utils.Sequence):
#     """
#     Class object creating a keras generator able to load and preprocess dataset object.
#     """
#     def __init__(self, pathEDF,shuffle=False, random_state=17):
#         self.pathEDF = pathEDF
#         self.shuffle = shuffle
#         self.random_state = random_state
#         self.Predictors_ = Predictors(pathEDF)
#         self.list_id = self.Predictors_.getallpart()
#         # self.list_id = [1,5,7]

#         # self.list_id = [1]
#         self.on_epoch_end()

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return len(self.list_id)

#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         np.random.seed(self.random_state)
#         self.indexes = np.arange(1,len(self.list_id)+1)
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)

#     def __getitem__(self, index):
#         exclude = ['Abdomen CaL', 'Abdomen Fast', 'Abdomen', 'Activity', 'AF3 Impedance', 'AF4 Impedance', 'AF7 Impedance',  'AF8 Impedance', 'AFZ Impedance', 'Light', 'Audio', 'Audio Volume', 'Audio Volume dB', 'cRIP Flow', 'cRIP Sum', 'E1 Impedance', 'E1-E4 (Imp)','E2 Impedance', 'E2-AFZ (Imp)', 'E2-E3 (Imp)', 'E3 Impedance', 'E3-AFZ (Imp)', 'E4 Impedance', 'ECG','ECG Impedance', 'ECG LA', 'ECG LA Impedance', 'ECG LF', 'ECG LF Impedance', 'ECG RA', 'ECG RA Impedance', 'Elevation', 'EMG.Frontalis-Le', 'EMG.Frontalis-Ri', 'Flow','Flow Limitation', 'Inductance Abdom', 'Inductance Thora', 'K', 'LA-RA', 'Left Leg', 'Left Leg Impedan', 'LF-LA', 'LF-RA', 'Nasal Pressure', 'Pulse Waveform', 'PosAngle', 'Pulse','PWA', 'Resp Rate', 'Right Leg', 'Right Leg Impeda', 'RIP Flow', 'RIP Phase', 'RIP Sum', 'Snore', 'Saturation', 'SpO2 B-B', 'Thorax Fast', 'Chest', 'Voltage (battery', 'Voltage (bluetoo','Voltage (core)', 'X Axis', 'Y Axis', 'Z Axis']
#         data_path = os.path.join(self.Predictors_.path_edf_data,self.Predictors_.allEdf[index-1])
#         tmpfile = [i for i in os.listdir(data_path) if i.split(".")[-1] == "edf"][0]
#         raw = mne.io.read_raw_edf(os.path.join(data_path,tmpfile), exclude=exclude)
#         channel_names_sas = ['E1', 'E3', 'E2', 'E4', 'AF3', 'AF4', 'AF7', 'AF8', 'AFZ'] #channels to use in re-referencing (deriving) the conventional SAS channels
#         raw_sas=raw.pick_channels(channel_names_sas)
#         raw_sas.set_channel_types(dict(zip(channel_names_sas, ['eog', 'eog', 'eog', 'eog', 'eeg', 'eeg', 'eeg' ,'eeg', 'eog'])))
#         raw_sas.load_data(verbose=True)
#         raw_sas.set_eeg_reference(ref_channels=['E3','E4'], ch_type='eeg')
#         raw_sas.rename_channels({'AF4' : 'AF4-E3E4','AF3' : 'AF3-E3E4','AF7' : 'AF7-E3E4','AF8' : 'AF8-E3E4'})
#         raw_sas_der = mne.set_bipolar_reference(raw_sas, anode=['E3', 'E2', 'E1', 'E2'], cathode=['AFZ', 'AFZ','E4', 'E3'])
#         raw_sas_der.set_channel_types(dict(zip(['E1-E4', 'E2-E3', 'E3-AFZ', 'E2-AFZ'], ['eeg','eeg','eeg','eeg'])))

#         sas_signals=raw_sas_der.get_data()
#         if np.shape(sas_signals[0])[0]%(200*30)!=0:
#             mod = np.shape(sas_signals[0])[0]%(200*30)
#             sas_signals = sas_signals[:,:-mod]
#         for i in range(0,8):
#             if i == 0:
#                 X = iqr_standardize(resample_2(cheby2_highpass_filtfilt(sas_signals[i,:], 200, 0.3)))
#                 # X = iqr_standardize(cheby2_highpass_filtfilt(resample_2(sas_signals[i,:]), 64, 0.3))
#                 X = X[np.newaxis, ..., np.newaxis]
#             else:
#                 X_tmp = iqr_standardize(resample_2(cheby2_highpass_filtfilt(sas_signals[i,:], 200, 0.3))) 
#                 # X_tmp = iqr_standardize(cheby2_highpass_filtfilt(resample_2(sas_signals[i,:]), 64, 0.3))
#                 X_tmp = X_tmp[np.newaxis, ..., np.newaxis]
#                 X = np.concatenate((X,X_tmp))
#         return X
#     #################################################################################################################################################################################




if __name__ == "__main__":
    pathToData = "/main/home/gabrielj@sleep.ru.is/GrayAreaDL/TmpData/"
    signalNames = ['C4-A1', 'C4-M1', 'AF3-E3E4' ,'ROC-LOC', '1-2', 'E2-AFZ']
    batch_size = 32
    predictionName = "HYPNOGRAMS"
    Numscorer = 1
    DG = DataGenerator(pathToData, batch_size, signalNames, predictionName, Numscorer, shuffle=False, random_state=17)
    
    print(DG.__getitem__(0))

# print('Preprocessing signals...')
# eeg_signal = iqr_standardize(cheby2_highpass_filtfilt(resample_2(s_eeg), 64, 0.3)) # preprocessing steps
# og_signal = iqr_standardize(cheby2_highpass_filtfilt(resample_2(s_eog), 64, 0.3)) # preprocessing steps