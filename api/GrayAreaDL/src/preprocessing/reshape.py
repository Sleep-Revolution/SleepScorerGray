from scipy import signal
import numpy as np


class Reshape:
    def __init__(self, epoch_number, times_stamps=0.005, epoch_time = 30):
        self.signal_len = signal_len
        self.epoch_number = epoch_number
        self.times_stamps = times_stamps
        self.epoch_time = epoch_time

        
    def fit(self, X,y=None):
        pass
        
    def transform(self, X):
        oldshape1 = X.shape[0]
        oldshape2 = X.shape[1]
        
        oldshape1 = oldshape1/self.epoch_number/self.signal_len
        
        X.reshape(newshape,self.epoch_number*self.times_stamps*self.epoch_time)

if __name__ == "__main__":
    print("To implement")