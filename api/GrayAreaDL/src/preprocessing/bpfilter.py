from scipy import signal
import mne
    
class BPFilter:
    def __init__(self,lowcut, highcut, order,package="scipy"):
        self.lowcut = lowcut
        self.highcut=highcut 
        self.order=order 
        self.package=package
        
    def fit(self, X,y=None):
        pass

        
    def transform(self, X):
        if self.package == "scipy":
            b = signal.firwin(self.order, [self.lowcut, self.highcut], window='hamming', pass_zero=False)
            y = signal.filtfilt(b, 1, X)
        if self.package == "mne":
            y = mne.filter.filter_data(X, data.shape[1]/self.epoch_time,lowcut,highcut,verbose=verbose)
        return y


if __name__ == "__main__":
    print("To implement")