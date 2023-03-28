from scipy import signal
import numpy as np

class ChebyHPFilter:
    def __init__(self,fs, cutoff, order=5, rs=40.0):
        self.fs = fs
        self.cutoff=cutoff 
        self.order=order 
        self.rs=rs
        
    def fit(self, X,y=None):
        pass

    def Operation(self,x):
        nyq = 0.5 * self.fs
        norm_cutoff = self.cutoff / nyq
        sos = signal.cheby2(self.order, self.rs, norm_cutoff, btype='highpass', output='sos')
        return signal.sosfiltfilt(sos, x)
    
    
    def transform(self, X):
        return np.array(list(map(lambda x : self.Operation(x),X)))
    

if __name__ == "__main__":
    X = np.random.randint(0,100,size=(10,100))
    
    PrePro = ChebyHPFilter(64, 0.3)
    
    print(X.shape,PrePro.transform(X).shape)
    print(X,PrePro.transform(X))


