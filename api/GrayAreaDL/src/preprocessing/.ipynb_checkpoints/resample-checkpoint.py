from scipy import signal
import numpy as np 

class Resample:
    def __init__(self, prev_hz=200, next_hz=100,filtering=False,up=8, down=25):
        self.prev_hz = prev_hz
        self.next_hz = next_hz
        self.up=up
        self.down=down
        self.filtering=filtering
        
    def fit(self, X,y=None):
        pass
    
    def Operation(self,x):
        if self.filtering:
            """ Resampling by upsampling, filtering then downsampling from prev_hz Hz to prev_hz*up/down hz."""
            new_X = signal.resample_poly(x, self.up, self.down)
        else:
            """ Traditionnal resample signal from prev_hz Hz to prev_hz hz."""
            new_X = signal.resample(x, int(len(x)/self.prev_hz*self.next_hz))
        return new_X
    
    
    def transform(self, X):
        return np.array(list(map(lambda x : self.Operation(x),X)))
    

if __name__ == "__main__":
    X = np.random.randint(0,100,size=(10,6000))
    
    PrePro = Resample(filtering=True)
    
    print(X.shape,PrePro.transform(X).shape)
    print(X,PrePro.transform(X))