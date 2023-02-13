import numpy as np

class IQRstd:
    def __init__(self,quantile=None):
        pass
    
    def fit(self, X,y=None):
        pass
    
    def Operation(self,x):
        q75, q25 = np.percentile(x, [75 ,25])
        self.iqr = q75 - q25
        return (x - np.median(x)) / self.iqr
    
    
    def transform(self, X):
        return np.array(list(map(lambda x : self.Operation(x),X)))

if __name__ == "__main__":
    
    X = np.random.randint(0,100,size=(10,100))
    
    PrePro = IQRstd()
    
    print(X.shape,PrePro.transform(X).shape)
    print(X,PrePro.transform(X))