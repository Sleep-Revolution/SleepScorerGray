import numpy as np
from scipy import signal

class Spectrograms:
    def __init__(self):
        pass
        
    def fit(self, X,y=None):
        pass
        
    def transform(self, X):
        megaZxx = np.empty((len(X), 29, 129))
        for i, e in enumerate(X):
            f, t, Zxx = signal.stft(e, fs=100, 
                                    window=signal.windows.general_hamming(200, 0.54), 
                                    nperseg=200, 
                                    noverlap=100, 
                                    nfft=256)
            # mlab specgram
            # f, t, Zxx = mlab.specgram(e, NFFT=256, Fs=100, noverlap=100, window=np.hamming(200), mode='complex')
            # remove first and last column of spectrogram
            t = t[1:-1]
            Zxx = np.delete(Zxx, 0, 1)
            Zxx = np.delete(Zxx, -1, 1)
            Zxx = np.log10(np.abs(Zxx))
            megaZxx[i] = np.transpose(Zxx)
        # transposed_megaZxx = np.moveaxis(megaZxx, 0, 2)
        return megaZxx

if __name__ == "__main__":
    print("To implement")