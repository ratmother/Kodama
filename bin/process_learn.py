import os
import numpy as np
import librosa
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors as KNeighbor
import matplotlib.pyplot as plt

# This script contains definitions used in audio_comparative_analysis.  
# Contains functions which process audio information. Storing, machine learning and feature extraction. 

# This RingBuffer numpy class was taken from here https://scimusing.wordpress.com/2013/10/25/ring-buffers-in-pythonnumpy/

class RingBuffer():
    "A 1D ring buffer using numpy arrays"
    def __init__(self, length, itr, thresh):
        self.data = np.zeros(length, dtype='f')
        self.index = 0
        self.itr = itr # Added for audio_comparative_analysis, not from original code.
        self.thrsh = thresh # Same as above.

    def extend(self, x):
        "adds array x to ring buffer"
        x_index = (self.index + np.arange(x.size)) % self.data.size
        self.data[x_index] = x
        self.index = x_index[-1] + 1

    def get(self):
        "Returns the first-in-first-out data in the ring buffer"
        idx = (self.index + np.arange(self.data.size)) %self.data.size
        return self.data[idx]

def ringbuff_numpy_test():
    ringlen = 100000
    ringbuff = RingBuffer(ringlen)
    for i in range(40):
        ringbuff.extend(np.zeros(10000, dtype='f')) # write
        ringbuff.get() #read

# The get_pca and get_features definitions were altered specfically for audio_comparative_analysis.py, original from here https://gist.github.com/fedden/52d903bcb45777f816746f16817698a0#file-dim_reduction_notebook-ipynb
# Medium article: https://medium.com/@LeonFedden/comparative-audio-analysis-with-wavenet-mfccs-umap-t-sne-and-pca-cb8237bfce2f
# Note original is not real time, and does not include nearest neighbour or compare_pca or Chroma/zero-crossing extraction. The original usage was for displaying similar sounds on a 2D graph essentially.  

def get_pca(features , n):
    pca = PCA(n_components= n)
    transformed = pca.fit(features).transform(features)
    scaler = MinMaxScaler()
    scaler.fit(transformed)
    return scaler.transform(transformed)

def get_nearest(pca_coords, n_neigh, rad):  
    neigh = KNeighbor(n_neigh, rad)
    neigh.fit(pca_coords)
    return neigh.kneighbors([pca_coords[-1]], n_neigh, return_distance=False) # -1 is the newest addition, which in this case is real time microphone input.

def get_features(data_, dataset_, file_, sample_rate, size):
    # MFCC EXTRACTION:
            trimmed_data, _ = librosa.effects.trim(y=data_)
            mfccs = librosa.feature.mfcc(trimmed_data, 
                                     sample_rate, 
                                      n_mfcc=size)

            stddev_mfccs = np.std(mfccs, axis=1)
            mean_mfccs = np.mean(mfccs, axis=1)
            average_difference = np.zeros((size,))
            for i in range(0, len(mfccs.T) - 2, 2):
                average_difference += mfccs.T[i] - mfccs.T[i+1]
            average_difference /= (len(mfccs) // 2)   
            average_difference = np.array(average_difference)
            concat_mfcc = np.hstack((stddev_mfccs, mean_mfccs))
            concat_mfcc = np.hstack((concat_mfcc, average_difference))
    # CHROMA EXTRACTION:
            hop_length = 512
            chroma =  librosa.feature.chroma_cens(data_, sr=sample_rate, hop_length=hop_length)
            mean_chroma = np.mean(chroma, axis = 1)
    # ZERO CROSSING EXRACTION:
            zero_crossings =  librosa.zero_crossings(data_, pad = False)
            zero_crossings_sum = sum(zero_crossings)
    # TEMPO ESTIMATION
            tempo = librosa.beat.tempo(data_, sr=sample_rate)
            print (tempo)
            #dataset_ += [(file_, concat_mfcc, mean_chroma, zero_crossings_sum)] Need to redo how features are saved out to use this
            concat_feats= np.hstack((concat_mfcc, mean_chroma))
            concat_feats= np.hstack((concat_feats, zero_crossings_sum))
            dataset_ += [(file_, concat_feats)] # temporary fix

# Note: If anything changes about the content in the directory of sounds, you must run get_features again before get_saved_features.
def get_saved_features(saved_features, dataset_, file_):
            dataset_ += [(file_, saved_features)]

# Compares the input PCA coordinates with a list of nearby PCA coordinates.
def compare_pca(input, nearest, thrs):
    dist = 0
    for i in nearest: 
        dist += np.linalg.norm(input-i) #  Euclidean distance. 
    return dist

