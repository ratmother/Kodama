import os
import numpy as np
import librosa
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors as KNeighbor

# This script contains definitions used in audio_comparative_analysis.  
# Contains functions which process audio information. Storing, machine learning and feature extraction. 

# This RingBuffer numpy class was taken from here https://scimusing.wordpress.com/2013/10/25/ring-buffers-in-pythonnumpy/
class RingBuffer():
    "A 1D ring buffer using numpy arrays"
    def __init__(self, length, itr, cd):
        self.data = np.zeros(length, dtype='f')
        self.index = 0
        self.itr = itr # Added for audio_comparative_analysis, not from original code.
        self.cooldown = cd # Added for audio_comparative_analysis, not from original code.

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

# The get_pca, get_nearest and get_features definitions were altered specfically for audio_comparative_analysis.py, originals from here https://gist.github.com/fedden/52d903bcb45777f816746f16817698a0#file-dim_reduction_notebook-ipynb
# Medium article: https://medium.com/@LeonFedden/comparative-audio-analysis-with-wavenet-mfccs-umap-t-sne-and-pca-cb8237bfce2f
# Note original is not real time, and does not include nearest neighbour. The original usage was an audio TSNE. 
# Why PCA and MFCC over wavenet and TSNE? Its faster + I'm using MFCC's already. PCA/MFCC is fast and currently the plan for Kodama is to have the user be able to add
# their own sounds for tailored experiences, or, possible future development utilizing my own recordings. 

# Note: n_components should be tested with different values to see what works best. 
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

def get_features(data_, dataset_, file_, sample_rate, mfcc_size):
    # file_ isn't always a file name,  eg. it is called "INPUT" when the source is microphone/ input.
    # This function extracts mfcc data from the file and places it into the dataset along with the name of the file.
            trimmed_data, _ = librosa.effects.trim(y=data_)
            mfccs = librosa.feature.mfcc(trimmed_data, 
                                     sample_rate, 
                                      n_mfcc=mfcc_size)

            stddev_mfccs = np.std(mfccs, axis=1)
            mean_mfccs = np.mean(mfccs, axis=1)
            average_difference = np.zeros((mfcc_size,))
            for i in range(0, len(mfccs.T) - 2, 2):
                average_difference += mfccs.T[i] - mfccs.T[i+1]
            average_difference /= (len(mfccs) // 2)   
            average_difference = np.array(average_difference)
            concat_features = np.hstack((stddev_mfccs, mean_mfccs))
            concat_features = np.hstack((concat_features, average_difference))
            print("features saved from " + file_)
            #print(concat_features)
            dataset_ += [(file_, concat_features)]

# Note: If anything changes about the content in the directory of sounds, you must run get_features again before get_saved_features.
def get_saved_features(saved_features, dataset_, file_):
            dataset_ += [(file_, saved_features)]

# Compares the input PCA coordinates with a list of nearby PCA coordinates, tests to see if above the threshold. 
def compare_pca(input, nearest, thrs):
    dist = 0
    for i in nearest: 
        dist += np.linalg.norm(input-i)
    if dist > thrs: 
        print(dist)
        return True
    else:
        print(dist)
        return False
