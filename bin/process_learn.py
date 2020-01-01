import os
import numpy as np
import librosa
import math
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors as KNeighbor
import matplotlib.pyplot as plt

mfcc_size = 24
sample_rate = 44100

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

# The get_pca and get_features definitions were originally from here https://gist.github.com/fedden/52d903bcb45777f816746f16817698a0#file-dim_reduction_notebook-ipynb
# Medium article: https://medium.com/@LeonFedden/comparative-audio-analysis-with-wavenet-mfccs-umap-t-sne-and-pca-cb8237bfce2f
# NOTE original is NOT real time, NOT segmented and does not includget_pcae nearest neighbour or compare_pca or Chroma/zero-crossing/etc extraction (just mfcc/wavenet).  

def get_pca(features , n):
   # print(features.size)
    pca = PCA(n_components= n)
    transformed = pca.fit(features).transform(features)
    scaler = MinMaxScaler()
    scaler.fit(transformed)
    return scaler.transform(transformed)

def get_nearest(pca_coords, n_neigh, rad):  
    neigh = KNeighbor(n_neigh, rad)
    neigh.fit(pca_coords)
    return neigh.kneighbors([pca_coords[-1]], n_neigh, return_distance=False) # -1 is the newest addition, which in this case is real time microphone input.


# SEGMENTATION AND FEATURE EXTRACTION

# partition() and segment() are direct recursive functions.
# They segment blocks into halves, avoiding NaN and making sure we segment larger blocks into all of the required sizes.
# segment() will take a paritioned (or not) block and save its features out, it will then halve itself and save the features of the front section as a block,
# It does this again and again until the limit is reached. 
# This effectively segments each block into multiple mini-blocks and extracts/saves out the audio features of each of these blocks to a list associted with a file name.

block_size = 256 # Multiple by 1024 to understand the librosa.stream utilization below.
max_subdivs = 256 # The amount of subdivisons possible on each block. Must be 2 ** n.  
max_size = 2 ** 9 # Max block size. 
min_size = 2 ** 4 # Min block and segment size. Lower numbers increase computation by 2 x n. 
        
def set_granulation_analysis(block_size_, max_subdivs_, max_size_, min_size_):
        block_size = block_size_
        max_subdivs = max_subdivs_
        max_size = max_size_
        min_size = min_size_
        

def partition(block):
        if -9000 in block: # If we find -1 in the block as this means the block overstepped our audio length, we then split the block in half and check if -1 is contained in the front section.
                # print("block of size" + str(block.size) + " contains NaN, segmenting...")
                segmented_block = np.array_split(block, 2)
                block = segmented_block[0]
                return partition(block)
        else:
                return block

def segment(block, segwise_slices, segments):
        if(block.size > (min_size * block_size) and block.size < (max_size * block_size) and segments < max_subdivs): # If our size is above the limit it is not the limit, it *should* be a mulitple of 2.
                segmented_block = np.array_split(block, segments)
                feature_shards = []
                audio_shards = []
                #print("segmenting...")
                for seg in segmented_block: 
                        shard_size = seg.size
                        if (shard_size / block_size) < min_size:
                                return block
                        mfccs = librosa.feature.mfcc(seg, 
                                             sample_rate, 
                                              n_mfcc=mfcc_size)

                        stddev_mfccs = np.std(mfccs, axis=1)
                        mean_mfccs = np.mean(mfccs, axis=1)
                        average_difference = np.zeros((mfcc_size,))
                        for i in range(0, len(mfccs.T) - 2, 2):
                                average_difference += mfccs.T[i] - mfccs.T[i+1]
                        average_difference /= (len(mfccs) // 2)   
                        average_difference = np.array(average_difference)
                        concat_mfcc = np.hstack((stddev_mfccs, mean_mfccs))
                        concat_feats = np.hstack((concat_mfcc, average_difference))
                        # ZERO CROSSING EXRACTION:
                        if shard_size > 2048:
                                zero_crossings =  librosa.zero_crossings(seg, pad = False)
                                zero_crossings_sum = sum(zero_crossings)
                                concat_feats = np.hstack((concat_feats, zero_crossings_sum))
                        # SPECTRAL FLATNESS
                        if shard_size > 4094:
                                flatness = librosa.feature.spectral_flatness(y=seg, center=True)
                                flatness = np.ravel(flatness)
                                concat_feats = np.hstack((concat_feats, flatness))
                        # CHROMA EXTRACTION:
                        if shard_size > 8192: 
                                hop_length = 256 * 2
                                chroma =  librosa.feature.chroma_cens(seg, sr=sample_rate, n_octaves = 7, hop_length=hop_length)
                                chroma = chroma.flatten()
                                concat_feats = np.hstack((concat_feats, chroma)) 
                        feature_shards.append(concat_feats)
                        audio_shards.append(seg)
                dims = 0
                for size in range(min_size, max_size): 
                        if size != 0 and ((size & (size - 1)) == 0):
                                dims += 1
                                if(shard_size / block_size) == size:
                                        segwise_slices[dims-1] = [audio_shards, feature_shards]
                segments = segments * 2
                return segment(block,segwise_slices, segments)
        else:
                return block
        

def get_features(file_path_, dataset_, file_name_, sample_rate_): 
        sample_rate = sample_rate_
        print("blockifying...")
        blocks = librosa.stream(file_path_,
                          block_length= block_size,
                          frame_length=2048,
                          hop_length= 2048,
                          fill_value= -9000)
        segwise_slices = [None] * int(math.log(max_size)/math.log(2) - math.log(min_size)/math.log(2))
        segments = 2
        dims = 0
        print(file_name_)
        for y_block in blocks:
            y_block = partition(y_block) # Cut the 2^n sized block into two halves and use the front half, repeat until we have a full block. 
            y_block = segment(y_block, segwise_slices, segments) 
        dataset_ += [(file_name_, segwise_slices)]

def get_features_raw(raw_data_, dataset_, name):
        segwise_slices = [None] * int(math.log(max_size)/math.log(2) - math.log(min_size)/math.log(2))
        segments = 2
        dims = 0
        raw_data = partition(raw_data) # Cut the 2^n sized block into two halves and use the front half, repeat until we have a full block. 
        raw_data = segment(raw_data, segwise_slices, segments) 
        dataset_ += [(name, segwise_slices)]


def get_segment_amounts():
        return int(math.log(max_size)/math.log(2) - math.log(min_size)/math.log(2))

# Note: If anything changes about the content in the directory of sounds, you must run get_features again before get_saved_features.

def get_saved_features(saved_features, dataset_, file_):
            dataset_ += [(file_, saved_features)]

# Compares the input PCA coordinates with a list of nearby PCA coordinates.
def compare_pca(input, nearest, thrs):
    dist = 0
    for i in nearest: 
        dist += np.linalg.norm(input-i) #  Euclidean distance. 
    return dist

