import os
import numpy as np
import librosa
import math
from sklearn import preprocessing
import sklearn.decomposition 
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors as KNeighbor
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
import soundfile as sf
import essentia.standard
from essentia.standard import *
import hdbscan
import time
import threading

mfcc_size = 24
sample_rate = 44100


# This script contains definitions used in audio_comparative_analysis.  
# Contains functions which process audio information. This includes storing, cluster analysis and feature extraction. 

# The get_pca and get_features definitions were originally from here https://gist.github.com/fedden/52d903bcb45777f816746f16817698a0#file-dim_reduction_notebook-ipynb
# Medium article: https://medium.com/@LeonFedden/comparative-audio-analysis-with-wavenet-mfccs-umap-t-sne-and-pca-cb8237bfce2f
# Compared with the original my script is real time and segments audio first. Additionally, I use Essentia instead of Librosa and added more task specfic functionality.

def get_pca(data, n):
    pca = sklearn.decomposition.PCA(n_components= n)
    transformed = pca.fit(data).transform(data)
    scaler = MinMaxScaler()
    scaler.fit(transformed)
    return scaler.transform(transformed)

def get_nearest(data, n_neigh, rad):  
    neigh = KNeighbor(n_neigh, rad)
    neigh.fit(data)
    return neigh.kneighbors([data[-1]], n_neigh, return_distance=False) # -1 is the newest addition, which in this case is real time microphone input.

# get_cluser_centers() returns the indices of data entries that are located in the center of the detected clusters. 

def get_cluster_centers(data, min):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min, prediction_data=True,cluster_selection_epsilon = 0.5 )
        clusterer = clusterer.fit(data)
        selected_clusters = clusterer.condensed_tree_._select_clusters()
        raw_condensed_tree = clusterer.condensed_tree_._raw_tree
        exemplars = []
        for cluster in selected_clusters:
                cluster_exemplars = np.array([], dtype=np.int64)
                for leaf in clusterer._prediction_data._recurse_leaf_dfs(cluster):
                        leaf_max_lambda = raw_condensed_tree['lambda_val'][
                                raw_condensed_tree['parent'] == leaf].max()
                        points = raw_condensed_tree['child'][
                                (raw_condensed_tree['parent'] == leaf) &
                                (raw_condensed_tree['lambda_val'] == leaf_max_lambda)]
                        cluster_exemplars = np.hstack([cluster_exemplars, points])
                cluster_exemplars.tolist()
                exemplars.append(cluster_exemplars)
        return exemplars
        print(exemplars)

# SEGMENTATION AND FEATURE EXTRACTION

# partition() and segment() are direct recursive functions.
# They segment blocks into halves, avoiding NaN and making sure we segment larger blocks into all of the required sizes.
# segment() will take a paritioned (or not) block and save its features out, it will then halve itself and save the features of the front section as a block,
# It does this again and again until the limit is reached. 
# This effectively segments each block into multiple mini-blocks and extracts/saves out the audio features of each of these blocks to a list associted with a file name.

block_size = 256 # Multiple by 1024 to understand the librosa.stream utilization below.
max_subdivs = 256 # The amount of subdivisons possible on each block. Must be 2 ** n.  
max_size = 2 ** 11 # Max block size. 
min_size = 2 ** 6 # Min block and segment size. Lower numbers increase computation by 2 x n. 
        
def set_granulation_analysis(block_size_, max_subdivs_, max_size_, min_size_):
        block_size = block_size_
        max_subdivs = max_subdivs_
        max_size = max_size_
        min_size = min_size_
        

def partition(block, amp):
        if -9000 * amp in block: # If we find n in the block as this means the block overstepped our audio length, we then split the block in half and check if n is contained in the front section.
                segmented_block = np.array_split(block, 2)
                block = segmented_block[0]
                return partition(block, amp)
        else:
                return block

# segment() is easily modified to include whatever feature extraction method is needed. 

def segment(block, segwise_slices, segments, name, threads):
        if(block.size > (min_size * block_size) and block.size < (max_size * block_size) and segments < max_subdivs): # If our size is above the limit it is not the limit, it *should* be a mulitple of 2.
                feature_shards = []
                audio_shards = []
                gates = []
                immun = []
                block_segments = segments
                segmented_block = np.array_split(block, block_segments)
                segments = segments * 2 
                seg_thread = threading.Thread(target=segment, args=(block, segwise_slices, segments, name, threads,)) # I have observed increase in speed using threading, almost double the speed in some cases. 
                threads.append(seg_thread)  
                seg_thread.start()
                equal_loudness = EqualLoudness()
                spectrum = Spectrum()
                loudness = LevelExtractor() 
                spectralpeaks = SpectralPeaks(orderBy='magnitude',
                                  magnitudeThreshold=0.00001,
                                  minFrequency=20,
                                  maxFrequency=3500,
                                  maxPeaks=60)
                # !!!!
                # We need to normalize the outcoming feature values, to do so we need to know the average values for it.
                # We can't normalize the vector of features together or larger numbers will remain large, just better proportioned which is an improvement but not a fix.
                # Need to implement weighting... We need to normalize each incoming feature selection to 1? 

                for index, seg in enumerate(segmented_block):
                        seg = equal_loudness(seg)
                        # if loudness(seg) > -50: # Current microphone is ducking transients I think, causing any attempts to ignore quiet sounds to fail.
                        w = Windowing(type = 'hann')
                        f = w(seg)
                        shard_size = seg.size
                        if (shard_size / block_size) < min_size:
                                return block
                        spec = spectrum(f)
                        # MFCC/SPEC EXTRACTION:
                        mfcc = MFCC(inputSize = spec.size)
                        mfcc_bands, mfcc_coeffs = mfcc(spec)
                        concat_feats = np.hstack((mfcc_coeffs))
                        # ZERO CROSSINGS EXRACTION:
                        if shard_size > 2048:
                                zc = ZeroCrossingRate()
                                zero_crossing = zc(seg)
                                concat_feats = np.hstack((concat_feats, zero_crossing))
                        # AUTOCORRECTION/PITCH EXTRACTION:
                        if shard_size > 8192:
                                #ac = AutoCorrelation()
                                #auto_correlation = ac(seg)
                                ps = PitchSalience()
                                pitch_sal = ps(seg)
                                concat_feats = np.hstack((concat_feats, pitch_sal))
                        # RHYTHM:
                               # re = RhythmExtractor()
                                #d = Danceability()
                                #dance = d(seg)
                                #rhythm = re(seg)
                                #concat_feats = np.hstack((concat_feats, dance))
                                #concat_feats = np.hstack((concat_feats, rhythm))
                                #peh = PercivalEnhanceHarmonics()
                                #en_harm = peh(auto_correlation)
                                #concat_feats = np.hstack((concat_feats, en_harm))
                        seg_name = name + str(block_segments+shard_size/ block_size) + str(index) + '.wav'
                        sf.write('./data/slices/' + seg_name, seg, sample_rate)
                        feature_shards.append(concat_feats) # <--- here
                        audio_shards.append(seg_name)
                        immun.append(1.0) # Starts with 100 percent chance of being pruned. If this goes below 0 the sample is immune to being prunned.
                dims = 0
                for size in range(min_size, max_size): 
                        if size != 0 and ((size & (size - 1)) == 0):
                                dims += 1
                                if(shard_size / block_size) == size:
                                        segwise_slices[dims-1] = [audio_shards, feature_shards, immun]
        else:
                return block
        

def get_features(file_path_, dataset_, file_name_, sample_rate_, amp): 
        t0 = time.time()
        threads = []
        sample_rate = sample_rate_
        name = file_name_
        print(file_path_)
        blocks = librosa.stream(file_path_,
                          block_length= block_size,
                          frame_length=2048,
                          hop_length= 2048,
                          fill_value= -9000)
        segwise_slices = [None] * int(math.log(max_size)/math.log(2) - math.log(min_size)/math.log(2))
        segments = 2
        dims = 0
        for y_block in blocks:
            y_block *= amp
            y_block = partition(y_block, amp) # Cut the 2^n sized block into two halves and use the front half, repeat until we have a full block. 
            y_block = segment(y_block, segwise_slices, segments, name, threads) 
        for thread in threads:
                thread.join() # Wait for all threads to complete before we add to the dataset...
        t1 = time.time()
        print("feature seg/collection speed: " + str(t1-t0)) # Speed check.
        dataset_ += [([file_name_], segwise_slices)]


def get_segment_amounts():
        return int(math.log(max_size)/math.log(2) - math.log(min_size)/math.log(2))

# Note: If anything changes about the content in the directory of sounds, you must run get_features again before get_saved_features.

def get_saved_features(saved_features, dataset_, file_):
            dataset_ += [(file_, saved_features)]

# Compares the input PCA coordinates with a list of nearby PCA coordinates.

def euclid(input, nearest, gate):
        dist = 0
        dist += np.linalg.norm(input-nearest) #  Euclidean distance. 
        if dist > gate:
                return True
        else:
                return False
