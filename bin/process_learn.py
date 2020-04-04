import os
import numpy as np
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
sample_rate = 44100 # Default


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

# segment() is a direct recursive function.
# They segment blocks into halves, avoiding NaN and making sure we segment larger blocks into all of the required sizes.
# segment() will take a paritioned (or not) block and save its features out, it will then halve itself and save the features of the front section as a block,
# It does this again and again until the limit is reached. 
# This effectively segments each block into multiple mini-blocks and extracts/saves out the audio features of each of these blocks to a list associted with a file name.

max_subdivs = 3 # The amount of subdivisons, also the amount of 'dims' or dimensions.      

# segment() is easily modified to include whatever feature extraction method is needed. 

def segment(block, segments, name, dim):
                sample_data = []
                segmented_block = np.array_split(block, segments)
                equal_loudness = EqualLoudness()
                rhythm_extract = RhythmExtractor()
                freq_bands = FrequencyBands()
                spectrum = Spectrum()
                loudness = LevelExtractor() 
                zero_crossing = ZeroCrossingRate()
                dissonance = Dissonance()
                spectralpeaks = SpectralPeaks(minFrequency = 100)
                ps = PitchSalience()
                auto_corr = AutoCorrelation()
                # !!!!
                # We need to normalize the outcoming feature values, to do so we need to know the average values for it.
                # We can't normalize the vector of features together or larger numbers will remain large, just better proportioned which is an improvement but not a fix.
                # Need to implement weighting... We need to normalize each incoming feature selection to 1? 
                for index, seg in enumerate(segmented_block):
                        if index <= segments/2: # We only analyze the first half of the split audio block.
                                if (seg.size % 2) != 0: # There is an error which occurs with taking an FFT of an unevenly sized window.
                                        seg = np.append(seg, 0.0)
                                seg = essentia.array(seg)
                                seg = equal_loudness(seg)
                                # if loudness(seg) > -50: # Current microphone is ducking transients I think, causing any attempts to ignore quiet sounds to fail.
                                w = Windowing(type = 'hann')
                                f = w(seg)
                                shard_size = seg.size
                                spec = spectrum(f)
                                # MFCC/SPEC EXTRACTION:
                                # Check normalization and weighting...
                                mfcc = MFCC(inputSize = spec.size)
                                mfcc_bands, mfcc_coeffs = mfcc(spec)
                                concat_feats = np.hstack((mfcc_coeffs))
                                # ZERO CROSSINGS EXRACTION:
                                if shard_size > 2048:
                                        #fb = freq_bands(spec)
                                        zc = zero_crossing(seg)
                                        #print(zc)
                                        concat_feats = np.hstack((concat_feats, zc))
                                # AUTOCORRECTION/PITCH EXTRACTION:
                                if shard_size > 8192:   
                                        #sp_freq, sp_mag = spectralpeaks(spec)
                                        #ac = auto_corr(seg)
                                        pitch_sal = ps(seg)
                                        concat_feats = np.hstack((concat_feats, pitch_sal))
                                       # concat_feats = np.hstack((concat_feats, ac))
                                seg_name = name + "_" + str(segments) + "_" + str(dim) + "_" + str(index) + '.wav'
                                sf.write('./data/slices/' + seg_name, seg, sample_rate)
                                 # Starts with 100 percent chance of being pruned. If this goes below 0 the sample is immune to being prunned.#
                                sample_data.append([seg_name, concat_feats, 0, dim, seg.size])
                return sample_data

def get_feats_frm_data(data,dataset,name,amp,dim,segments, sr):
        sample_rate = sr
        t0 = time.time()
        data *= amp
        sample_data = segment(data, segments, name, dim)
        t1 = time.time()
        #print("feature seg/collection speed: " + str(t1-t0)) # Speed check.
        dataset += [([name], sample_data)]


def get_segment_amounts():
        return max_subdivs

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
