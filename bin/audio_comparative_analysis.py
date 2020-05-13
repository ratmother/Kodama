import os
import time
import math
import queue
import itertools
import random
import time
import operator
import random
import string
import wavio
import pickle
import ringbuffer
import numpy as np
import soundfile as sf
import hdbscan
import seaborn as sns
import essentia.standard
from essentia.standard import *
from scipy.io.wavfile import write
import scipy.signal
from sklearn import preprocessing
import sklearn.decomposition 
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors as KNeighbor
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

 ##  KODAMA ##

 ## Audio Comparative Analysis ##

 # The functionality of this script is based on an inspired intrepretation of memory.
 # This script subdivides an audio buffer into discrete segments of varying sizes. 
 # These  are collected together based on length. These collections of discrete segments are refered to as belonging to a 'dim', short for dimension.
 # After segmenting the audio we extract the features with Essentia and produce a sample which contains: [wav file name, feature vector, mortality, dim, segment size, cooldown].
 # We run PCA and KNN on each segment (input sample) using a dim-based selection of the dataset. We then place the sample into the dataset if that sample passes a certain euclidean threshold.
 # At this point we also collect the nearest samples to the input sample. 
 # Each dim(slices of the dataset based on audio/feature length) is pruned down using HDBSCAN probabilities. 
 # The output of this script is a list of nearest audio wav files.
 # The optional graphical plot shows the HDBSCAN analysis of the dataset (split into subplots by dim). Lower alpha values imply lower probablity of being in a cluster. Colour refers to the cluster class.
 # The result is a collection of clusters seperated by dimension. The clusters remodel themselves based on recall upon new input. 


class seg_analysis():
    def __init__(self, device_index, pkl_file, plot_results,dim, audio_length, sample_rate, size_limit):
      self.limit = size_limit * 100 # The limit is a value used to control how many files can exist at a time, it is approximately 100 mb per 10000 limit value.
      self.sample_rate = int(sample_rate)
      self.dim = dim
      self.dataset,self.input_data,self.nearest, self.input_data = [],[],[],[]
      self.nearest = [[] for x in range(0, dim)]
      self.input_data = [[] for x in range(0, dim)]
      self.distance_thresholds = [0.1, 0.6, 0.7, 0.8, 0.9, 0.9] # The inital distance threshold for different dims. 
      self.num_neighbours = [4, 5, 8, 10, 11, 12, 13] # Number of neighbours for each dim to be checked against in dim-wise analysis.
      self.axs = [0 for x in range(0, dim)]
      self.plot = plot_results 
      self.removal_queue = []
      if self.plot:
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for i in range(0, dim):
            self.axs[i] = fig.add_subplot(2, 3, i+1)
      self.rb = ringbuffer.RingBuffer(audio_length * self.sample_rate)

    ### Dim-wise cluster analysis functions ###

    # Consolidate runs cluster prunning on each dim, also handles the waste disposal queue. 

    def consolidate(self):
        for dim in range(0, self.dim):
            self.cluster_prunning(dim)
        if len(self.dataset) > 1:
            names,features = zip(*self.dataset)
            for index, sample in enumerate(features): 
                delete_sample = True
                for sample_index, sample_data in enumerate(sample):
                    if sample_data[0] is None:
                        del sample_data
                    else:
                        delete_sample = False
                if delete_sample:
                    del sample
        if len(self.removal_queue) > 100: # We hold these audio files for a little while in case our audio generator needs them after they are prunned. 
            for file in self.removal_queue[:100]:
                try:
                    os.remove(file)
                except:
                    print("A file to be deleted failed because it is already deleted")
                    pass
            del self.removal_queue[:100]

    # seg_dim_analysis finds the nearest samples to the input sample while handling cooldowns and thresholds.

    def seg_dim_analysis(self, dim): 
        saa = [] # Sample-audio-arrays.
        sfv = [] # Sample-feature-vectors.
        swp = [] # Sample file with parameters.
        if len(self.dataset) > 1:
            names,features = zip(*self.dataset)
            for index, sample in enumerate(features): 
                for sample_index, sample_data in enumerate(sample):
                    if sample_data[1] is not None and sample_data[3] == dim:
                        swp.append([sample_data[0],sample_data[2],sample_data[3], sample_data[4], sample_data[5]])
                        saa.append(sample_data[0]) 
                        sfv.append(sample_data[1])
        self.dataset += self.input_data[dim] # From the last cycle of this dim we add the input data.
        self.input_data[dim] = [] # Wipe the array so we can fill it with the new input data.
        self.nearest[dim].clear()
        self.get_feats_frm_data(self.rb.get(),self.input_data[dim], self.random_generator(), 0.8, dim, 2**dim)
        name,features = zip(*self.input_data[dim]) 
        for index, sample in enumerate(features): 
            for sample_index, sample_data in enumerate(sample): # sample_data[0] = audio sample, sample_data[1] = feature data, sample_data[2] = sample immunity, sample_data[3] = sample dim
                if sample_data[1] is not None:
                    # Once we have a sample, we append it to the temporary dimension wise array, if it doesn't pass the cluster analysis check it is removed.
                    save = False
                    saa.append(sample_data[0]) 
                    swp.append([sample_data[0],sample_data[2],sample_data[3], sample_data[4]])
                    sfv.append(sample_data[1])
                    sfv_arr = np.stack(sfv, axis=0) # Converts the list of feature vectors into an array, which is what process_learn needs.
                    try:
                        sfv_pca = self.pca(sfv_arr, 3)
                    except:
                        return False # We keep the sample as PCA should only fail in the case of not having enough samples.
                    try:
                        nearest = self.knearest(sfv_pca, self.num_neighbours[dim], 0.2) 
                    except:
                        return False # We also keep the sample when we do not have enough samples for k-nearest neighbour. 
                    for i in range (1,self.num_neighbours[dim]):
                        dist_check = self.euclid(sfv_pca[-1], sfv_pca[i], self.distance_thresholds[dim]) # Theory: If we always add shards when our nearest neighbour(s) is too far away, we should make more clusters.
                        if dist_check:
                            save = True 
                        else:
                            swp[nearest[0][i]][4] -= 1 # Cooldown ticks down 1 per selection.
                            if swp[nearest[0][i]][4] <= 0: # When less or equal to 0, we take the selection to the audio engine.
                                self.nearest[dim].append(swp[nearest[0][i]])
                                swp[nearest[0][i]][4] = 3 # Setting cooldown to 3 selections.
                                swp[nearest[0][i]][1] -= 0.001 # Reduction of prune chance due to being selected.
                    del sfv[-1]
                    del saa[-1]
                    if not save:
                        if self.distance_thresholds[dim] > 0.1:
                            self.distance_thresholds[dim] -= 0.01
                        if sample_data[0] not in self.nearest[dim]: # Don't delete a sample that is in nearest.
                            file_path = './data/slices/' + sample_data[0]
                            self.removal_queue.append(file_path)
                            sample_data[0] = None
                            sample_data[1] = None
                            sample_data[2] = None
                    else:
                        self.distance_thresholds[dim] += 0.01

    # cluster_prunning runs HDBSCAN cluster analysis on a dim section of the dataset, assigns a higher chance of removal to samples which have low probablity of being in a cluster.

    def cluster_prunning(self, dim): 
        # Theory: By running PCA and DBSCAN on a growing sample collection and granting prune immortality to cluster centers, we get a stronger sense of which samples are representative of our growing audio collection.
        saa = []
        sfv = []
        swp = []
        if len(self.dataset) > 1:
            names,features = zip(*self.dataset)
            for index, sample in enumerate(features): 
                for sample_index, sample_data in enumerate(sample):
                    if sample_data[1] is not None and sample_data[3] == dim:
                        saa.append(sample_data[0]) 
                        sfv.append(sample_data[1]) # Do we need this if we have swp...
                        swp.append([sample_data[0], sample_data[1], sample_data[2],sample_data[3], sample_data[4], sample_data[5]])
        if len(sfv) >= 5:
            sfv_arr = np.stack(sfv, axis=0)
            try:
                sfv_pca = self.pca(sfv_arr, 2)
            except:
                return False
            coeff = self.maprange(len(sfv), (0, self.limit), (1.0, 10.0)) # Multipler for mortality based on size of dataset and user limit variable.
            clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
            gen_min_span_tree=True, leaf_size=40,
            metric='euclidean', min_cluster_size= 5, min_samples=None, p=None)
            clusterer.fit(sfv_pca)
            if self.plot:
                palette = sns.color_palette()
                cluster_colors = [sns.desaturate(palette[col], sat)
                  if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                  zip(clusterer.labels_, clusterer.probabilities_)]
            # Run over the dataset assigning mortalities probabilities.
            for index, (label, prob) in enumerate(zip(clusterer.labels_, clusterer.probabilities_)):
                x = sfv_pca[index][0]
                y = sfv_pca[index][1]
                if self.plot:
                    self.axs[dim].scatter(x, y, s = 6 - swp[index][2] * 1000, c= cluster_colors[index], alpha = prob) 
                mortality = (1 - prob) * 0.05
                swp[index][2] += mortality
                if swp[index][2] * coeff > random.random(): # If our mortality times coeff is greater than random remove the sample.
                    if swp[index][0] not in self.nearest[dim]:
                            try:
                                print("deleting because of prunning")
                                file_path = './data/slices/' + swp[index][0]
                                self.removal_queue.append(file_path)
                            except:
                                print("Could not remove " + str('./data/slices/' + swp[index][0]) + " ,likely because it was already deleted.") 
                            swp[index][0] = None
                            swp[index][1] = None
                            swp[index][2] = None
            if self.plot:
                plt.pause(0.01)

    # fill_ringbuff fills the ring buffer with the incoming audio signal.

    def fill_ringbuff(self,data, window_size):
        window = scipy.signal.general_gaussian(window_size+1,p=1,sig=(window_size*0.8)) # Do we need this?
        audio_data = np.fromstring(data, dtype=np.float32)
        audio_data = np.append(audio_data, [0])
        self.rb.extend(audio_data * window)

    # get_nearest retrieves the nearest neighbour sample list.

    def get_nearest(self,dim):
       # Our nearest sample list tends to have collected a few duplicates along the way which is unessesary, we use itertools to avoid this problem.
        self.nearest[dim].sort()
        dict_nearest = list(self.nearest[dim] for self.nearest[dim],_ in itertools.groupby(self.nearest[dim]))
        if len(self.nearest) > 0:
            files = []
            for index, file in enumerate(dict_nearest):
                if index < 5:
                    files.append(file)
                else:
                    pass
            return files

    ### Audio feature analysis functions ###

    def get_feats_frm_data(self, data,dataset,name,amp,dim,segments):
            t0 = time.time()
            data *= amp
            sample_data = self.feat_extract(data, segments, name, dim)
            t1 = time.time()
            #print("feature seg/collection speed: " + str(t1-t0)) # Speed check.
            dataset += [([name], sample_data)]

    # get_cluser_centers() returns the indices of data entries that are located in the center of the detected clusters. 
    def feat_extract(self, block, segments, name, dim):
                    sample_data = []
                    segmented_block = np.array_split(block, segments) # Splits the ringbuffer into segments, this is effectively what makes a dim a dim.
                    spectrum = Spectrum()
                    zero_crossing = ZeroCrossingRate()
                    ps = PitchSalience()
                    for index, seg in enumerate(segmented_block):
                                    if (seg.size % 2) != 0: # There is an error which occurs with taking an FFT of an unevenly sized window, this fixes it.
                                            seg = np.append(seg, 0.0)
                                    seg = essentia.array(seg)
                                    w = Windowing(type = 'hann')
                                    f = w(seg)
                                    spec = spectrum(f)
                                    # MFCC/SPEC EXTRACTION:
                                    mfcc = MFCC(inputSize = spec.size)
                                    mfcc_bands, mfcc_coeffs = mfcc(spec)
                                    concat_feats = np.hstack((mfcc_coeffs))
                                    # Larger segments have additional features extracted. Pitch salience being irrelevant for short segments. Experimental.
                                    # ZERO CROSSINGS EXRACTION:
                                    if seg.size > 2048:
                                            zc = zero_crossing(seg)
                                            concat_feats = np.hstack((concat_feats, zc))
                                    # PITCH EXTRACTION:
                                    if seg.size > 8192:   
                                            pitch_sal = ps(seg)
                                            concat_feats = np.hstack((concat_feats, pitch_sal))
                                    seg_name = name + "_" + str(segments) + "_" + str(dim) + "_" + str(index) + '.wav'
                                    sf.write('./data/slices/' + seg_name, seg, self.sample_rate)
                                    sample_data.append([seg_name, concat_feats, 0, dim, seg.size,0]) # [segment file name, feature vector, mortality, dim, segment size, cooldown]
                    return sample_data

    # Note: If anything changes about the content in the directory of sounds, you must run get_features again before get_saved_features.
    @staticmethod  
    def get_saved_features(saved_features, dataset_, file_):
                dataset_ += [(file_, saved_features)]

    # Compares the input PCA coordinates with a list of nearby PCA coordinates.
    @staticmethod  
    def euclid(input, nearest, gate):
            dist = 0
            dist += np.linalg.norm(input-nearest) #  Euclidean distance. 
            if dist > gate:
                    return True
            else:
                    return False

    # Medium article: https://medium.com/@LeonFedden/comparative-audio-analysis-with-wavenet-mfccs-umap-t-sne-and-pca-cb8237bfce2f

    ### PCA and KNeighbour functions ###

    @staticmethod  
    def pca(data, n):
        pca = sklearn.decomposition.PCA(n_components= n)
        transformed = pca.fit(data).transform(data)
        scaler = MinMaxScaler()
        scaler.fit(transformed)
        return scaler.transform(transformed)

    @staticmethod  
    def knearest(data, n_neigh, rad):  
        neigh = KNeighbor(n_neigh, rad)
        neigh.fit(data)
        return neigh.kneighbors([data[-1]], n_neigh, return_distance=False) # -1 is the newest addition, which in this case is real time microphone input.

    ### Utility and math functions ###

    @staticmethod
    def random_generator(size=3, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for x in range(size))
        
    @staticmethod
    def maprange(s,a,b): # From https://rosettacode.org/wiki/Map_range
	    (a1, a2), (b1, b2) = a, b
	    return  b1 + ((s - a1) * (b2 - b1) / (a2 - a1))