import os
import time
import itertools
import random
import string
import soundfile as sf
import wavio
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import pickle
import process_learn as pl
import time
import operator
import random
import scipy.signal
import numpy as np
import queue
import ringbuffer

class seg_analysis():
    def __init__(self, device_index, pkl_file, plot_results,dim, audio_length, sample_rate, size_limit):
      self.limit = size_limit * 100 # The limit is a value used to control how many files can exist at a time, it is approximately 100 mb per 10000 limit value.
      self.sample_rate = int(sample_rate)
      self.pkl_file = pkl_file
      self.audio = None
      self.dim = dim
      self.dataset,self.input_data,self.nearest, self.input_data = [],[],[],[]
      self.nearest = [[] for x in range(0, dim)]
      self.input_data = [[] for x in range(0, dim)]
      self.distance_thresholds = [0.1, 0.6, 0.7, 0.8, 0.9, 0.9] # The distance threshold for different dims can be different, as they are scaled differently. 
      self.num_neighbours = [4, 5, 8, 10, 11, 12, 13] # Number of neighbours for each dim to be checked against.
      self.growth_rates = [0 for x in range(0, dim)]
      self.plot = plot_results 
      self.removal_queue = []
      if self.plot:
            fig, self.axs = plt.subplots(dim, figsize=(4,12)) 
      pl.max_subdivs = dim
      self.rb = ringbuffer.RingBuffer(audio_length * self.sample_rate)

    def consolidate(self):
        for dim in range(0, self.dim):
            self.basis_prunning(dim)
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
        if len(self.removal_queue) > 100: # We hold these audio files for a little while in case our audio generator needs them after they are technically prunned. 
            for file in self.removal_queue[:100]:
                os.remove(file)
            del self.removal_queue[:100]

    def seg_dim_analysis(self, dim): 
        saa = [] # Sample-audio-arrays.
        sfv = [] # Sample-feature-vectors.
        swp = [] # Sample with parameters for sending control information out of Python.
        if len(self.dataset) > 1:
            names,features = zip(*self.dataset)
            for index, sample in enumerate(features): 
                for sample_index, sample_data in enumerate(sample):
                    if sample_data[1] is not None and sample_data[3] == dim:
                        swp.append([sample_data[0],sample_data[2],sample_data[3], sample_data[4]])
                        saa.append(sample_data[0]) 
                        sfv.append(sample_data[1])
        self.dataset += self.input_data[dim] # From the last cycle of this dim we add the input data.
        self.input_data[dim] = [] # Wipe the array so we can fill it with the new input data.
        self.nearest[dim].clear()
        pl.get_feats_frm_data(self.rb.get(),self.input_data[dim], self.random_generator(), 0.8, dim, 2**dim, self.sample_rate) # REMEMBER TO CHANGE THIS

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
                        sfv_pca = pl.get_pca(sfv_arr, 3)
                    except:
                        print(len(sfv))
                        print(str(dim) + " Failed get pca!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        return False # We keep the shard as PCA should only fail in the case of not having enough samples.
                    try:
                        nearest = pl.get_nearest(sfv_pca, self.num_neighbours[dim], 0.2) 
                    except:
                        print(len(sfv))
                        print(str(dim) + " Failed get nearest")
                        return False # We also keep the shard when we do not have enough samples for k-nearest neighbour. 
                    for i in range (1,self.num_neighbours[dim]):
                        dist_check = pl.euclid(sfv_pca[-1], sfv_pca[i], self.distance_thresholds[dim]) # Theory: If we always add shards when our nearest neighbour(s) is too far away, we should make more clusters.
                        if dist_check:
                            save = True # Keeping this simple for now.
                        else:
                            self.nearest[dim].append(swp[nearest[0][i]])
                            #self.nearest[dim].append(saa[nearest[0][i]])
                    del sfv[-1]
                    del saa[-1]
                    if not save:
                        if self.distance_thresholds[dim] > 0.1:
                            self.distance_thresholds[dim] -= 0.01
                        #print(str(dim) + " did not save sample.")
                        if sample_data[0] not in self.nearest[dim]: # Don't delete a sample that is in nearest.
                            file_path = './data/slices/' + sample_data[0]
                            self.removal_queue.append(file_path)
                            sample_data[0] = None
                            sample_data[1] = None
                            sample_data[2] = None
                    else:
                        self.distance_thresholds[dim] += 0.01
                        #print(str(dim) + " saved sample.")
                   # print(str(self.distance_thresholds[dim]) + " for dim " + str(dim))

    def basis_prunning(self, dim): 
        # Theory: By running PCA and DBSCAN on a growing sample collection and granting prune immortality to cluster centers, we get a stronger sense of which samples are representative of our growing audio collection.
        saa = []
        sfv = []
        if len(self.dataset) > 1:
            names,features = zip(*self.dataset)
            for index, sample in enumerate(features): 
                for sample_index, sample_data in enumerate(sample):
                    if sample_data[1] is not None and sample_data[3] == dim:
                        saa.append(sample_data[0]) 
                        sfv.append(sample_data[1])
        if len(sfv) >= (5 + self.growth_rates[dim]):
            self.growth_rates[dim] += 1 
            sfv_arr = np.stack(sfv, axis=0)
            try:
                sfv_pca = pl.get_pca(sfv_arr, 2)
            except:
                return False
            centers = pl.get_cluster_centers(sfv_pca, 3)
            flat_centers = [item for sublist in centers for item in sublist]
            coeff = self.maprange(len(sfv), (0, self.limit), (1.0, 10.0))
            # PLOTTING
            if self.plot:
                self.axs[dim].clear()
                self.axs[dim].set_yticklabels([])
                self.axs[dim].set_xticklabels([])
                cpt = sum([len(files) for r, d, files in os.walk("./data/slices/")])
                plt.title('Amount of files: ' + str(cpt), y = -0.5)
                #plt.title('Amount of files: ' + str(len(self.dataset)), y = -0.5)
                plt.axis([0, 1, 0, 1])
                for index, cords in enumerate(sfv_pca):
                    x = cords[0]
                    y = cords[1]
                    if index in flat_centers:
                        self.axs[dim].scatter(x, y, s = 5.9 - coeff,  c='blue', alpha = 0.8) 
                        self.axs[dim].scatter(x, y, s = 11.9 - coeff,  c='red', alpha = 0.2) 
                    else:
                        self.axs[dim].scatter(x, y, s = 5.9,  c='blue') 
                self.axs[dim].text(0.5,-0.1, "{:.2f}".format(self.distance_thresholds[dim]), size=12, ha="center", 
                transform=self.axs[dim].transAxes)
                plt.pause(0.01)
            # PRUNNING
            count = 0
            for index, sample in enumerate(features): 
                for sample_index, sample_data in enumerate(sample):
                    if sample_data[1] is not None:
                        count += 1
                        if count not in flat_centers:
                            sample_data[2] += 0.001 # This sample is now less likely to be pruned.
                        #if sample_data[2] > random.random(): # If the samples prune immunity is below 0, this will never pass.
                        if sample_data[2] * coeff > random.random():
                            print ("%.6f" % (sample_data[2] * coeff))
                            if sample_data[0] not in self.nearest[dim]: # Don't delete a sample that is in nearest.
                                #print(str(dim) + " pruned by chance of" + str(sample_data[2]))
                                try:
                                    print("deleting because of prunning")
                                    file_path = './data/slices/' + sample_data[0]
                                    self.removal_queue.append(file_path)
                                except:
                                    print("Could not remove " + str('./data/slices/' + sample_data[0]) + " ,likely because it was already deleted.") 
                                sample_data[0] = None
                                sample_data[1] = None
                                sample_data[2] = None

    def fill_buff(self,data, window_size):
        window = scipy.signal.general_gaussian(window_size+1,p=1,sig=(window_size*0.8)) # Do we need this?
        audio_data = np.fromstring(data, dtype=np.float32)
        audio_data = np.append(audio_data, [0])
        self.rb.extend(audio_data * window)

    def get_buffer(self,dim):
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

    @staticmethod
    def random_generator(size=3, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for x in range(size))
        
    @staticmethod
    def maprange(s,a,b): # From https://rosettacode.org/wiki/Map_range
	    (a1, a2), (b1, b2) = a, b
	    return  b1 + ((s - a1) * (b2 - b1) / (a2 - a1))