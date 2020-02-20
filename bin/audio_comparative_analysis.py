import os
import csv
import time
import random
import string
import soundfile as sf
import sounddevice as sd
import wavio
import scipy.signal
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import numpy as np
import pickle
import librosa
import process_learn as pl
import time
import operator
import random

# print(sd.query_devices())

# TO DO:
# Make prune-immortality dynamic rather than on/off.
# Test accuracy.
# Make features weighted.
# Make distance check dynamic rather than on/off.
# Increase prune chance as dataset gets bigger. 


class seg_analysis ():
    def __init__(self, device_index, pkl_file, plot_results):
      self.pkl_file = pkl_file
      sd.default.device = device_index
      self.sr = sd.query_devices(device_index, 'input')['default_samplerate']
      self.audio = None
      self.timer = 0
      self.dataset,self.input_data,self.nearest,self.segf,self.in_segf= [],[],[],[],[]
      self.prune_chances = [0.5, 0.7, 0.8] # e.g for dim 0 we have a prune chance of 50%, for dim 1 we have a prune chance of 30%.
      self.distance_thresholds = [0.9, 0.9, 0.9] # The distance threshold for different dims can be different, as they are scaled differently. 
      self.num_neighbours = [10, 7, 5] # Number of neighbours for each dim to be checked against.
      self.growth_rates = [0 for x in range(0, 5)]
      self.plot = plot_results
      if self.plot:
            fig, self.axs = plt.subplots(3, figsize=(4,12))

    def segment_dir(self, dir):
        count = 0
        for file in os.listdir(dir):
            if file.endswith('.wav') or file.endswith('.Wav'):
                count += 1
                file_path = os.path.join(dir, file)
                data, _ = librosa.load(file_path)
                pl.get_features(file_path, self.dataset, file, librosa.get_samplerate(file_path), 1)
        try:
            all_file_paths,features = zip(*self.dataset)
        except:
            print("No files in learning sounds.")
            quit()
        file_out = open(self.pkl_file,"wb")
        pickle.dump(features, file_out)
        file_out.close()
        print(count)

    def read_pkl(self, dir):
        features_in = open(self.pkl_file,'rb+')
        conv_saved_features = pickle.load(features_in)
        index = 0
        for file in os.listdir(dir):
            if file.endswith('.wav') or file.endswith('.Wav'):
                file_path = os.path.join(dir, file)
                if index <= len(conv_saved_features):
                    self.dataset += [(file, conv_saved_features[index])]
                index += 1 

    def parcelization(self):
        # Prune samples outside of clusters.
        if(len(self.segf) > 0):
            for dim in range(0, self.seg_range):
                self.basis_prunning(dim)
        # Clear the data buffers, add the remaining input data from the last cycle to the main dataset.
        self.dataset += self.input_data
        self.nearest.clear()
        self.input_data.clear()
        self.in_segf.clear()
        self.segf.clear()
        # Wait for the previous cycle's audio recording to finish, extract the features from it and begin recording new audio for the next cycle.
        if self.audio is None:
            self.audio = sd.rec(int(2 * self.sr), samplerate=self.sr, channels=2, dtype = "float32")
        sd.wait()
        current_name = self.random_generator()
        file_path = os.path.join("./data/", current_name + ".wav")
        wavio.write(file_path, self.audio, self.sr, sampwidth=3)
        self.audio = sd.rec(int(3.0 * self.sr), samplerate=self.sr, channels=2, dtype = "float32")
        pl.get_features(file_path,self.input_data, current_name, self.sr, 0.1)
        os.remove(file_path)
        # Re-configure dataset to be ordered by 'dimension' of the size of the samples.
        self.seg_range = pl.get_segment_amounts()
        self.nearest = [[] for x in range(0, self.seg_range)]
        self.segf = [[] for x in range(0, self.seg_range)] # Segmented feature storage. Creates matrices. The dimensions are [subdivisons x [dataset - none]]
        self.in_segf = [[] for x in range(0, self.seg_range)]
        #
        all_file_paths,features = zip(*self.dataset)
        for index, file in enumerate(features): # Iterate over all the dataset entries. Each entry contains a list of length seg_range, each segment range is called a dim from now on.
            for dim, type in enumerate(file): # In each of the dim's list are two lists pertaining to the type of the contents, either an audio array or feature vector.
                if file[dim] is not None:  # Ignore empty dims, sometimes we do not have information for higher dims because the data did not intially fit into a block of the highest dims size.
                    self.segf[dim].append([type, index]) # We need to save the index along with the data so that we know what dataset entry we are looking at.
        # Do the same process for the input data.
        all_file_paths,features = zip(*self.input_data)
        for index, file in enumerate(features): 
            for dim, type in enumerate(file): 
                if file[dim] is not None: 
                    self.in_segf[dim].append([type, index]) 
        # Save after a certain number of cycles.
        self.timer += 1
        if self.timer > 50:
            self.timer = 0
            file_out = open(self.pkl_file,"wb")
            pickle.dump(features, file_out)
            file_out.close()
            print("Saving out pickle data, do not end script.")

    def seg_dim_analysis(self, dim): 
        saa = [] # Sample-audio-arrays.
        sfv = [] # Sample-feature-vectors.
        # We look through the dataset dimension wise, storing audio and feature data temporarily. 
        for entries in self.segf[dim]: 
            for index, shard in enumerate(entries[0][1]):
                if shard is not None:  
                    saa.append(entries[0][0][index])  
                    sfv.append(entries[0][1][index])  # Append all feature shards in entry in dim. segf[dim][type, index][audio array, feature vector][shards]
        # We look through the input data dimension wise, running cluster analysis on each sample. 
        for i, entries in enumerate(self.in_segf[dim]): 
            if entries is not None:
                for index, shard in enumerate(entries[0][1]): 
                    if shard is not None:
                        # Once we have a sample, we append it to the temporary dimension wise array, if it doesn't pass the cluster analysis check it is removed.
                        save = False
                        saa.append(entries[0][0][index]) 
                        sfv.append(shard)
                        sfv_arr = np.stack(sfv, axis=0) # Converts the list of feature vectors into an array, which is what process_learn needs.
                        try:
                            sfv_pca = pl.get_pca(sfv_arr, 3)
                        except:
                            return False # We keep the shard as PCA should only fail in the case of not having enough samples.
                        try:
                            nearest = pl.get_nearest(sfv_pca, self.num_neighbours[dim], 0.2) 
                        except:
                            return False # We also keep the shard when we do not have enough samples for k-nearest neighbour. 
                        for i in range (1,self.num_neighbours[dim]):
                            dist_check = pl.euclid(sfv_pca[-1], sfv_pca[i], self.distance_thresholds[dim]) # Theory: If we always add shards when our nearest neighbour(s) is too far away, we should make more clusters.
                            if dist_check:
                                save = True # Keeping this simple for now.
                        self.nearest[dim].append(saa[nearest[0][1]])
                        del sfv[-1]
                        del saa[-1]
                        if not save:
                            print(str(dim) + " did not save shard.")
                            os.remove('./data/slices/' + self.in_segf[dim][0][0][0][index])
                            self.in_segf[dim][0][0][2][index] = None
                            self.in_segf[dim][0][0][1][index] = None
                            self.in_segf[dim][0][0][0][index] = None
                        else:
                            print(str(dim) + " saved shard.")

    def basis_prunning(self, dim): 
        # Theory: By running PCA and DBSCAN on a growing sample collection and granting prune immortality to cluster centers, we get a stronger sense of which samples are representative of our growing audio collection.
        saa = []
        sfv = []
        for entries in self.segf[dim]:
            for index, shard in enumerate(entries[0][1]):
               # if shard is not None and entries[0][2][index] is False:  
               if shard is not None:
                    saa.append(entries[0][0][index])  
                    sfv.append(entries[0][1][index])
        if len(sfv) >= (10 + self.growth_rates[dim]):
            self.growth_rates[dim] += 1 
            sfv_arr = np.stack(sfv, axis=0)
            sfv_pca = pl.get_pca(sfv_arr, 2)
            centers = pl.get_cluster_centers(sfv_pca, 3)
            flat_centers = [item for sublist in centers for item in sublist]
            # PLOTTING
            if self.plot:
                self.axs[dim].clear()
                plt.axis([0, 1, 0, 1])
                for index, cords in enumerate(sfv_pca):
                    x = cords[0]
                    y = cords[1]
                    if index in flat_centers:
                        self.axs[dim].scatter(x, y, s = 8.9,  c='blue', alpha = 0.8) 
                        self.axs[dim].scatter(x, y, s = 70.9,  c='red', alpha = 0.2) 
                    else:
                        self.axs[dim].scatter(x, y, s = 8.9,  c='blue') 
                plt.pause(0.01)
            # PRUNNING
            count = 0
            for entries in self.segf[dim]:
                for index, shard in enumerate(entries[0][1]):
                    #if shard is not None and entries[0][3][index] is False: 
                    if shard is not None:
                        count += 1
                        if count in flat_centers:
                            entries[0][2][index] -= self.maprange(len(sfv), (0, 1000), (0.8, 0.10)) # This sample is now less likely to be pruned by the mapped amount (100%, 15%).
                            #entries[0][2][index] -= 0.25 # This shard is now less likely to be pruned (by 0.25% percent).
                          #  print("Strengthing immunity of " + str(entries[0][0][index]) + " in dim " + str(dim))
                        # If we have cluster centers, have an sfv greater than 10, and our prune chance passes a random check.
                        #elif len(sfv) > 10 and random.random() > (self.prune_chances[dim] - len(sfv)*0.001) and len(flat_centers) > 0:
                              #  print("Prunning " + str(entries[0][0][index])+ " in dim " + str(dim))
                        elif(entries[0][2][index] >= random.random()): # If the samples prune immunity is below 0, this will never pass.
                            os.remove('./data/slices/' + entries[0][0][index]) 
                            entries[0][0][index] = None
                            entries[0][1][index] = None
                            entries[0][2][index] = None
            print(str(dim) + " <-dim, prune chance->" + str((self.prune_chances[dim] - len(sfv)*0.001)))

        # flat centers is a list of the indexes of sfv_pca we are keeping, this is shared with saa and indices sp
        # we know that we are                         print(entries[0][0])keeping e.g 173 which happens to be sfv[1,2,6,4] saa[test.wav] and indices[400]
        # we need saa and sfv to be NONE if they are not in centers. 

    def get_buffer(self,dim):
        if len(self.nearest) > 0:
            files = None
            if len(self.nearest[dim]) > 0:
                files = []
                for file in self.nearest[dim]:
                    files.append(file)
                    print(file)
            return files

    @staticmethod
    def random_generator(size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for x in range(size))
        
    @staticmethod
    def maprange(s,a,b): # From https://rosettacode.org/wiki/Map_range
	    (a1, a2), (b1, b2) = a, b
	    return  b1 + ((s - a1) * (b2 - b1) / (a2 - a1))