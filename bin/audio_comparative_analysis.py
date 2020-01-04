import os
import queue
import csv
import time
import random
import string
import queue
import soundfile as sf
import sounddevice as sd
import wavio
import scipy.signal
from scipy.io.wavfile import write
import numpy as np
import pickle
#import fifoutil
import librosa
import process_learn as pl
import time

# See process_learn.py for information regarding the audio feature extraction and machine learning aspects.
# This script handles read and write functions through the usage of pickle to save PCA data out quickly. 
# It also directs all the sound files into process_learn. Whether this be in real time, reading a whole directory of sounds or reading a file of saved features.
# As well as ... the addition of a nearest neighbour algorithm, the functionality of real time audio analysis with PCA and nearest neighbour via librosa and pyaudio.
# Overall script structure and real time input functionality with PCA/NNeighbour by ratmother. See process_learn.py. Designed to communicate with Openframeworks via pipes.
# While comments reference microphone input, there is nothing stopping this input from being from an internal source, other than this being a pain on Ubuntu.
# If your microphone input is low quality the system will act strangely, 
# e.g, a microphone that picks up a lot of bass and has a ringing high pitch will in turn find similar sounds to the *mirophone* instead of the intended sound, so a lot of 808 drums and high pitched synths.

INDEX = 1
CHANNELS = 1
RATE = 44100
BUFFER = 8448
RBLENGTH = 2

directory = './data/learning_sounds'
global dataset 
dataset = []
# min_thrsh is the minimum Threshold for current-PCA-to-nearest-PCAs comparison, see process_learn.py and below in callback.
# A smaller value will cause more files to be created as every time the threshold is past, we save the audio buffer to a wav file along with that buffer's MFCC data.
# Hopefully a convergence of sorts of the environment-stimuli memory happens at lower values of min_thrsh, as the system will store new sounds until there are no new sounds.
# However, if the min_thrsh is too low the system will just reflect sounds it just heard, as ALL sounds are considered new at extremely low thrsh.
# At very low values the system becomes Funes the Memorious. If compare_pca never reaches a stable state when expected, turn min_thrsh up.


q = queue.Queue()

class seg_analysis ():
    def __init__(self, device_index, ringbuffer_length, buffer_size, min_thrsh, pkl_file):
      self.dataset = []
      self.device_index = device_index
      self.buffer_size = buffer_size
      self.min_thrsh = min_thrsh
      self.pkl_file = pkl_file
      self.ringBuffer =  pl.RingBuffer(ringbuffer_length * 44100, 0, self.min_thrsh) # The ring buffer stores audio for real time analysis.
      self.q = queue.Queue()
      self.thresholds = []
      self.timer = 0
      self.sr= 44100
      self.num_comp = 3
      self.segf = []
      self.in_segf = []
      self.all_file_paths = None
      self.file_path_temp = None
      self.input_data = []
      self.current_name = "nun"
      self.nearest = []
      self.thresh_limit = 0.3
    def segment_dir(self, dir):
        self.dir = dir
        count = 0
        for file in os.listdir(self.dir):
            if file.endswith('.wav') or file.endswith('.Wav'):
                count += 1
                file_path = os.path.join(self.dir, file)
                data, _ = librosa.load(file_path)
                pl.get_features(file_path, self.dataset, file, librosa.get_samplerate(file_path))
        all_file_paths,features = zip(*self.dataset)
        file_out = open(self.pkl_file,"wb")
        pickle.dump(features, file_out)
        file_out.close()
        print(count)
    def read_pkl(self):
        features_in = open(self.pkl_file,'rb+')
        conv_saved_features = pickle.load(features_in)
        index = 0
        for file in os.listdir(directory):
            if file.endswith('.wav') or file.endswith('.Wav'):
                file_path = os.path.join(directory, file)
                if index <= len(conv_saved_features):
                    self.dataset += [(file, conv_saved_features[index])]
                index += 1 
    def callback(self, in_data, frame_count, time_info, status):
        window = scipy.signal.general_gaussian(self.buffer_size + 1, p=1, sig= (self.buffer_size * 0.8) )
        audio_data = np.fromstring(in_data, dtype=np.float32)
        audio_data = np.append(audio_data, [0])
        self.ringBuffer.extend(audio_data * window)
        q.put(in_data[:,0])
    def random_generator(self, size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for x in range(size))
    def parcelization(self):
        for dim in self.in_segf:
            for entries in dim[0][0][1]:
                if entries is not None:
                    self.dataset += self.input_data
                    break
        self.nearest.clear()
        self.input_data.clear()
        self.in_segf.clear()
        self.segf.clear()
        if self.file_path_temp is not None:
            os.remove(self.file_path_temp)
        for x in range(pl.get_segment_amounts() - 1):
            self.thresholds.append(2**(x + 1) * 0.1)
        t0 = time.time()
        self.timer += 1
        name = self.random_generator()
        self.current_name = name
        file_path = os.path.join("./data/", name + ".wav")
        self.file_path_temp = file_path
        print(file_path)
        print(len(self.dataset))
        wavio.write(file_path, self.ringBuffer.get(), self.sr, sampwidth=3)
        pl.get_features(file_path,self.input_data, name, self.sr)
        self.seg_range = pl.get_segment_amounts() - 1

        all_file_paths,features = zip(*self.dataset)
        self.segf = []
        for x in range(self.seg_range): # Creates matrices. The dimensions are [subdivisons x [dataset - none]]
            self.segf.append([]) # Segmented feature storage. Includes indexes and None's.
        for index, file in enumerate(features): # Iterate over all the dataset entries. Each entry contains a list of length seg_range, each segment range is called a dim from now on.
            for dim, type in enumerate(file): # In each of the dim's list are two lists pertaining to the type of the contents, either an audio array or feature vector.
                if file[dim] is not None:  # Ignore empty dims, sometimes we do not have information for higher dims because the data did not intially fit into a block of the highest dims size.
                    self.segf[dim].append([type, index]) # We need to save the index along with the data so that we know what dataset entry we are looking at.

        all_file_paths,features = zip(*self.input_data)
        self.in_segf = []
        for x in range(self.seg_range): 
            self.in_segf.append([]) #
        for index, file in enumerate(features): 
            for dim, type in enumerate(file): 
                if file[dim] is not None: 
                    self.in_segf[dim].append([type, index]) 

        if self.timer > 100:
            self.timer = 0
            file_out = open(self.pkl_file,"wb") # We've detected an abnormal sound, saved it to our output directory and now we update the saved features.
            pickle.dump(features, file_out)
            file_out.close()
            print("Saving out pickle data, do not end script.")

    def seg_dim_analysis(self, dim):  # for dim in range(seg_range): # For every segment size or dim we find the nearest neighbour sound files to the ringbuffer.
        saa = []
        sfv = []
        indexes = [] # Keeps track of each shards dataset index.
        all_file_paths,features = zip(*self.dataset)
        for entries in self.segf[dim]: 
            for index, shard in enumerate(entries[0][1]):
                if shard is not None:  
                    indexes.append(entries[1])
                    saa.append(entries[0][0][index])  
                    sfv.append(shard) # Append all feature shards in entry in dim. segf[dim][type, index][audio array, feature vector][shards]
        for i, entries in enumerate(self.in_segf[dim]): 
            if entries is not None:
                for j, shard in enumerate(entries[0][1]): 
                    if shard is not None:
                        seg_diffs = []
                        nearest_pca = []
                        indexes.append(entries[1])
                        saa.append(entries[0][0][j]) 
                        sfv.append(shard)
                        sfv_arr = np.stack(sfv, axis=0) # Converts the list of feature vectors into an array, which is what process_learn wants
                        try:
                            sfv_pca = pl.get_pca(sfv_arr, self.num_comp)
                        except: 
                            print("FAIL -> for current dim: not enough samples of this size..." + self.in_segf[dim][0][0][0][j])
                            break
                        num_neighbours = int(10/(dim+1))
                        nearest = pl.get_nearest(sfv_pca, num_neighbours, 0.2)
                        for i in range (1,num_neighbours):
                            file_index = indexes[nearest[0][i]]  
                            nearest_pca.append(sfv_pca[nearest[0][i]]) # This stores the PCA coordinates of the nearest shards.
                            seg_diffs.append(pl.compare_pca(sfv_pca[-1], nearest_pca, self.thresholds[dim])) #  Euclidean distance.
                            self.nearest.append(saa[nearest[0][i]])
                            #  print("nn for rb in dim " + str(dim) + " is: " + str(saa[nearest[0][i]]))
                        sum_diffs = sum(seg_diffs)
                        if sum_diffs < self.thresholds[dim]: # If the distances of the nearest neigbours to the current shard is large enough we keep the shard, if not we remove it. 
                            print("FAIL -> dim: " + str(dim) + " w/ thrsh: " + str(self.thresholds[dim]) + " dist: " + str(sum_diffs) + self.in_segf[dim][0][0][0][j])
                            self.in_segf[dim][0][0][1][j] = None
                            if self.thresholds[dim] > self.thresh_limit:
                                self.thresholds[dim] -=  (0.05 * (1+ dim))**2
                            os.remove('./data/slices/' + self.in_segf[dim][0][0][0][j])
                        else:
                            self.thresholds[dim] +=  (0.1 * (1+ dim))**2
                            print("PASS -> dim: " + str(dim) + " w/ thrsh: " + str(self.thresholds[dim]) + " dist: " + str(sum_diffs) + " " + self.in_segf[dim][0][0][0][j])
                        del sfv[-1]
                        del saa[-1]
    def run_analysis(self):
        device_info = sd.query_devices(self.device_index, 'input')
        samplerate = device_info['default_samplerate']
        stream = sd.InputStream(device = self.device_index,
                            blocksize = self.buffer_size,
                            samplerate = samplerate,
                            channels= 1,
                            dtype = "float32",
                            callback = self.callback,
                            latency = 0.2)
        with stream:
            self.parcelization()
    def get_buffer(self):
        files = []
        for near in self.nearest:
            files.append(near)
        return files

