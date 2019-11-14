import os
import csv
import librosa
import pickle
import fifoutil
import numpy as np
import pyaudio
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors as KNeighbor
import wave

# https://scimusing.wordpress.com/2013/10/25/ring-buffers-in-pythonnumpy/
class RingBuffer():
    "A 1D ring buffer using numpy arrays"
    def __init__(self, length):
        self.data = np.zeros(length, dtype='f')
        self.index = 0

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
# Simplified and modified script to only use PCA with MFCC by ratmother, original from https://gist.github.com/fedden/52d903bcb45777f816746f16817698a0#file-dim_reduction_notebook-ipynb
# From the Medium article https://medium.com/@LeonFedden/comparative-audio-analysis-with-wavenet-mfccs-umap-t-sne-and-pca-cb8237bfce2f
# Why PCA and MFCC over wavenet and TSNE? Its faster + I'm using MFCC's already. PCA/MFCC is fast and currently the plan for Kodama is to have the user be able to add
# their own sounds for tailored experiences, or, possible future development utilizing my own recordings. 
# Additional changes by ratmother include: read and write functions, usage of pickle to save PCA data out quickly. 
# As well as ... the addition of a nearest neighbour algorithm, the functionality of real time audio analysis with PCA and nearest neighbour.
# Another script used was https://github.com/mvniemi/librosa_live_tinker , a python script for getting real time audio data into librosa modfied and added to this script.
# Script strcture, functions and real time input functionality with PCA/NNeighbour by ratmother. 
# While comments reference microphone input, there is nothing stopping this input from being from an internal source. 
p = pyaudio.PyAudio()
directory = './data/learning_sounds'
dataset = []
sample_rate = 44100
mfcc_size = 13
read_data = True # Read the saved features.
comp_data = False # Compose input data from a directory into saved features.
buffer_input = True # Input sound from microphone and run the comparative analysis.
# Note: n_components should be tested with different values to see what works best. 
def get_pca(features):
    pca = PCA(n_components= 5)
    transformed = pca.fit(features).transform(features)
    scaler = MinMaxScaler()
    scaler.fit(transformed)
    return scaler.transform(transformed)

def get_nearest(pca_coords, n_neigh, rad):  
    neigh = KNeighbor(n_neigh, rad)
    neigh.fit(pca_coords)
    print(pca_coords[-1])
    return neigh.kneighbors([pca_coords[-1]], n_neigh, return_distance=False) # -1 is the newest addition, which in this case is real time microphone input.

def get_features(data_, dataset_, file_):
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
            #print("features saved from " + file_)
            #print(concat_features)
            dataset_ += [(file_, concat_features)]

# Note: If anything changes about the content in the directory of sounds, you must run get_features again before get_saved_features.
def get_saved_features(saved_features, dataset_, file_):
            print(file_)
            print(saved_features)
            dataset_ += [(file_, saved_features)]


# Here, the feature data is gathered from a directory of sounds. This processes the feature data of each sound file, taking some time.
# We also save out the names in the correct index order so that Openframeworks knows which sound the resulting indexes point to.
# This is important as the real time analysis of nearest neighbour gives us the nearest sounds in the form of indexes. 
if comp_data == True: 
    count = 0
    for file in os.listdir(directory):
        if file.endswith('.wav') or file.endswith('.Wav'):
            count += 1
            file_path = os.path.join(directory, file)
            try:
                data, _ = librosa.load(file_path)
                get_features(data, dataset, file)
            except:
                print("MFCC/Read error!")
    all_file_paths,mfcc_features = zip(*dataset)
    file_out = open("feature_data.pkl","wb")
    pickle.dump(mfcc_features, file_out)
    file_out.close()
    mfcc_features = np.array(mfcc_features)
    pca_mfcc = get_pca(mfcc_features)
    pca_data = zip(pca_mfcc,all_file_paths)
    file_paths_string = ','.join(all_file_paths)
    file_paths_string = file_paths_string.replace("'", '')
    file_paths_string = file_paths_string.strip()
    #print(file_paths_string)
    print(count)
    fifoutil.write_txt(file_paths_string.encode(), "data/sound_names_out") 

# After comp_data is done, a .pkl file contains all the feature data of the dataset, and can be accessed quickly via pickle below.
# Its important that if there are any changes to the content of the sound directory, that comp_data be re-ran.
if read_data == True:
    file_in = open("feature_data.pkl",'rb')
    conv_saved_features = pickle.load(file_in)
    conv_saved_features = np.array(conv_saved_features)
    index = 0
    for file in os.listdir(directory):
        if file.endswith('.wav') or file.endswith('.Wav'):
            file_path = os.path.join(directory, file)
            if index <= len(conv_saved_features):
                get_saved_features(conv_saved_features[index], dataset, file)
            index += 1 

ringBuffer = RingBuffer(5 * 44100)
timer = 0

def callback(in_data, frame_count, time_info, flag):
    global timer
    audio_data = np.fromstring(in_data, dtype=np.float32)
    timer += 1
    ringBuffer.extend(audio_data)
    if timer > 10:
        timer = 0
        get_features(ringBuffer.get(), dataset, "INPUT") # Input is the name of the data chunk, whereas all other 'data chunks' have file .wav names.
        all_file_paths,mfcc_features = zip(*dataset)
        mfcc_features = np.array(mfcc_features)
        pca_mfcc = get_pca(mfcc_features)
        nearest_files = []
        nearest_display = []
        nearest = get_nearest(pca_mfcc, 5, 0.2)
        for i in range (0,5): # We ignore first entry as this is just the input, as the input is nearest to itself.
            nearest_files.append(str(nearest[0][i]))
            nearest_display.append(all_file_paths[nearest[0][i]])
        nearest_files = ','.join(nearest_files)
        nearest_files = nearest_files.replace("'", '')
        del dataset[-1] # Deletes the input from the dataset, we've used it for the PCA and nneighbour analysis and don't need it there anymore.
        fifoutil.write_txt(nearest_files.encode(), "data/sound_ids") 
        print("nearest sounds are " + str(nearest_display))
    return (in_data, pyaudio.paContinue)


CHUNK = 1024
INDEX = 7
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 1
FORMAT = pyaudio.paFloat32

if buffer_input == True:
    stream = p.open(input_device_index = INDEX, format= FORMAT,
                channels= CHANNELS,
                rate= RATE,
                input=True,
                frames_per_buffer= CHUNK, 
                stream_callback = callback)


    for i in range(p.get_device_count()):
        print p.get_device_info_by_index(i).get('name')

    print p.get_default_input_device_info()


# start the stream
    stream.start_stream()

    while stream.is_active():
        time.sleep(0.25)

    stream.close()
    p.terminate()