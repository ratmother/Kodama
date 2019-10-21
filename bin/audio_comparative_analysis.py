import os
import csv
import librosa
import pickle
import fifoutil
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors as KNeighbor
# Simplified and modified to only use PCA with MFCC by ratmother original from https://gist.github.com/fedden/52d903bcb45777f816746f16817698a0#file-dim_reduction_notebook-ipynb
# From the Medium article https://medium.com/@LeonFedden/comparative-audio-analysis-with-wavenet-mfccs-umap-t-sne-and-pca-cb8237bfce2f
# Why PCA and MFCC over wavenet and TSNE? Its faster + I'm using MFCC's already. PCA/MFCC is fast and currently the plan for Kodama is to have the user be able to add
# their own sounds for tailored experiences, or, possible future development utilizing my own recordings. 
# Additional changes by ratmother include: read and write functions, usage of pickle to save PCA data out quickly. 
# As well as ... the addition of a nearest neighbour algorithm, the functionality of real time audio analysis with PCA and nearest neighbour.
# What does it do? 
# https://github.com/mvniemi/librosa_live_tinker , a python script for getting real time audio data into librosa modfied and added to this script.
directory = './data/learning_sounds'
dataset = []
sample_rate = 44100
mfcc_size = 13
read_input = False # Read the saved features.
comp_data = True # Compose input data from a directory into saved features.
buffer_input = True # Input sound from microphone and run the comparative analysis.

# Note: n_components should be tested with different values to see what works best. 
def get_pca(features):
    pca = PCA(n_components=2)
    transformed = pca.fit(features).transform(features)
    scaler = MinMaxScaler()
    scaler.fit(transformed)
    return scaler.transform(transformed)

def get_nearest(pca_coords, n_neigh, rad):  
    neigh = KNeighbor(n_neigh, rad)
    neigh.fit(pca_coords)
    return neigh.kneighbors([pca_coords[-1]], n_neigh, return_distance=False)

def get_features(data_, dataset_, file_):
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
            dataset_ += [(file_, concat_features)]

#   When learning a directory, this should be set to false. It loads the compared data. 
if read_input == True:
    file_in = open("pca_data.pkl",'rb')
    input_pca_data = pickle.load(file_in)
    pca_data = input_pca_data
    file_in.close()
# Here, the feature data is gathered from a directory of sounds and saved with pickle.
elif comp_data == True: 
    for file in os.listdir(directory):
        if file.endswith('.wav') or file.endswith('.Wav'):
            file_path = os.path.join(directory, file)
            try:
                data, _ = librosa.load(file_path)
                get_features(data, dataset, file)
            except:
                print("MFCC/Read error!")
    all_file_paths,mfcc_features = zip(*dataset)
    mfcc_features = np.array(mfcc_features)
    pca_mfcc = get_pca(mfcc_features)
    pca_data = zip(pca_mfcc,all_file_paths)
    file_out = open("pca_data.pkl","wb")
    pickle.dump(pca_data, file_out)
    file_out.close()
    pca_mfcc_string = str(pca_mfcc)
    file_paths_string = str(all_file_paths)
    fifoutil.write_txt(pca_mfcc_string.encode(), "data/pca_out") 
    fifoutil.write_txt(file_paths_string.encode(), "data/sound_names_out") 
    #print(pca_mfcc_string)

# If there is a buffer input, we get its features, apply PCA to it and then nearest neighbour. As a result, we get a selection of similar sounds.
if buffer_input == True:
        try:
            p = pyaudio.PyAudio()

            stream = p.open(input_device_index = 7, format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=1024)

            frames = []

            for i in range (0,10):
                buffer = stream.read(1024, exception_on_overflow = False)
                nfloat = librosa.util.buf_to_float(buffer, 4)
                frames.append(nfloat)

            data = np.concatenate(frames)
            get_features(data, dataset, "INPUT")
            all_file_paths,mfcc_features = zip(*dataset)
            mfcc_features = np.array(mfcc_features)
            pca_mfcc = get_pca(mfcc_features)
            nearest = get_nearest(pca_mfcc, 5, 0.4)
            print (all_file_paths[nearest[0][0]])
            print("The nearest to the above")
            print (all_file_paths[nearest[0][1]])
            print (all_file_paths[nearest[0][2]])
            print (all_file_paths[nearest[0][3]])
            print (all_file_paths[nearest[0][4]])
            print("Ended")

        except:
            print("Buffer error!")   