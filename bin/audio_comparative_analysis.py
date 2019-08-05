import os
import csv
import librosa
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen
#Simplified and modified to only use PCA with MFCC by ratmother original from https://gist.github.com/fedden/52d903bcb45777f816746f16817698a0#file-dim_reduction_notebook-ipynb
#From the Medium article https://medium.com/@LeonFedden/comparative-audio-analysis-with-wavenet-mfccs-umap-t-sne-and-pca-cb8237bfce2f
#Why PCA and MFCC over wavenet and TSNE? Its faster + I'm using MFCC's already. PCA/MFCC is fast and currently the plan for Kodama is to have the user be able to add
#their own sounds, or, possible future development utilizing recordings.
#Additional changes include: read and write functions, usage of pickle to save PCA data out quickly. 

def get_pca(features):
    pca = PCA(n_components=2)
    transformed = pca.fit(features).transform(features)
    scaler = MinMaxScaler()
    scaler.fit(transformed)
    return scaler.transform(transformed)


directory = './Drums/'
dataset = []
sample_rate = 44100
mfcc_size = 13
read_input = False

if read_input == True:
    file_in = open("pca_data.pkl",'rb')
    input_pca_data = pickle.load(file_in)
    pca_data = input_pca_data
    file_in.close()
else: 
    for file in os.listdir(directory):

        if file.endswith('.wav') or file.endswith('.Wav'):
            print("Loaded " + file) 
            file_path = os.path.join(directory, file)
            try:
                data, _ = librosa.load(file_path)
                trimmed_data, _ = librosa.effects.trim(y=data)
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
                dataset += [(file, concat_features)]

            except:
                print("MFCC/Read error!")
    all_file_paths,mfcc_features = zip(*dataset)
    mfcc_features = np.array(mfcc_features)
    pca_mfcc = get_pca(mfcc_features)
    pca_data = zip(pca_mfcc,all_file_paths)
    file_out = open("pca_data.pkl","wb")
    pickle.dump(pca_data, file_out)
    file_out.close()


for a, b in pca_data:
    print a, b

