import os
import csv
import pickle
import fifoutil
import librosa
import process_learn as pl
import numpy as np
import pyaudio
import time
import matplotlib.pyplot as plt
import wave
from scipy.io.wavfile import write
import wavio
import scipy.signal

# See process_learn.py for information regarding the audio feature extraction and machine learning aspects.
# This script handles read and write functions through the usage of pickle to save PCA data out quickly. 
# It also directs all the sound files into process_learn. Whether this be in real time, reading a whole directory of sounds or reading a file of saved features.
# As well as ... the addition of a nearest neighbour algorithm, the functionality of real time audio analysis with PCA and nearest neighbour via librosa and pyaudio.
# Overall script structure and real time input functionality with PCA/NNeighbour by ratmother. See process_learn.py. Designed to communicate with Openframeworks via pipes.
# While comments reference microphone input, there is nothing stopping this input from being from an internal source, other than this being a pain on Ubuntu.
# If your microphone input is low quality the system will act strangely, 
# e.g, a microphone that picks up a lot of bass and has a ringing high pitch will in turn find similar sounds to the *mirophone* instead of the intended sound, so a lot of 808 drums and high pitched synths.

INDEX = 7
CHANNELS = 1
RATE = 44100
BUFFER = 8192

directory = './data/learning_sounds'
dataset = []
mfcc_size = 24

# min_thrsh is the minimum Threshold for current-PCA-to-nearest-PCAs comparison, see process_learn.py and below in callback.
# A smaller value will cause more files to be created as every time the threshold is past, we save the audio buffer to a wav file along with that buffer's MFCC data.
# Hopefully a convergence of sorts of the environment-stimuli memory happens at lower values of min_thrsh, as the system will store new sounds until there are no new sounds.
# However, if the min_thrsh is too low the system will just reflect sounds it just heard, as ALL sounds are considered new at extremely low thrsh.
# At very low values the system becomes Funes the Memorious. If compare_pca never reaches a stable state when expected, turn min_thrsh up.

min_thrsh = 0.08

# num_neighbours is the amount of nearest neighbours to access, see process_learn.py. More NN's means more distances to be averaged together and compared to the input's PCA.

num_neighbours = 5

read_data =  True # Read the saved features.
comp_data = not read_data # Compose input data from a directory into saved features.
buffer_input = read_data # Input sound from microphone and run the comparative analysis.


# Here, the feature data is gathered from a direis, x * BUFFER is x seconds.ctory of sounds, it works best if all the sounds levels are normalized. https://github.com/slhck/ffmpeg-normalize.
# comp_data processes the feature data of each sound file, taking some time.
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
                pl.get_features(data, dataset, file, RATE, mfcc_size)
            except:
                print("Read error!")
    all_file_paths,mfcc_features = zip(*dataset)
    file_out = open("feature_data.pkl","wb")
    pickle.dump(mfcc_features, file_out)
    file_out.close()
    mfcc_features = np.array(mfcc_features)
    pca_mfcc = pl.get_pca(mfcc_features, 5)
    pca_data = zip(pca_mfcc,all_file_paths)
    file_paths_string = ','.join(all_file_paths)
    file_paths_string = file_paths_string.replace("'", '')
    file_paths_string = file_paths_string.strip()
    fifoutil.write_txt(file_paths_string.encode(), "data/sound_names_out") 
    print(count)

# After comp_data is done, a .pkl file contains all the feature data of the dataset, and can be accessed quickly via pickle below.
# Its important that if there are any changes to the content of the sound directory, that comp_data be re-ran.

if read_data == True:
    file_in = open("feature_data.pkl",'rb+')
    conv_saved_features = pickle.load(file_in)
    conv_saved_features = np.array(conv_saved_features)
    index = 0
    for file in os.listdir(directory):
        if file.endswith('.wav') or file.endswith('.Wav'):
            file_path = os.path.join(directory, file)
            if index <= len(conv_saved_features):
                pl.get_saved_features(conv_saved_features[index], dataset, file)
            index += 1 

timer = 0 # Timer is used for pyaudio callback 

ringBuffers = []
for x in range(1, 3): # We have n-1 ring buffers, with increasing amounts of length. This enables us to extract features and then run PCA/NN on various lengths of input audio.
    ringBuffers.append(pl.RingBuffer(x * RATE, 0, min_thrsh)) # The ring buffer stores audio for real time analysis.

# Change the incoming sound to sound like its nearest sounds... maybe repitch the incoming sound to fit its near pitches, and tempo

def callback(in_data, frame_count, time_info, flag):
    global timer
    timer += 1
    window = scipy.signal.general_gaussian(BUFFER + 1, p=1, sig= (BUFFER * 0.8) )
    window = window
    audio_data = np.fromstring(in_data, dtype=np.float32)
    audio_data = np.append(audio_data, [0])
    for r in ringBuffers:   # Fill the ring buffers of varying length.
        r.extend(audio_data * window)
    if timer > 10:
        timer = 0
        nearest_files = []
        nearest_display = []
        for index, r in enumerate(ringBuffers):
            name = "Ye_" + str(index) + str(r.itr) + '.wav'
            nearest_pca = []
            pl.get_features(r.get(), dataset, name, RATE, mfcc_size) 
            all_file_paths,mfcc_features = zip(*dataset)
            mfcc_features = np.array(mfcc_features)
            pca_mfcc = pl.get_pca(mfcc_features, 5)
            nearest = pl.get_nearest(pca_mfcc, num_neighbours, 0.2)
            for i in range (1,num_neighbours):
                nearest_files.append(str(nearest[0][i])) # Here the file names are stored to be sent out to Openframeworks
                nearest_display.append(all_file_paths[nearest[0][i]]) # This is just to display it in terminal
                nearest_pca.append(pca_mfcc[nearest[0][i]]) # This stores the PCA coordinates of the nearest sounds
            distance = pl.compare_pca(pca_mfcc[-1], nearest_pca, r.thrsh) #  Euclidean distance. 
            if distance > r.thrsh:
                r.thrsh += 0.005 / distance
                r.itr += 1
                file_path = os.path.join(directory, name)
                print("P: " + str(index) + " -> r.thrsh: " + str(r.thrsh) + " dist: " + str(distance))
                wavio.write(file_path, r.get(), RATE, sampwidth=3) # This seems to work.
                file_out = open("feature_data.pkl","wb") # We've detected an abnormal sound, saved it to our output directory and now we update the saved features.
                pickle.dump(mfcc_features, file_out)
                file_out.close()
            else:
                if r.thrsh > min_thrsh:  # When the threshold is reached we increase the buffer's threshold (below, in next if statement), this if statement causes the buffer's threshold to decay back down to the min_thrsh.
                    r.thrsh -= 0.05 * distance
                print("F: " + str(index) + " -> r.thrsh: " + str(r.thrsh) + " dist: " + str(distance))
                del dataset[-1] # Deletes the input from the dataset, we've used it for the PCA and nneighbour analysis and it didn't pass the threshold.
        file_paths_string = ','.join(all_file_paths)
        file_paths_string = file_paths_string.replace("'", '')
        file_paths_string = file_paths_string.strip()
        fifoutil.write_txt(file_paths_string.encode(), "data/sound_names_out") 
        nearest_files = ','.join(nearest_files)
        nearest_files = nearest_files.replace("'", '')
        fifoutil.write_txt(nearest_files.encode(), "data/sound_ids") 
        #print("nearest sounds are " + str(nearest_display))
    return (in_data, pyaudio.paContinue)


p = pyaudio.PyAudio()
FORMAT = pyaudio.paFloat32  

# This is the real time stream, stream_callback calls the callback function defined above every frame.

if buffer_input == True:
    stream = p.open(input_device_index = INDEX, format= FORMAT,
                channels= CHANNELS,
                rate= RATE,
                input=True,
                frames_per_buffer= BUFFER,
                stream_callback = callback)

    #for i in range(p.get_device_count()):
    #    print p.get_device_info_by_index(i).get('name') # Important to get the name as the default device is often not the microphone.

    #print p.get_default_input_device_info()

    stream.start_stream()

    while stream.is_active():
        time.sleep(0.0001)

    stream.close()
    p.terminate()

