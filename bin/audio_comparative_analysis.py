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


# See process_learn.py for information regarding the audio feature extraction and machine learning aspects.
# This script handles read and write functions through the usage of pickle to save PCA data out quickly. 
# It also directs all the sound files into process_learn. Whether this be in real time, reading a whole directory of sounds or reading a file of saved features.
# As well as ... the addition of a nearest neighbour algorithm, the functionality of real time audio analysis with PCA and nearest neighbour via librosa and pyaudio.
# Overall script structure and real time input functionality with PCA/NNeighbour by ratmother. See process_learn.py. Designed to communicate with Openframeworks via pipes.
# While comments reference microphone input, there is nothing stopping this input from being from an internal source, other than this being a pain on Ubuntu.
# If your microphone input is low quality the system will act strangely, 
# e.g, a microphone that picks up a lot of bass and has a ringing high pitch will in turn find similar sounds to the *mirophone* instead of the intended sound, so a lot of 808 drums and high pitched synths.

directory = './data/learning_sounds'
output_dir = directory # './data/discovered_sounds'
dataset = []
sample_rate = 44100
mfcc_size = 24     # Resolution of the MFCC extract. 
read_data =  True # Read the saved features.
comp_data = False # Compose input data from a directory into saved features.
buffer_input = read_data # Input sound from microphone and run the comparative analysis.
p = pyaudio.PyAudio()
INDEX = 7
CHANNELS = 1
RATE = 44100
BUFFER = 512
FORMAT = pyaudio.paFloat32  

ringBuffers = []
for x in range(1, 6): # We have n-1 ring buffers, with increasing amounts of length. This enables us to extract features and then run PCA/NN on various lengths of input audio.
    ringBuffers.append(pl.RingBuffer(x * RATE, 0, 0)) # The ring buffer stores audio for real time analysis, x * RATE is x seconds.

# Here, the feature data is gathered from a directory of sounds, it works best if all the sounds levels are normalized. https://github.com/slhck/ffmpeg-normalize.
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
                pl.get_features(data, dataset, file, sample_rate, mfcc_size)
            except:
                print("MFCC/Read error!")
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
                pl.get_saved_features(conv_saved_features[index], dataset, file)
            index += 1 

frames = []
cooldown = 0 # Cooldown for the abnormal sound detection
timer = 0 # Timer is used for pyaudio callback 

def callback(in_data, frame_count, time_info, flag):
    global timer
    global cooldown
    timer += 1
    frames.append(in_data)
    audio_data = np.fromstring(in_data, dtype=np.float32)
    for r in ringBuffers:   # Fill the ring buffers of varying length.
        r.extend(audio_data)
    if timer > 25:
        timer = 0
        nearest_files = []
        nearest_display = []
        # Maybe generate a name using elements from the nearest name?
        # The unique name given to the sound that is coming in, used if it passes the distant check.
        for index, r in enumerate(ringBuffers):
            name = "O_" + str(index) + str(r.itr) + '.wav'
            r.cooldown -= 1
            nearest_pca = []
            pl.get_features(r.get(), dataset, name, sample_rate, mfcc_size) 
            all_file_paths,mfcc_features = zip(*dataset)
            #print(all_file_paths[-1])
            mfcc_features = np.array(mfcc_features)
            pca_mfcc = pl.get_pca(mfcc_features, 5)
            nearest = pl.get_nearest(pca_mfcc, 5, 0.2)
            nearest_files.append(str(nearest[0][1])) # Here the file names are stored to be sent out to Openframeworks
            nearest_display.append(all_file_paths[nearest[0][1]]) # This is just to display it in terminal
            for i in range (1,5):
                nearest_pca.append(pca_mfcc[nearest[0][i]]) # This stores the PCA coordinates of the nearest sounds
            if pl.compare_pca(pca_mfcc[-1], nearest_pca, 0.5) == True and r.cooldown < 0:
                r.cooldown = 20
                r.itr += 1
                file_path = os.path.join(output_dir, name)
                print(file_path)
                write(file_path, RATE, r.get())
                file_out = open("feature_data.pkl","wb") # We've detected an abnormal sound, saved it to our output directory and now we update the saved features.
                pickle.dump(mfcc_features, file_out)
                file_out.close()
            else:
                del dataset[-1] # Deletes the input from the dataset, we've used it for the PCA and nneighbour analysis and it didn't pass the threshold.
        nearest_files = ','.join(nearest_files)
        nearest_files = nearest_files.replace("'", '')
        fifoutil.write_txt(nearest_files.encode(), "data/sound_ids") 
        print("nearest sounds are " + str(nearest_display))
    return (in_data, pyaudio.paContinue)


# This is the real time stream, stream_callback calls the callback function defined above every frame.

if buffer_input == True:

    stream = p.open(input_device_index = INDEX, format= FORMAT,
                channels= CHANNELS,
                rate= RATE,
                input=True,
                frames_per_buffer= BUFFER,
                stream_callback = callback)


    for i in range(p.get_device_count()):
        print p.get_device_info_by_index(i).get('name') # Important to get the name as the default device is often not the microphone.

    print p.get_default_input_device_info()


    stream.start_stream()

    while stream.is_active():
        time.sleep(0.0001)

    stream.close()
    p.terminate()

