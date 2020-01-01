import os
import csv
import pickle
import fifoutil
import librosa
import process_learn as pl
import numpy as np
import time
import wave
from scipy.io.wavfile import write
import wavio
import scipy.signal
import random
import string
import soundfile as sf
import queue
import sounddevice as sd

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
BUFFER = 8448
RBLENGTH = 2

directory = './data/learning_sounds'
dataset = []

# min_thrsh is the minimum Threshold for current-PCA-to-nearest-PCAs comparison, see process_learn.py and below in callback.
# A smaller value will cause more files to be created as every time the threshold is past, we save the audio buffer to a wav file along with that buffer's MFCC data.
# Hopefully a convergence of sorts of the environment-stimuli memory happens at lower values of min_thrsh, as the system will store new sounds until there are no new sounds.
# However, if the min_thrsh is too low the system will just reflect sounds it just heard, as ALL sounds are considered new at extremely low thrsh.
# At very low values the system becomes Funes the Memorious. If compare_pca never reaches a stable state when expected, turn min_thrsh up.

min_thrsh = 0.08
auto_save_timer = 0
# num_neighbours is the amount of nearest neighbours to access, see process_learn.py. More NN's means more distances to be averaged together and compared to the input's PCA.
# num_comps is the number of components in the PCA
num_neighbours = 5
num_comps = 3

read_data =  True # Read the saved features.
comp_data = not read_data # Compose input data from a directory into saved features.
buffer_input = read_data # Input sound from microphone and run the comparative analysis.


# Here, the feature data is gathered from a direis, x * BUFFER is x seconds.ctory of sounds, it works best if all the sounds levels are normalized. https://github.com/slhck/ffmpeg-normalize.
# comp_data processes the feature data of each sound file, taking some time.
# We also save out the names in the correct index order so that Openframeworks knows which sound the resulting indexes point to.
# This is important as the real time analysis of nearest neighbour gives us the nearest sounds in the form of indexes.

pl.set_granulation_analysis(256, 256, 2 ** 9,  2 ** 4)

if comp_data == True: 
    count = 0
    for file in os.listdir(directory):
        if file.endswith('.wav') or file.endswith('.Wav'):
            count += 1
            file_path = os.path.join(directory, file)
            data, _ = librosa.load(file_path)
            pl.get_features(file_path, dataset, file, librosa.get_samplerate(file_path))
    all_file_paths,features = zip(*dataset)
    file_out = open("feature_data.pkl","wb")
    pickle.dump(features, file_out)
    file_out.close()
    #features = np.array(features)
    file_paths_string = ','.join(all_file_paths)
    file_paths_string = file_paths_string.replace("'", '')
    file_paths_string = file_paths_string.strip()
    fifoutil.write_txt(file_paths_string.encode(), "data/sound_names_out") 
    print(count)

# After comp_data is done, a .pkl file contains all the feature data of the dataset, and can be accessed quickly via pickle below.
# Its important that if there are any changes to the content of the sound directory, that comp_data be re-ran.

if read_data == True:
    features_in = open("feature_data.pkl",'rb+')
    conv_saved_features = pickle.load(features_in)
    index = 0
    for file in os.listdir(directory):
        if file.endswith('.wav') or file.endswith('.Wav'):
            file_path = os.path.join(directory, file)
            if index <= len(conv_saved_features):
                dataset += [(file, conv_saved_features[index])]
            index += 1 
 

ringBuffer = pl.RingBuffer(RBLENGTH * RATE, 0, min_thrsh) # The ring buffer stores audio for real time analysis.

def random_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))

q = queue.Queue()

def callback(in_data, frame_count, time_info, status):
    window = scipy.signal.general_gaussian(BUFFER + 1, p=1, sig= (BUFFER * 0.8) )
    audio_data = np.fromstring(in_data, dtype=np.float32)
    audio_data = np.append(audio_data, [0])
    ringBuffer.extend(audio_data * window)
    q.put(in_data[:,0])


def parcelization(timer):
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            print("Queue is empty.")
            break
        timer += 1
        name = './data/silly' + '.wav' # Why must we convert a perfectly nice audio array to wav for librosa to accept it into stream?
        wavio.write(file_path, ringBuffer.get(), RATE, sampwidth=3)
        pl.get_features(file_path,dataset,random_generator(), RATE)
        seg_range = pl.get_segment_amounts() - 1
        all_file_paths,features = zip(*dataset)
        nearest_files = []
        nearest_pca = []
        segf = []  
        segs = []
        overalldiff = []
        for x in range(seg_range): # Creates matrices. The dimensions are [subdivisons x [dataset - none]]
            segf.append([]) # Segmented feature storage. Includes indexes and None's.
            segs.append([])

        for index, file in enumerate(features): # Iterate over all the dataset entries. Each entry contains a list of length seg_range, each segment range is called a dim from now on.
            for dim, type in enumerate(file): # In each of the dim's list are two lists pertaining to the type of the contents, either an audio array or feature vector.
                if file[dim] is not None:  # Ignore empty dims, sometimes we do not have information for higher dims because the data did not intially fit into a block of the highest dims size.
                    segf[dim].append([type, index]) # We need to save the index along with the data so that we know what dataset entry we are looking at.
        
        for dim in range(seg_range): # For every segment size or dim we find the nearest neighbour sound files to the ringbuffer.
            sfv = [] # Shardwise feature vectors.
            saa = [] # Shardwise audio arrays.
            indexes = [] # Keeps track of each shards dataset index.
            seg_diffs = []
            nearest_display = []
            for entries in segf[dim]: 
                for shards in entries[0][0]:
                    saa.append(shards)  
                for shards in entries[0][1]:
                    indexes.append(entries[1])
                    sfv.append(shards) # Append all feature shards in entry in dim. segf[dim][type, index][audio array, feature vector][shards]
            sfv_arr = np.stack(sfv, axis=0) # Converts the list of feature vectors into an array, which is what process_learn wants.
            sfv_pca = pl.get_pca(sfv_arr, num_comps)
            nearest = pl.get_nearest(sfv_pca, num_neighbours, 0.2)
            for i in range (1,num_neighbours):
                file_index = indexes[nearest[0][i]] 
                if file_index == len(features):
                    print("the nearest is itself...")
                    continue
                namej = "data/" + str(dim) + str(i) + '.wav'
                sf.write(namej,saa[nearest[0][i]], 44100)
                nearest_files.append(str(file_index)) # Here the file names are stored index-wise to be sent out to Openframeworks
                nearest_display.append(all_file_paths[file_index]) # This is just to display it in terminal.
                nearest_pca.append(sfv_pca[nearest[0][i]]) # This stores the PCA coordinates of the nearest shards.
                seg_diffs.append(pl.compare_pca(sfv_pca[-1], nearest_pca, ringBuffer.thrsh)) #  Euclidean distance. 
                overalldiff.append(pl.compare_pca(sfv_pca[-1], nearest_pca, ringBuffer.thrsh)) #  Euclidean distance. 
                print("nn for rb in dim " + str(dim) + " is: " + str(all_file_paths[file_index]) + " at slice " + str(nearest[0][i]) + " of audio length " + str(saa[nearest[0][i]].size) + " and feat vec of length " + str(sfv[nearest[0][i]].size))
            sum_diffs = sum(seg_diffs)
        sum_diffs = sum(overalldiff)
        if sum_diffs > ringBuffer.thrsh:
            ringBuffer.thrsh += 0.5 
            ringBuffer.itr += 1
            if(timer > 60):
                timer = 0
                print("Saving out pickle data, do not end script.")
                file_out = open("feature_data.pkl","wb") # We've detected an abnormal sound, saved it to our output directory and now we update the saved features.
                pickle.dump(features, file_out)
                file_out.close()
            print("PASS -> r.thrsh: " + str(ringBuffer.thrsh) + " dist: " + str(sum_diffs))
        else:
            if ringBuffer.thrsh > min_thrsh:  # This if statement causes the buffer's threshold to decay back down to the min_thrsh.
                ringBuffer.thrsh -= 0.5
                print( "FAIL -> r.thrsh: " + str(ringBuffer.thrsh) + " dist: " + str(sum_diffs))
            del dataset[-1] # Deletes the input from the dataset, we've used it for the PCA and nneighbour analysis and it didn't pass the threshold.
        file_paths_string = ','.join(all_file_paths)
        file_paths_string = file_paths_string.replace("'", '')
        file_paths_string = file_paths_string.strip()
        fifoutil.write_txt(file_paths_string.encode(), "data/sound_names_out") 
        nearest_files = ','.join(nearest_files)
        nearest_files = nearest_files.replace("'", '')
        fifoutil.write_txt(nearest_files.encode(), "data/sound_ids") 
        #print("nearest sounds are " + str(nearest_display))
        #return (in_data, pyaudio.paContinue)


#p = pyaudio.PyAudio()
#FORMAT = pyaudio.paFloat32  

# This is the real time stream, stream_callback calls the callback function defined above every frame.

if buffer_input == True:
    while True:
        device_info = sd.query_devices(INDEX, 'input')
        samplerate = device_info['default_samplerate']

        stream = sd.InputStream(device = INDEX,
                    blocksize = BUFFER,
                    samplerate = samplerate,
                    channels= CHANNELS,
                    dtype = "float32",
                    callback = callback,
                    latency = 2.0)
        #sd.sleep(int(1 * 1000))
        #for i in range(p.get_device_count()):
        #    print p.get_device_info_by_index(i).get('name') # Important to get the name as the default device is often not the microphone.

        #print p.get_default_input_device_info()
        with stream:
            parcelization(auto_save_timer)

   # while stream.is_active():
    ##    time.sleep(0.0001)

    #stream.close()
    #p.terminate()
