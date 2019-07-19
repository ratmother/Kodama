import keras
import numpy as np
import librosa
import fifoutil

'''
Edited for realtime functionality in Openframeworks by ratmother.
Uncomment the #print for debugging.
'''

class livePredictions:

    def __init__(self, path):

        self.path = path

    def load_model(self):

        self.loaded_model = keras.models.load_model(self.path)
        return self.loaded_model.summary()

    def makepredictions(self):
            m = None
            try:
                m = fifoutil.read_txt("data/mfcc")
            except:
                #print("Failed to read MFCC data!")
                pass #Can change to break if required but data occasionally gets corrupted.
            if m is not None:
               #print("MFCC data has been read.")
               #print("Python reads...", m, " which is a ", type(m))
               m = np.fromstring(m, dtype=float, sep=',')
               #print("Python converted the input to", type(m))
               #print(m)
               x = np.expand_dims(m, axis=2)
               x = np.expand_dims(x, axis=0)
               predictions = self.loaded_model.predict_classes(x)
               fifoutil.write_txt(self.convertclasstoemotion(predictions).encode(), "data/emotion_voice")
               print( "Prediction is", " ", self.convertclasstoemotion(predictions))
        

    def convertclasstoemotion(self, pred):
        '''
        I am here to convert the predictions (int) into human readable strings.
        '''
        self.pred  = pred

        if pred == 0:
            pred = "neutral"
            return pred
        elif pred == 1:
            pred = "calm"
            return pred
        elif pred == 2:
            pred = "happy"
            return pred
        elif pred == 3:
            pred = "sad"
            return pred
        elif pred == 4:
            pred = "angry"
            return pred
        elif pred == 5:
            pred = "fearful"
            return pred
        elif pred == 6:
            pred = "disgust"
            return pred
        elif pred == 7:
            pred = "surprised"
            return pred

# Here you can replace path and file with the path of your model and of the file from the RAVDESS dataset you want to use for the prediction,
# Below, I have used a neutral file: the prediction made is neutral.

pred = livePredictions(path='/home/red/OF/Emotion-Classification/Emotion_Voice_Detection_Model.h5')


                
'''
pred = livePredictions(path='/home/red/OF/Emotion-Classification/Emotion_Voice_Detection_Model.h5',
                       file='/home/red/OF/Emotion-Classification/01-01-01-01-01-01-01.wav')
'''

pred.load_model()
pred.makepredictions()

'''
Could we take out the librosa stuff and replace librosa.feature.mfcc(y=data.. with mfcc from ofx?
'''
'''
        mfccs = np.mean(librosa.feature.mfcc(y=im, sr=44000, n_mfcc=40).T, axis=0)
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=2)
        x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict_classes(x)
        print( "Prediction is", " ", self.convertclasstoemotion(predictions))
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
'''