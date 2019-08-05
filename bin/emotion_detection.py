from statistics import mode
import cv2
from keras.models import load_model
import numpy as np
import fifoutil
import librosa
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# Modified version of this facial emotion detection script (MIT License) https://github.com/oarriaga/face_classification/blob/master/src/video_emotion_color_demo.py
# Modifications by ratmother for project Kodama, exchanges information regarding BOTH facial and vocal features with a c++ app using two keras models.
# MFCCs are sent from ofxAudioanalyzer (C++ Openframeworks app using Essentia) to python, it is then classified with the voice emotion model.
# Voice emotion model is licensed under CC BY-NA-SC 4.0.
# The resulting vocal emotion is sent back to C++.
# The facial emotion detection simply sends whatever emotion it captures to C++. In the future I could have C++ send the facial data to be classified which is probably more efficient.
# I've placed these two models in the same script for more seamless debugging and simplicity, the plan is to have a seperate script for both gesture and Darknet if I go that route.

# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml' 
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
drowsy_predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotion_labels = get_labels('fer2013')

# parameters for loading VOCAL MODEL
emotion_vocal_model_path = 'Emotion_Voice_Detection_Model.h5'

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
voice_classifer = load_model(emotion_vocal_model_path)
detect = dlib.get_frontal_face_detector()

#  Landmark detection for drowsy detector
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

# Setting up parameters
emotion_target_size = emotion_classifier.input_shape[1:3]
m = None #MFCC, if found, it is placed into this variable.
thresh = 0.25 #Thresold used for drowsy detection.
frame_check = 20
drowsiness = 0
vocalEmotion = "None" #The output string of the vocal emotion detector.
faceEmotion = "None"
drowsyEmotion = "None"
emotionOut = "None"

# starting lists for calculating modes
emotion_window = []
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

#Main loop for all three detection algorithms
while True:

    #DROWSY DETECTION LOOP
    ret, frame=video_capture.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
    	shape = drowsy_predict(gray, subject)
    	shape = face_utils.shape_to_np(shape)#converting to NumPy Array
    	leftEye = shape[lStart:lEnd]
    	rightEye = shape[rStart:rEnd]
    	leftEAR = eye_aspect_ratio(leftEye)
    	rightEAR = eye_aspect_ratio(rightEye)
    	ear = (leftEAR + rightEAR) / 2.0
    	leftEyeHull = cv2.convexHull(leftEye)
    	rightEyeHull = cv2.convexHull(rightEye)
    	cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    	cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    	if ear < thresh:
    		drowsiness += 1
        else:
            drowsiness = 0
        drowsyEmotion = str(drowsiness)

    #VOCAL EMOTION DETECTION LOOP
    emotion = "none"
    try:
        m = fifoutil.read_txt("data/mfcc") #Read data from ofxAudioanalyzer 
    except:
        pass 
    if m is not None:
        m = np.fromstring(m, dtype=float, sep=',')
        x = np.expand_dims(m, axis=2)
        x = np.expand_dims(x, axis=0)
        pred = voice_classifer.predict_classes(x)
        if pred == 0:
            emotion = "neutral"
        elif pred == 1:
            emotion = "calm"
        elif emotion == 2:
            emotion = "happy"
        elif pred == 3:
            emotion = "sad"
        elif pred == 4:
            emotion = "angry"
        elif pred == 5:
            emotion = "fearful"
        elif pred == 6:
            emotion = "disgust"
        elif pred == 7:
            emotion = "surprised"
        elif pred == None:
            emotion = "none"
        vocalEmotion = emotion

    #FACE EMOTION DETECTION LOOP
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)
    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)
        faceEmotion = emotion_text

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()
        emotion_face = "Face:" + emotion_mode
        emotion_voice = "Voice:" + emotion
        emotion_drowsiness = "Sleepiness:" + str(drowsiness)
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_face,
                  color, 0, -45, 1, 1)
        draw_text(face_coordinates, rgb_image, emotion_voice,
                  color, 0, -75, 1, 1) 

        draw_text(face_coordinates, rgb_image, emotion_drowsiness, color, 0, -100, 1, 1) 

    emotionOut = vocalEmotion + "," + faceEmotion + "," + drowsyEmotion
    fifoutil.write_txt(emotionOut.encode(), "data/emotion_voice") #Sends emotions to the pipe to be picked up by openframeworks.
    print(emotionOut)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()


