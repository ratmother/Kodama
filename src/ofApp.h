#pragma once

#include "ofMain.h"
#include "ofxAudioAnalyzer.h"
#include "ofxGui.h"
#include "synapse.h"
#include "ofxFifo.h"
#include "ofxFifoThread.h"
#include "ofSoundPlayerExtended.h"
#include <boost/algorithm/string.hpp>

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();
        void exit();
		void keyPressed(int key);
        void startSynapse(synapse &input, ofSoundPlayerExtended &splayer, float face, float voice, float drowsy);
        //OFXAUDIOANALYZER
        ofxAudioAnalyzer audioAnalyzer;
        ofSoundBuffer soundBuffer;
        ofSoundPlayerExtended player;
        int sampleRate;
        int bufferSize;
        vector<float> mfcc;
        bool isOnset;
        ofxPanel gui;
        ofxFloatSlider smoothing;
        //OFXFIFO
        float assignFaceValue(std::string face);
        float assignVoiceValue(std::string voice);
        std::string emotion_face;
        std::string emotion_voice;
        std::string emotion_drowsy;
        ofxFifo::vidWriteThread vwt;
        std::vector<std::string> emotion_split;
        std::string emotion_pipe;
        int timer;
        synapse test1;
};
