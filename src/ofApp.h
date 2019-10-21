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
        void runSynapse(synapse &input, ofSoundPlayerExtended &splayer);
        void inpDif (dectExpr &dec);
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
        float assignFaceValue(dectExpr faceIn);
        float assignVoiceValue(dectExpr voiceIn);
        dectExpr voice;
        dectExpr face;
        dectExpr drowsy;
        dectExpr movement;
        std::string last_face;
        int face_cons_multipler = 0;
        ofxFifo::vidWriteThread vwt;
        std::vector<std::string> emotion_split;
        std::string emotion_pipe;
        int timer;
        int long_timer;
        synapse test1;       
};
