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
        void startSynapse(synapse &input, ofSoundPlayerExtended &splayer, std::string face, std::string voice, std::string drowsy);
        //OFXAUDIOANALYZER
        ofxAudioAnalyzer audioAnalyzer;
        ofSoundBuffer soundBuffer;
        ofSoundPlayerExtended player;
        int sampleRate;
        int bufferSize;
        float rms;
        float power;
        float pitchFreq;
        float pitchFreqNorm;
        float pitchConf;
        float pitchSalience;
        float hfc;
        float hfcNorm;
        float specComp;
        float specCompNorm;
        float centroid;
        float centroidNorm;
        float inharmonicity;
        float dissonance;
        float rollOff;
        float rollOffNorm;
        float oddToEven;
        float oddToEvenNorm;
        float strongPeak;
        float strongPeakNorm;
        float strongDecay;
        float strongDecayNorm;
        vector<float> spectrum;
        vector<float> melBands;
        vector<float> mfcc;
        vector<float> hpcp;
        vector<float> tristimulus;
        bool isOnset;
        ofxPanel gui;
        ofxFloatSlider smoothing;
        //OFXFIFO
        std::string emotion_face;
        std::string emotion_voice;
        std::string emotion_wakeful;
        ofxFifo::vidWriteThread vwt;
        std::vector<std::string> emotion_split;
        std::string emotion_pipe;
        int timer;
        synapse test1;
};
