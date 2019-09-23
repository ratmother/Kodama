#pragma once

#include "ofMain.h"
#include "ofxAudioAnalyzer.h"
#include "ofSoundPlayerExtended.h"
#include "ofxGui.h"
#include "ofxFifo.h"
#include "ofxFifoThread.h"
#include <boost/algorithm/string.hpp>

struct soundSynapse {

	string fileName;
    float x; //X and Y are taken from the audio_comparative_analysis.py script.
    float y;
    float volatility; //Intensity of the users reaction.
    float reaction; //Direction of the emotional quality, e.g sad or happy.
    float wakefulness; //Drowsiness or excitement.

};

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();
        void exit();
		void keyPressed(int key);
        //OFXAUDIOANALYZER
        ofxAudioAnalyzer audioAnalyzer;
        ofSoundPlayerExtended player;
        int sampleRate;
        int bufferSize;
        ofSoundBuffer soundBuffer;
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
        //KODAMA
        soundSynapse createSoundSynapse(float x, float y, string fileName);
        void datSnap();
        void emotamine(soundSynapse synapse);
        void getEmotionData(std::string pipe);
        vector<soundSynapse> soundSynapses;
        vector<float> reaction_snapshots;
        //Emotion system experimental
        int timer;
        int snap_timer;
        std::vector<std::string> emotion_split;
        std::string emotion_pipe;
        float current_reaction;
        int current_wakefulness;
        float avgReact;
        float accumWakeful;
        bool synapseIsRunning;
};
