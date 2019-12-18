#pragma once

#include "ofMain.h"
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
		void playNearest();
        void runSynapse(synapse &input, ofSoundPlayerExtended &splayer);
        void updateSynapse(synapse &input, ofSoundPlayerExtended &splayer);
        void inpDif (dectExpr &dec);
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
        std::vector<std::string> names_split;
        std::string file_names;
        std::string file_path;
        std::string sound_index_pipe;
        std::vector<std::string> index_split; 
        std::vector<int> index_conv; //Probably not the best way to do this.
        std::vector<synapse> synapses;
        std::vector<synapse *> displayed_synapses;
        int timer;
        int long_timer;
        int xlt;   
};
