#pragma once

#include "ofMain.h"


class synapse : public ofBaseApp{

	public:
		void setup(std::string file);
		void update(std::string face, std::string voice, std::string drowsy, float position);
		void draw();
        void setCoord(float xin, float yin);
        void start();
        void end();
        float weightedAverage(std::vector<float> v);
        float getEMA(std::vector<float> v);
        std::vector<float> reaction_snapshots;
        std::vector<std::string> emotion_split;
        std::vector<float> snapped_reaction;
        std::vector<float> snapped_drowsy;
        std::string emotion_pipe;
        std::string fileName;
        float current_reaction;
        int snap_timer;
        int current_wakefulness;
        float avgReact;
        float sar;
        float accumWakeful;
        float cooldown;
        bool synapseIsRunning;
        float x;
        float y;
        bool isEnded;
        float lastEMA;
        int bufferSize;
};
