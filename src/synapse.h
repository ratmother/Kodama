#pragma once

#include "ofMain.h"


class synapse : public ofBaseApp{

	public:
		void setup(std::string file);
		void update(float face, float voice, float drowsy, float position);
        void setCoord(float xin, float yin);
        float getWeightedAverage(std::vector<float> &v, std::vector<float> &snapshots, int span, float average);
        float getAverage(std::vector<float> &v, float input);
        std::vector<float> face_snapshots;
        std::vector<float> face_data;
        std::vector<float> drowsy_snapshots;
        std::vector<float> drowsy_data;
        std::string emotion_pipe;
        std::string fileName;
        int snap_timer;
        float avg_face;
        float avg_drowsy;
        float face_result;
        float drowsy_result;
        float cooldown;
        float x;
        float y;
        bool isEnded;
};
