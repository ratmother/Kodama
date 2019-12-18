#pragma once

#include "ofMain.h"

// Detected inputs, expressions, e.g movement, facial expression, drowsiness. Each type of world-input has its own dectExpr. 
struct dectExpr{
    std::string input;
    std::vector<float> mvg_avg;
    float stable;
    float stable_compare;
    float conv;
    float stable_dist;
};

class synapse : public ofBaseApp{

	public:
		void setup(std::string file);
        void draw(int displace);
		void update(dectExpr &input_face, dectExpr input_voice, dectExpr input_drowsy, dectExpr input_movement, float position);
        float getWeightedAverage(std::vector<float> &v, std::vector<float> &snapshots, int span, float average);
        float getAverage(std::vector<float> &v, float input);
        void updateStabl(float face, float voice, float drowsy, float movement);

        //Raw input data analysis and averaging.
        std::vector<float> face_snapshots;
        std::vector<float> face_data;
        std::vector<float> drowsy_snapshots;
        std::vector<float> drowsy_data;
        std::vector<float> voice_snapshots;
        std::vector<float> voice_data; 
        std::vector<float> movement_snapshots;
        std::vector<float> movement_data;   
        float avg_face;
        float avg_drowsy;
        float avg_voice;
        float avg_movement;
        float face_result;
        float movement_result;
        float voice_result;
        float drowsy_result;

        float face_volt;
        float movement_volt;

        std::string emotion_pipe;
        std::string fileName;
        int snap_timer;
        float cooldown;
        float x;
        float y;
        bool isEnded;
};
