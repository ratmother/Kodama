#include "synapse.h"

//TO DO: test vocal detection, implement PCA into synapse.
//Once thats done we need to start testing. Everything works? Try implementing the video aspect of it.

void synapse::setup(std::string file){
    fileName = file;    
    isEnded = true;
    //std::cout <<  "Synapse set up ... " << file << endl;
}
/*
    The update function takes in all the dectExpr's and adds them to the weighted average vector system.
    Effectively, each synapse 'knows' the average expression value it correlates to for each type (e.g movement, facial expression)
    and how volatile the value tends to be. e.g A synapse knows that it 'causes' stable facial happiness, sudden drowsiness, and sudden low movement.
*/

void synapse::update(dectExpr &input_face, dectExpr input_voice, dectExpr input_drowsy, dectExpr input_movement, float position){ //float movement, float attention...make vector?
    if(position < 0.95){
        isEnded = false;
        cooldown -= 40; 
        snap_timer += 1;
        face_volt += input_face.stable_dist;
        movement_volt += input_movement.stable_dist;
        if (snap_timer >= 3){ 
            avg_face = getAverage(face_snapshots, input_face.conv);
            avg_movement = getAverage(movement_snapshots, (input_movement.conv));
            avg_drowsy = getAverage(drowsy_snapshots, input_drowsy.conv);
            avg_voice = getAverage(voice_snapshots, input_voice.conv);
            snap_timer = 0;
        }
        if(cooldown < 10){
            face_result = getWeightedAverage(face_data, face_snapshots, 3, avg_face);
            movement_result = getWeightedAverage(movement_data, movement_snapshots, 3, avg_movement);
            voice_result = getWeightedAverage(voice_data, voice_snapshots, 3, avg_voice);
            drowsy_result = getWeightedAverage(drowsy_data,drowsy_snapshots, 3, avg_drowsy);
            cooldown = 600;
        }
    } else {
        isEnded = true;
    }
}

void synapse::draw(){
    ofDrawBitmapStringHighlight("This Synapse is for: "+fileName,0,50);
    ofDrawBitmapStringHighlight("Average Facial Expression: "+ofToString(face_result),0,75);
    ofDrawBitmapStringHighlight("Average Movement Expression: "+ofToString(movement_result),0,100);
    ofDrawBitmapStringHighlight("Average Drowsiness: "+ofToString(drowsy_result),0,125);
    ofDrawBitmapStringHighlight("Average Vocal Expression: "+ofToString(voice_result),0,150);
    ofDrawBitmapStringHighlight("Facial Expression Volatility: "+ofToString(face_volt),0,200);
    ofDrawBitmapStringHighlight("Movement Expression Volatility: "+ofToString(movement_volt),0,250);
}

float synapse::getAverage(std::vector<float> &v, float input){
    v.push_back(input); 
    return std::accumulate(v.begin(), v.end(), 0.0)/v.size();
}

/* 
    This function was inspired by attempting to get an EMA (exponential moving average) to work, is in't an EMA though.
    It simply takes the most recent entries (span), averages them, and adds( +, so can be negative) half of the difference 
    (to the overall entries) to the average of all the entries.
    After the size of the input vector reaches span, the fv (front vector of size span) will start to diverge from the 
    input vector more and more as entries are added.
*/

float synapse::getWeightedAverage(std::vector<float> &v, std::vector<float> &snapshots, int span, float average){
    snapshots.clear();
    if(average != 0){ // Averages of 0 mean that the input is very likely not recieving any interesting information. Neutral inputs that hover near zero will still go through.
        v.push_back(average);
        if(v.size() > span){
            std::vector<float> fv = v;
            fv.erase(fv.begin(),fv.end()-span);
            for(int i = 0; i < fv.size(); i++){
                std::cout << "fv " << fv[i] << endl;
            }
            for(int i = 0; i < v.size(); i++){
                std::cout << "v " << v[i] << endl;
            }
            float maj = std::accumulate(v.begin(), v.end(), 0.0)/v.size();
            float min = std::accumulate(fv.begin(), fv.end(), 0.0)/fv.size();
            // std::cout << min << " the fv averaged and " << maj << " the overall average, results in " << (((min - maj)/2) + maj) << endl;
            return (((min - maj)/2) + maj);
        }
    }
    return std::accumulate(v.begin(), v.end(), 0.0)/v.size();
}
