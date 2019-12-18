#include "ofApp.h"


void ofApp::setup(){
    timer = 0;
    //system("killall python"); 
   // system("gcc -v; pwd; python audio_comparative_analysis.py; python emotion_detection.py &");
    ofBackground(34);
    ofSetFrameRate(60);
    gui.setup();
    gui.setPosition(20, 150);
    gui.add(smoothing.setup  ("Smoothing", 0.0, 0.0, 1.0));
    file_names = ofxFifo::read_str("data/sound_names_out");
    boost::split(names_split, file_names, [](char c){return c == ',';});
    for (int i = 0; i < names_split.size(); i++ ){
        file_path = "learning_sounds/";
        synapse in;
        file_path.append(names_split[i]); 
        in.setup(file_path);
        synapses.push_back(in);
    }
}


void ofApp::update(){    
    if(timer >= 30){ // Nothing below requires precise real time analysis, so weight is taken off of computation this way.
        timer = 0;
        emotion_pipe = ofxFifo::read_str("data/emotion_out");
        boost::split(emotion_split, emotion_pipe, [](char c){return c == ',';}); 
        voice.input = emotion_split[0];
        face.input = emotion_split[1];
        drowsy.input = emotion_split[2];
        movement.input = emotion_split[3];
        face.conv = assignFaceValue(face);
        voice.conv = assignFaceValue(voice);
        drowsy.conv = std::stoi(drowsy.input);
        movement.conv = std::stof(movement.input);
        inpDif(face);
        inpDif(movement);
        playNearest();
        for(int i = 0; i < synapses.size(); i++ ){
            updateSynapse(synapses[i], player);
        }
    }
    timer ++;
    ofSetWindowTitle(ofToString(ofGetFrameRate()));
}

void ofApp::draw(){
    ofBackground(34);
    gui.draw();
    ofSetColor(255);
    ofSetColor(ofColor::hotPink);
    for(int i = 0; i < synapses.size(); i++ ){ //this is probably sub-optimal..
        if(synapses[i].isEnded == false){ 
            displayed_synapses.push_back(& synapses[i]);
            if(displayed_synapses.size() >= 4){
                displayed_synapses.erase(displayed_synapses.begin());
            }
        }
    }
    for(int i = 0; i < displayed_synapses.size(); i++ ){
        displayed_synapses[i]->draw(i);
    }
}

void ofApp::playNearest(){
    //get synapse with least distance from desired emotional object state (not yet created)
    sound_index_pipe = ofxFifo::read_str("data/sound_ids");
    boost::split(index_split, sound_index_pipe, [](char c){return c == ',';});
    int rand = (int) ofRandom(0,index_split.size());
    int str_to_int = std::stoi(index_split[rand]);
    if(str_to_int > synapses.size()){ // This is kinda hacky and probably not optimal, the whole way I'm currently doing directory/file name stuff seems a bit wrong.
        std::cout <<  "this is new right " << endl;
        file_names = ofxFifo::read_str("data/sound_names_out");
        boost::split(names_split, file_names, [](char c){return c == ',';});
            for (int i = 0 + synapses.size(); i < names_split.size(); i++ ){
                file_path = "learning_sounds/";
                synapse in;
                file_path.append(names_split[i]); 
                in.setup(file_path);
                synapses.push_back(in);
            }
        runSynapse(synapses[str_to_int], player);  
    } else {
        runSynapse(synapses[str_to_int], player);  
    }
}

void ofApp::exit(){
   // system("killall python"); 
}

void ofApp::runSynapse(synapse &input,ofSoundPlayerExtended &splayer){
    std::cout << input.fileName << endl;
    splayer.load(input.fileName);
    splayer.play();
    input.cooldown = 600;
    input.isEnded = false;
}

void ofApp::updateSynapse(synapse &input, ofSoundPlayerExtended &splayer){
    if(input.isEnded == false){
        input.update(face, voice, drowsy, movement, splayer.getPosition());
    } 
}


/*
    inpDif is the function which takes in a specfic dectExpr (struct which each type of world-input has, e.g facial expression, movement...) 
    and measures the stability of its current value (see variable conv, short for converted-input), and then stores how much distance there is between
    its stable state and the current state. The stable state is a moving average of the past values of the dectExpr's value.
*/

void ofApp::inpDif(dectExpr &dec){
    if(dec.conv == 0){ // If the input is 0, we subtract from the stability so that neutral inputs can cause stability differences more effectively. 
        dec.stable -= 10;
    } else {
        dec.stable += dec.conv * 2; // Similarly, we double the input to increase differences and capture more stability differences.
    }
    if(dec.stable >= 1){ // This is a scaling decay, positive or negative. 
        dec.stable -= 1 + abs(dec.stable/5);
    } else if (dec.stable <= -1) {
        dec.stable += 1 + abs(dec.stable/5);
    }
    dec.mvg_avg.push_back(dec.stable); // Moving average captures the current stability. 
    float dist;
    dec.stable_compare = std::accumulate(dec.mvg_avg.begin(), dec.mvg_avg.end(), 0.0)/dec.mvg_avg.size();
    if(dec.stable_compare >= dec.stable){ // Change the forumula if the position of the two stability comparison variables differs in polarity.
        dist = abs(1 - abs((((dec.stable_compare + dec.stable)/dec.stable)/2)));
    } else {
        dist = abs(1 - abs((((dec.stable + dec.stable_compare)/dec.stable_compare)/2)));
    }
    //std::cout << "input is " << dec.conv << " dec.stable is " << dec.stable << " dec.stable compare is " << dec.stable_compare << endl;
    if(!isfinite(dist)){ // If one of the stability variables is 0, then we could get infinity. 
        dec.stable_dist = 0;
    } else if (dist > 0.5) {
        dec.stable_dist = dist;
    } else {
        dec.stable_dist = 0;
    }
    //std::cout << dec.stable_dist << endl;
    if(dec.mvg_avg.size() > 20){
        dec.mvg_avg.erase(dec.mvg_avg.begin());
    }
}

/*
    assign___Value functions take dectExpr world-inputs and converts them to floats, e.g 'happy' input to 1000. 
    These functions also include other features, such as an input consistency multiplier.
*/

float ofApp::assignFaceValue(dectExpr faceIn){
    float output = 0;
    std::string face_ = faceIn.input;
    if(face_ == "happy"){
        output =  1000;
        if(last_face == "happy"){
            face_cons_multipler += 1;
            output += (output/10) * face_cons_multipler;
        } else {
            face_cons_multipler = 0; 
        }
    }
    if(face_ == "sad"){
        output = -1000;
        if(last_face == "sad"){
            face_cons_multipler += 1;
            output += (output/10) * face_cons_multipler;
        } else {
            face_cons_multipler = 0; 
        }
    }
    if(face_ == "angry"){
        output = -1500;
        if(last_face == "angry"){
            face_cons_multipler += 1;
            output += (output/10) * face_cons_multipler;
        } else {
            face_cons_multipler = 0; 
        }
    }
    last_face = face_;
    return output;
}

float ofApp::assignVoiceValue(dectExpr voiceIn){
    float output = 0;
    std::string voice_ = voiceIn.input;
    if(voice_ == "happy"){
        output =  1000;
    }
    if(voice_ == "sad"){
       output = -250;
    }
    if(voice_ == "angry"){
       output = -300;
    }
    return output;
}