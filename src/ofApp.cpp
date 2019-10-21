#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    timer = 0;
    system("killall python"); 
    system("gcc -v; pwd; python audio_comparative_analysis.py; python emotion_detection.py &");
    ofBackground(34);
    ofSetFrameRate(60);
    //ofxFifo::make("data/emotion_voice");
    //ofxFifo::make("data/mfcc");
    sampleRate = 44100;
    int channels = 1; //Mono...
    //int channels = 2; //Stereo...
    bufferSize = 512;
    audioAnalyzer.setup(sampleRate, bufferSize, channels);
    gui.setup();
    gui.setPosition(20, 150);
    gui.add(smoothing.setup  ("Smoothing", 0.0, 0.0, 1.0));
    test1.setup("learning_sounds/beatTrack.wav");
}

//--------------------------------------------------------------
void ofApp::update(){    
    if(timer >= 30){
        timer = 0;
        long_timer ++;
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
      //  inpDif(face);
        inpDif(movement);
       // inpDif(voice);
       // inpDif(drowsy);
       // inpDif(movement);
        runSynapse(test1, player);
        soundBuffer = player.getCurrentSoundBuffer(bufferSize);
        audioAnalyzer.analyze(soundBuffer);
        mfcc = audioAnalyzer.getValues(MFCC, 0, smoothing);
        if (long_timer >= 25){
            long_timer = 0;
        }
    }
    timer ++;
    ofSetWindowTitle(ofToString(ofGetFrameRate()));
}

//--------------------------------------------------------------
void ofApp::draw(){
    //-Gui & info:  
    gui.draw();
    ofSetColor(255);
    ofSetColor(ofColor::hotPink);
    test1.draw();
   // ofDrawBitmapStringHighlight("input emotion state state: "+ofToString(state),0,50);
}

void ofApp::keyPressed(int key){
    ofxFifo::write_array(mfcc,"data/mfcc"); //File needs read and write permissions by the way.
    //std::cout << "Delivered to mfcc pipe: " << ofxFifo::read_str("data/mfcc") << endl;
    //std::cout << "PCA coordinates: " << ofxFifo::read_str("data/pca_out") << endl;
}
//--------------------------------------------------------------
void ofApp::exit(){
    audioAnalyzer.exit();
    system("killall python"); 
}

void ofApp::runSynapse(synapse &input,ofSoundPlayerExtended &splayer){
    if(input.isEnded == true){
        splayer.load(input.fileName);
        splayer.play();
        input.cooldown = 600;
    }
    input.update(face, voice, drowsy, movement, splayer.getPosition());
}

void ofApp::inpDif(dectExpr &dec){
//Should it be that each 'medium' of input, facial, vocal, movement, has its own function...? 
    if(dec.conv == 0){
        dec.stable -= 10;
    } else {
        dec.stable += dec.conv * 2;
    }
    if(dec.stable >= 1){
        dec.stable -= 1 + abs(dec.stable/5);
    } else if (dec.stable <= -1) {
        dec.stable += 1 + abs(dec.stable/5);
    }
    dec.mvg_avg.push_back(dec.stable); 
       // dec.stable_compare = dec.stable;
    float dist;
    dec.stable_compare = std::accumulate(dec.mvg_avg.begin(), dec.mvg_avg.end(), 0.0)/dec.mvg_avg.size();
    if(dec.stable_compare >= dec.stable){
        dist = abs(1 - abs((((dec.stable_compare + dec.stable)/dec.stable)/2)));
    } else {
        dist = abs(1 - abs((((dec.stable + dec.stable_compare)/dec.stable_compare)/2)));
    }
    std::cout << "input is " << dec.conv << " dec.stable is " << dec.stable << " dec.stable compare is " << dec.stable_compare << endl;
    if(!isfinite(dist)){
        dec.stable_dist = 0;
    } else if (dist > 0.5) {
        dec.stable_dist = dist;
    } else {
        dec.stable_dist = 0;
    }
    std::cout << dec.stable_dist << endl;
    if(dec.mvg_avg.size() > 20){
        dec.mvg_avg.erase(dec.mvg_avg.begin());
    }
}

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

//make consistentcy multiplier!
