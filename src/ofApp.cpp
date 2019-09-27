#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    timer = 0;
    system("killall python"); 
   // system("gcc -v; pwd; python audio_comparative_analysis.py; python emotion_detection.py &");
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
        emotion_pipe = ofxFifo::read_str("data/emotion_out");
        boost::split(emotion_split, emotion_pipe, [](char c){return c == ',';}); 
        emotion_voice = emotion_split[0];
        emotion_face = emotion_split[1];
        emotion_wakeful = emotion_split[2];
        emotion_face = "happy"; //testing
        startSynapse(test1, player, emotion_face, emotion_voice, emotion_wakeful);
        soundBuffer = player.getCurrentSoundBuffer(bufferSize);
        audioAnalyzer.analyze(soundBuffer);
        //-:get Values:
        rms     = audioAnalyzer.getValue(RMS, 0, smoothing);
        power   = audioAnalyzer.getValue(POWER, 0, smoothing);
        pitchFreq = audioAnalyzer.getValue(PITCH_FREQ, 0, smoothing);
        pitchConf = audioAnalyzer.getValue(PITCH_CONFIDENCE, 0, smoothing);
        pitchSalience  = audioAnalyzer.getValue(PITCH_SALIENCE, 0, smoothing);
        inharmonicity   = audioAnalyzer.getValue(INHARMONICITY, 0, smoothing);
        hfc = audioAnalyzer.getValue(HFC, 0, smoothing);
        specComp = audioAnalyzer.getValue(SPECTRAL_COMPLEXITY, 0, smoothing);
        centroid = audioAnalyzer.getValue(CENTROID, 0, smoothing);
        rollOff = audioAnalyzer.getValue(ROLL_OFF, 0, smoothing);
        oddToEven = audioAnalyzer.getValue(ODD_TO_EVEN, 0, smoothing);
        strongPeak = audioAnalyzer.getValue(STRONG_PEAK, 0, smoothing);
        strongDecay = audioAnalyzer.getValue(STRONG_DECAY, 0, smoothing);
        mfcc = audioAnalyzer.getValues(MFCC, 0, smoothing);
    }
    timer ++;
    ofSetWindowTitle(ofToString(ofGetFrameRate()));
    //This should be sent to livepredictions somehow...
}

//--------------------------------------------------------------
void ofApp::draw(){
    //-Gui & info:  
    gui.draw();
    ofSetColor(255);
    ofSetColor(ofColor::hotPink);
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

void ofApp::startSynapse(synapse &input,ofSoundPlayerExtended &splayer, std::string face, std::string voice, std::string drowsy){
    if(input.isEnded == true){
        input.start();
        splayer.load(input.fileName);
        splayer.play();
    }
    input.update(face, voice, drowsy, splayer.getPosition());
}

//I'm going to remove the video aspect I think, as the X and Y positions from the comparative analysis, and it focuses the project. 
//Perhaps it should be animations, and/or the video aspect will just be simpler..s

   /* check if something that is being learnt about is running, audio video animation etc, if it isnt, reset the snapshot container every 100 seconds
   //its possible in the final version that there is no down time for learning, so we will need to cut the vector to be smaller
   //or each content type gets its own reaction snap shots and accumulator, 100 seconds is the effective spillover rate
   //a better change is to slowly chop  away the front of the vector while there is no content playing
   //OR make it so every 100 snaps, the system adds a data point for the running synapse. 
   if(reaction_snapshots.size() >= 60){
       //add info to snap function (which is also called at the end of a clip --- if its not long enough, dont add data point)
       //actually, add a cooldown period after this so that the end snapshot isnt saved as a data point if there was a recent capture
       whateversynapsethisis.snapped_reaction.push_back(avgReact);
       whateversynapsethisis.snapped_wakefulness.push_back(accumWakeful);
       accumWakeful = 0;
       reaction_snapshots.clear();
    }
    */
    //Save mag as data point in the synapse. Probably the output data point for each synapse.
    //Actually, each synapse should be submitted as a data point, inclding emoMag as its output, but each synapse will only have one emoMag at a time.
    //At the end of the playback, submit the synapse for data entry with the emoMag. 
    //Take average of a collection of snapshots of mag
    //behavour idea ['jump around', playsound('drum.wav'), 'turn red'] ---> check effectiveness via snapshots
//make consistentcy multiplier!
//Next make a function which calculates boredum. longer in the negative --> slower back up drift, quicker from happiness. 
//When happy start recording and place in happy class and vice versa. Voice/sounds associated with happy triggers happy-associated music. 
