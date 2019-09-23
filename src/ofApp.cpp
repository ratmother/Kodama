#include "ofApp.h"

soundSynapse ofApp::createSoundSynapse(float x, float y, string fileName) {; 
    soundSynapse synapse = {{},x, y};
    synapse.fileName = fileName;
	return synapse;
}

videoSynapse ofApp::createVideoSynapse(string fileName) {
    videoSynapse synapse = {{}};
    synapse.fileName = fileName;
	return synapse;
}

//--------------------------------------------------------------
void ofApp::setup(){
    timer = 0;
    mag_timer = 0;
    state = 0;
    stable = 0;
    learn_neg = 1;
    learn_pos = 1;
    current_reaction = 0;
    system("killall python"); 
    system("gcc -v; pwd; audio_comparative_analysis.py; python emotion_detection.py &");
    ofBackground(34);
    ofSetFrameRate(60);
    //ofxFifo::make("data/emotion_voice");
    //ofxFifo::make("data/mfcc");
    sampleRate = 44100;
    bufferSize = 512;
    int channels = 1; //Mono...
    //int channels = 2; //Stereo...
    
    audioAnalyzer.setup(sampleRate, bufferSize, channels);
    
    player.load("learning_sounds/beatTrack.wav");
    
    gui.setup();
    gui.setPosition(20, 150);
    gui.add(smoothing.setup  ("Smoothing", 0.0, 0.0, 1.0));
    soundSynapse test = createSoundSynapse(0,0, "learning_sounds/test440mono.wav");
   
}

//--------------------------------------------------------------
void ofApp::update(){
    ofSetWindowTitle(ofToString(ofGetFrameRate()));
    //This should be sent to livepredictions somehow...
    soundBuffer = player.getCurrentSoundBuffer(bufferSize);
    
    //-:ANALYZE SOUNDBUFFER:
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
     datSnap();
}

//--------------------------------------------------------------
void ofApp::draw(){
    if(timer >= 60){
        timer = 0;
        getEmotionData("data/emotion_out");
    }
    timer ++;
    //-Gui & info:  
    gui.draw();
    ofSetColor(255);
    ofSetColor(ofColor::hotPink);
   // ofDrawBitmapStringHighlight("input emotion state state: "+ofToString(state),0,50);
    if(learn_neg > 2){
        ofDrawBitmapStringHighlight("I am  a little moody today",0,200);
    }
     if(learn_pos > 2){
        ofDrawBitmapStringHighlight("I am feeling pretty good",0,200);
    }
       if(learn_neg > 3){
        ofDrawBitmapStringHighlight("I am unhappy",0,200);
    }
     if(learn_pos > 3){
        ofDrawBitmapStringHighlight("I am happy",0,200);
    }
    if(learn_neg > 4){
        ofDrawBitmapStringHighlight("I am gonna cry soon",0,200);
    }
     if(learn_pos > 4){
        ofDrawBitmapStringHighlight("I am just having the time of my life",0,200);
    }
}

//--------------------------------------------------------------
void ofApp::getEmotionData(std::string pipe){
    //Read the pipe which contains emotion-detection data (from the python script).
    //Split the string by comma and place them into a vector, emotion_split.
    //Note: emotion_'attribute' names refer to emotion detection (trings describing current emotion states) data only.
    //Happiness is what we're after and it is scored highly, sadness and neurtrality usually equate to disinterest.
    emotion_pipe = ofxFifo::read_str(pipe);
    boost::split(emotion_split, emotion_pipe, [](char c){return c == ',';}); 
    emotion_voice = emotion_split[0];
    emotion_face = emotion_split[1];
    emotion_wakeful = emotion_split[2];
    //Current numbers are yet to be tested with users.
    if(emotion_face == "neutral"){
        current_reaction = 0;
    }
    if(emotion_face == "happy"){
        current_reaction = 1000;
    }
    if(emotion_face == "sad"){
        current_reaction = -250;
    }
    if(emotion_face == "angry"){
        current_reaction = -300;
    }
    current_wakefulness = std::stoi(emotion_wakeful);
    //std::cout << current_reaction << endl;
    //std::cout << current_wakefulness << endl;
}

void ofApp::keyPressed(int key){
    getEmotionData("data/emotion_out");
    player.stop();
    switch (key) {
       
        case '1':
            player.load("learning_sounds/test440mono.wav");
            break;
        case '2':
            player.load("learning_sounds/flute.wav");
            break;
        case '3':
            player.load("learning_sounds/laugh_2.wav");
            break;
        case '4':
            player.load("learning_sounds/yes_2.wav");
            break;
        case '5':
            player.load("learning_sounds/beatTrack.wav");
            break;
        case '6':
            player.load("learning_sounds/come_on_1.wav");
            break;
            
            
        default:
            break;
    }
    player.play();
    ofxFifo::write_array(mfcc,"data/mfcc"); //File needs read and write permissions by the way.
    //std::cout << "Delivered to mfcc pipe: " << ofxFifo::read_str("data/mfcc") << endl;
    //std::cout << "PCA coordinates: " << ofxFifo::read_str("data/pca_out") << endl;
}
//--------------------------------------------------------------
void ofApp::exit(){
    audioAnalyzer.exit();
    player.stop();
    system("killall python"); 
}

void ofApp::emotamine(soundSynapse synapse){ //this needs to be pass by reference not copy
   // if(finished){
      //synapse.wakefulness =  accumWakeful;
     // synapse.reaction = avgReact;
    //}
}
//I'm going to remove the video aspect I think, as the X and Y positions from the comparative analysis, and it focuses the project. 
//Perhaps it should be animations, and/or the video aspect will just be simpler..s

void ofApp::datSnap(){ //datSnap(vector snapshots, int wakefulness)?
    snap_timer += 1;
    if (snap_timer >= 60){
        accumWakeful += current_wakefulness;
        if(reaction_snapshots.size() == 0){
          reaction_snapshots.push_back(current_reaction);
          avgReact = std::accumulate( reaction_snapshots.begin(), reaction_snapshots.end(), 0.0)/reaction_snapshots.size(); 
        }
        else {
            reaction_snapshots.push_back(current_reaction * mag_snapshots.size());
            avgReact = std::accumulate( reaction_snapshots.begin(), reaction_snapshots.end(), 0.0)/reaction_snapshots.size(); 
            std::cout << current_reaction * reaction_snapshots.size() << " " << avgReact << endl;
        }
        snap_timer = 0;
    }

   /* check if something that is being learnt about is running, audio video animation etc, if it isnt, reset the snapshot container every 100 seconds
   //its possible in the final version that there is no down time for learning, so we will need to cut the vector to be smaller
   //or each content type gets its own reaction snap shots and accumulator, 100 seconds is the effective spillover rate
   //a better change is to slowly chop  away the front of the vector while there is no content playing
   //OR make it so every 100 snaps, the system adds a data point for the running synapse. 
   if(reaction_snapshots.size() >= 60){
       //add info to snap function (which is also called at the end of a clip --- if its not long enough, dont add data point)
       //actually, add a cooldown period after this so that the end snapshot isnt saved as a data point if there was a recent capture
       capture(reaction_snapshots, acummwakeful, cooldown 600)
        cooldown = 600;
        accumWakeful = 0;
       reaction_snapshots.clear();
    }
    */
    //Save mag as data point in the synapse. Probably the output data point for each synapse.
    //Actually, each synapse should be submitted as a data point, inclding emoMag as its output, but each synapse will only have one emoMag at a time.
    //At the end of the playback, submit the synapse for data entry with the emoMag. 
    //Take average of a collection of snapshots of mag
}
//make consistentcy multiplier!
//Next make a function which calculates boredum. longer in the negative --> slower back up drift, quicker from happiness. 
//When happy start recording and place in happy class and vice versa. Voice/sounds associated with happy triggers happy-associated music. 
