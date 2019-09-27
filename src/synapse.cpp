#include "synapse.h"


void synapse::setup(std::string file){
    fileName = file;    
    current_reaction = 0;
    bufferSize = 512;
    isEnded = true;
    lastEMA = 0;
    std::cout <<  "synapse set up ... " << file << endl;
}

//Synapses for sound include the PCA x and y coordinates for giving unlearned content an estimated emotional value.
void synapse::setCoord(float xin, float yin){
    x = xin;
    y = yin;
}

//Synapses for video could include whats in the video, e.g 'dog', 'human'. 
//Synapses for behaviours (UI behaviours) could include colours, speed, strength.
//Proactive and reactive states of behaviours (set by user), e.g, depressive states avoided (proactive) or complimented by known-depressive content (reactive)

//Idea, two settings: simple (party mode, baby mode and proactive or reactive) or advanced mode (allowing user to set how much sad or happy is awarded)
//Ergo, a user could set up Kodama to be scary if they wanted to use it for halloween(desiring fear), or relaxed (desiring minmial movement) if they want to relax.
//The CIA could use it to find what spooks someone the most! Spooky!
void synapse::update(std::string face, std::string voice, std::string drowsy, float position){
    if(position < 0.9){
        isEnded = false;
        cooldown -= 50;
        snap_timer += 1;
        if(face == "neutral"){
            current_reaction = 0;
        }
        if(face == "happy"){
            current_reaction =  ofRandom(1000);
        }
        if(face == "sad"){
            current_reaction = -250;
        }
        if(face == "angry"){
            current_reaction = -300;
        }
        current_wakefulness = std::stoi(drowsy);
        if (snap_timer >= 3){
            accumWakeful += current_wakefulness;
            if(reaction_snapshots.size() == 0){
                reaction_snapshots.push_back(current_reaction);
                avgReact = std::accumulate( reaction_snapshots.begin(), reaction_snapshots.end(), 0.0)/reaction_snapshots.size(); 
            }else{
                reaction_snapshots.push_back(current_reaction * reaction_snapshots.size());
                avgReact = std::accumulate( reaction_snapshots.begin(), reaction_snapshots.end(), 0.0)/reaction_snapshots.size(); 
            }
            snap_timer = 0;
        }
        //Capture averaged reaction as a permanant data point in this synapse at regular intervals.
        if(reaction_snapshots.size() >= 5 && cooldown < 10){
            snapped_reaction.push_back(avgReact);
            sar = getEMA(snapped_reaction);
            snapped_drowsy.push_back(accumWakeful);
            accumWakeful = 0;
            reaction_snapshots.clear();
            cooldown = 600;
        }
    } else if(position >= 0.91){
        end();
    }
}


void synapse::draw(){
    ofSetColor(255);
    ofSetColor(ofColor::hotPink);
    ofDrawBitmapStringHighlight("This content is running: "+fileName,0,50);
}

void synapse::end(){
    isEnded = true;
    std::cout << "ended" << endl; 
    //At the end of the content, add the averaged reaction as a permanant data point in this synapse.
    if(reaction_snapshots.size() >= 5 && cooldown < 10){
        std::cout << "ended and snapped" << endl; 
        std::cout << reaction_snapshots.size() << endl; 
        snapped_reaction.push_back(avgReact);
        sar = weightedAverage(snapped_reaction);
        std::cout << "weighted average is  " << sar << endl; 
        snapped_drowsy.push_back(accumWakeful);
        //snapped_movement, snapped_attention, snapped_popularity... etc.
        cooldown = 600;
        accumWakeful = 0;
        reaction_snapshots.clear();
    }
}

void synapse::start(){
    isEnded = false;
    cooldown = 0;
    accumWakeful = 0;
    reaction_snapshots.clear();
}

//After playing around with moving averages and EMAs I decided to make a simple weighted average to fit the intended goal.
//This function causes the latest additions to the vector to be weighted higher than older additions, its biased towards newer data.
//I'm not sure if this is a 'real' or standard algorithm to use, I've never seen it before, its very simple. 
float synapse::weightedAverage(std::vector<float> v){
    std::vector<float> wa;
    float size = v.size();
    for (float i = 0; i < size; i++) {
        float smoothing = (i+1)/size;
        std::cout << v[i] << "smoothed by"  << smoothing << " is weighted..." << (v[i] * smoothing) << endl;
        wa.push_back(v[i] * smoothing);
    }
    return std::accumulate(wa.begin(), wa.end(), 0.0)/wa.size(); 
}

float synapse::getEMA(std::vector<float> v){
    float EMA;
    if(v.size() > 1){
        float mult = 2 / (v.size()+ 1);
        float ma = std::accumulate(v.begin(), v.end(), 0.0)/v.size();
        float EMA = (v[v.size()-1] - ma) * mult + ma;   
        std::cout << v.size() << endl;
        std::cout << v[v.size()-1] << " EMA is "  << EMA  << endl;
        return EMA;
    }
    return 0;
}
/* 
Initial SMA: 10-period sum / 10 

Multiplier: (2 / (Time periods + 1) ) = (2 / (10 + 1) ) = 0.1818 (18.18%)

EMA: {Close - EMA(previous day)} x multiplier + EMA(previous day). 
*/

