Name ideas:
Kodama Emotional Support System 
Kess

Notes:

A Linux Tensorflow based AI which learns about content you enjoy, Kodama creates generative music and selects visual content for you based on your mood and what you like when in that mood.

(!!!) <== Nessesary for Kodama to be at all functional

The Thaw -- Star Trek episode with funny quotes.

------USER EMOTION DETECTORS--------
Below will share a similar system which detects happiness/sadness(up/down) and excited/bored(right/left).
Sadness = lazy, ambient, moody, dark, atonal, minor.
Happiness = wakeful, bright, cheerful, major.
System is biased towards 'wanting' to create happiness ut not excitement (happy/excited are two different parameters).
excitement/bored are neutral parameters (not all situations call for excitement).
attention is a secondary (but important) 'desire'. System learns what creates maximum amount of attention and happiness.
Good for kids, music events, installastions -- would be quite draining as a personal program I think but perhaps has some kind of theraputic benefit. Its potential utility for a single device user would be the mood following, perhaps there could be three modes which act slightly different (one for installation/event, children, and single user). Single user version wouldn't desire attention as much and would focus on the playlist system, whereas children/installation version would.

Emotion Detection (Face) -- Ready.

Emotion Detection (Voice) -- OfxAudioAnalyzer to python working, python back to Ofx not working yet.

Gesture detection -- Nothing yet (!!!).

Attention detection -- Nothing yet.

------CONTENT LEARNING--------------

Darknet Detection -- Python Ready. Need to try it with openframeworks.

Bitwig OSC into OFX -- Bitwig is up and running, has audio engine problems however.

Playlist system -- Nothing yet (!!!).

-----KODAMA-------------------------

Association Learning -- Nothing yet.

Generative Music -- Nothing yet (Bitwig or puredata).

Responsive video content -- Nothing yet (Darknet required).

Anime girl AI graphic content -- Nothing yet.

IDEAS:

If you yell angrily at Kodama she should become upset and teary eyed!

For voice input filter out the low end. Make sure in installtion set up the speakers are directed away from the mic and the audio pick up thresold is high enough (avoid feedback loop). If nothing works, have her listen only when she isn't playing music. (there should always be very soft ambient music on even in this off peiord)

Kodama learns music features AND learns what music features tie to what musical parameters rather than tie music pars to emotions

