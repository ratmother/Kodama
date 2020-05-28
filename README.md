# Kodama

## Description

Kodama is an experimental ambient intelligence which uses machine learning techniques to form a collaborative sonic experience between Kodama and its environment in an installation context or generally any public environment context. Kodama utilizes metastable self-definition to form an internal idenity that is active and fluid in its relation to the local soundscape ecology.

## Requirements

A microphone. 

Python 3 along with an audio generative system which receives OSC messages(I use Supercollider). The OSC messages coming out of Kodama is a list of the nearest sound files as well as additional attributes: [file_name.wav, mortality, dimension, buffer size].

Also needed are these libraries: numpy, pickle, scipy, matploblib, osc4py3, sklearn, essentia, soundfile, hdbscan, wavio.

Tested and working on Linux (Ubuntu 19.04). 

## User information

Users can control Kodama with argparse when running Python. This allows the user to decide how much space Kodama can use, how large its audio recording length is and how finely it subdivides audio for learning. 

 Important argparse commands:
    -l :Lists devices on your computers.
    -d :Selects the device to use by index. So, when -l lets you know that your microphones index is, for example, 7 you run Kodama by calling 'python threader.py -d 7'. 
    -p :Whether or not to show the plot.
    -sl :Size limit of the dataset. The dataset grows so it's important to set the limit by mb(s). This is approximate as the datasets size is controlled by probabilities and multipliers.

You must create a ‘slices’ folder in the ‘data’ folder to store audio. Currently Kodama does not regulate this folder (delete the files after its done running) so if you want to remove sounds you have to delete them manually. However Kodama functions perfectly fine if previously recorded sounds are in the slices folder.

