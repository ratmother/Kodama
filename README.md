# Kodama

## Description

Kodama is an experimental ambient intelligence which uses machine learning techniques to form a more immanently spiritual experience between technology and human in an installation or generally any public environment context. Kodama will utilize self-definition to form an internal memory that is active and fluid in its relation to the world rather than conventional methods seen in classification systems which use symbolic prior definitions such as ‗drum‘ or ‗human‘, require frustration-free domestic applicability and have enormous datasets with the intention of reaching a perfect representation of the world.

## Requirements

A microphone. Python 3 along with an audio generative system which receives OSC messages(I use Supercollider). The OSC message coming out of Kodama is a list of the nearest sound files as well as additional attributes: [file_name.wav, mortality, dimension, buffer size].

Also needed are these libraries: numpy, pickle, scipy, matploblib, osc4py3, sklearn, essentia, soundfile, hdbscan, wavio.

Tested and working on Linux (Ubuntu 19.04). 

## User information

Users can control Kodama with argparse when running Python. This allows the user to decide how much space Kodama can use, how large its audio recording length is and how finely it subdivides audio for learning. 
