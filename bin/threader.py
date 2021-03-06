import audio_comparative_analysis as aca
from osc4py3.as_eventloop import *
from osc4py3 import oscbuildparse
import threading
import time
import numpy as np 
import queue
import sounddevice as sd
import argparse
import sys

"""
          ⎥	   KODAMA   ⎥	

          ⎥	  Threader  ⎥	

 This is the secondary script, it handles mutli-threading, audio input, OSC and user control. 
 Data is sent to audio_comparative_analysis for segmentation and analysis.
 OSC messages containing sample information is sent out to Supercollider.
 We use argparse to allow for user terminal control functionality which in turn smooths out the testing process.
 Important argparse commands:
    -l :Lists devices on your computers.
    -d :Selects the device to use by index. So, when -l lets you know that your microphones index is, for example, 7 you run Kodama by calling 'python threader.py -d 7'. 
    -p :Whether or not to show the plot.
    -sl :Size limit of the dataset. The dataset grows so it's important to set the limit by mb(s). This is approximate as the datasets size is controlled by probabilities and multipliers.
"""

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
parser.add_argument(
    '-d', '--device', type=int_or_str, default = 7,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-b', '--blocksize', type=int, default=44100, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-p', '--plot', type=bool, default=True, help='Plot the samples')
parser.add_argument(
    '-al', '--audio_length', type=int, default=6, help='Length of audio ring buffer in seconds')
parser.add_argument(
    '-dm', '--dims', type=int, default=5, help='Number of sample dimensions, e.g 1,4,8 second long samples')      
parser.add_argument(
    '-sl', '--size_limit', type=int, default=10000, help='Approximate limit (in mbs) for file storage') 

args = parser.parse_args()
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1

if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
    
if args.samplerate is None:
    device_info = sd.query_devices(args.device, 'input')
    args.samplerate = device_info['default_samplerate']

osc_startup()
osc_udp_client("127.0.0.1", 1050, "Kodama")
q = queue.Queue()
sa = aca.seg_analysis(args.device, args.plot, args.dims, args.audio_length, args.samplerate, args.size_limit)
print("Kodama has started. Ignore the decomposition/_pca.py:456 runtime warning.")

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata)

class main_thread (threading.Thread):
   def __init__(self, threadID, name, counter):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
   def run(self):
        if self.threadID == 1: # Data cleanup and cluster prunning.
            while True:
                sa.consolidate()
        if self.threadID == 2: # OSC processing (which must run constantly).
            while True:
                osc_process()
        if self.threadID == 3: # Audio stream and buffering. We fill the ringbuffer with the incoming audio data unless we have an empty queue.
            stream = sd.InputStream(blocksize = int(args.samplerate),
            device=args.device, channels=max(args.channels),
            samplerate=args.samplerate, callback=audio_callback)
            with stream:
                while True:
                    try:
                        data = q.get_nowait()
                        sa.fill_ringbuff(data,args.blocksize)
                    except queue.Empty:
                        pass

class sdt(threading.Thread): # Threads for each 'dimension', or sample sizetype.
   def __init__(self, threadID, name, counter, dim):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
      self.dim = dim
   def run(self):
        while True:
            #print("THREAD DIM RUN: " + str(self.dim)) # Helpful for observing threads running behaviours.
            sa.seg_dim_analysis(self.dim)
            msg = oscbuildparse.OSCMessage("/kodama/", None, sa.get_nearest(self.dim))
            osc_send(msg, "Kodama")
            time.sleep(args.audio_length/(self.dim+1))

# RUNNING THE THREADS #

thr_sa = main_thread(1, "Thread-1 PRUNNING", 1)
thr_osc_process = main_thread(2, "Thread-2 OSC PROCESS", 2)
thr_audio = main_thread(3, "Thread-3 AUDIO RECORDING", 3)
sd_threads = []

for dim in range(0,args.dims):
    sd_threads.append(sdt(dim+3, "Thread SEGMENT-" + str(dim + 3), dim+3, dim))
    sd_threads[dim].start()

thr_audio.start()
thr_sa.start()
thr_osc_process.start()

