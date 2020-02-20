import audio_comparative_analysis as aca
from osc4py3.as_eventloop import *
from osc4py3 import oscbuildparse
import threading
import time
import numpy as np 


exitFlag = 0

osc_startup()
osc_udp_client("127.0.0.1", 1050, "Kodama")


sa = aca.seg_analysis(6,'feature_data.pkl', True)
counter = 0

class main_thread (threading.Thread):
   def __init__(self, threadID, name, counter):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
   def run(self):
        if self.threadID == 1: # Segmentation
            sa.parcelization()
        if self.threadID == 2: # OSC
            while True:
                osc_process()
                if sa.get_buffer(2) is not None:
                    msg = oscbuildparse.OSCMessage("/main/", None, sa.get_buffer(2))
                    osc_send(msg, "Kodama")


class sdt(threading.Thread): # Segment dimensional thread. 
   def __init__(self, threadID, name, counter, dim):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
      self.dim = dim
   def run(self):
        print("THREAD DIM RUN: " + str(self.dim))
        sa.seg_dim_analysis(self.dim)


sa.segment_dir('./data/learning_sounds')
##sa.read_pkl('./data/learning_sounds')

# Create new threads
thr_sa = main_thread(1, "Thread-2 ANALYSIS", 1)
thr_osc_process = main_thread(2, "Thread-3 OSC", 2)

 #Start new Threads
threads = []
sd_threads = []

thr_sa.start()
thr_sa.join()
thr_osc_process.start()
threads.append(thr_sa)

for dim in range(0,aca.pl.get_segment_amounts() - 1):
    sd_threads.append(sdt(dim+3, "Thread SEGMENT" + str(dim + 3), dim+3, dim))
    sd_threads[dim].start()
    threads.append(sd_threads[dim])

for t in threads:
    t.join()

while True:
    thr_sa.run()
    thr_sa.join()
    for dim in range(0,aca.pl.get_segment_amounts() - 1):
        sd_threads[dim].run() 
    for t in threads:
        t.join() 

