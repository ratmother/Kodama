import threading
import time
import audio_comparative_analysis as aca
from osc4py3.as_eventloop import *
from osc4py3 import oscbuildparse

exitFlag = 0
osc_startup()
osc_udp_client("127.0.0.1", 1050, "aclientname")
msg = oscbuildparse.OSCMessage("/test/me", ",sif", ["teaasaaaaaat", 672, 8.871])
osc_send(msg, "aclientname")
sa = aca.seg_analysis(11, 2, 8448, 10, 'feature_data.pkl')

class main_thread (threading.Thread):
   def __init__(self, threadID, name, counter):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
   def run(self):
        if self.threadID == 1:
            print("run osc ")
            osc_process()
            buffer = str(sa.get_buffer())
            buffer = (buffer.replace('[', ''))
            buffer = (buffer.replace(']', ''))
            msg = oscbuildparse.OSCMessage("/KODAMA/dim1", None , ["dim1", buffer])
            osc_send(msg, "aclientname")
        if self.threadID == 2:
            print("run seg analyssi")
            sa.run_analysis()

class sdt(threading.Thread): # Segment dimensional thread. 
   def __init__(self, threadID, name, counter, dim):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
      self.dim = dim
   def run(self):
        print("run DIM analyssi" + str(self.dim))
        sa.seg_dim_analysis(self.dim)


sa.segment_dir('./data/learning_sounds')
#sa.read_pkl()
# Create new threads
thr_osc = main_thread(1, "Thread-1", 1)
thr_sa = main_thread(2, "Thread-2", 2)

 #Start new Threads
threads = []
sd_threads = []

thr_sa.start()
thr_sa.join()
thr_osc.start()
threads.append(thr_osc)
threads.append(thr_sa)

for dim in range(0,4):
    sd_threads.append(sdt(dim+3, "sdt Thread-" + str(dim + 3), dim+3, dim))
    sd_threads[dim].start()
    threads.append(sd_threads[dim])

for t in threads:
    t.join()

while True:
    thr_osc.run()
    thr_sa.run()
    thr_sa.join()
    for dim in range(0,4):
        sd_threads[dim].run()
    for t in threads:
        t.join() 