import Queue as qu
import multiprocessing as mp
import multiprocessing.managers as mpm
import time
import socket
import sys
import io
import multiprocessing.sharedctypes as mpmct
import pickle as pic
import numpy as np
import scipy as sp
import scipy.linalg as la
import nullspace as ns
import matmul as m
import matplotlib.pyplot as plt

authkey = "testcase"
ip = "130.75.25.161"
port = 5678

def do_calculations(shared_job_q,shared_result_q,codebase):
    #nums = make_nums(N)

    # The numbers are split into chunks. Each chunk is pushed into the job
    # queue.
    #chunksize = 3
    #for i in range(0, len(nums), chunksize):
            
    cdbs = codebase.__str__() + "\n"
    code = cdbs[12:-3]
    decoded = code.replace("\\n","\n")
    decoded = decoded.replace("\\r","\r")         
    exec(decoded)
    #global Kx
    #Kx = sp.zeros((total_iterations,total_steps),dtype=sp.complex128)
    global result_list
    result_list = []
    
    # Measure execution time

    t0 = time.time()    
    
    for i in xrange(1,total_iterations+1):
        print "Sending job#" + str(i)
        #shared_job_q.put(str("print \"test: This is process #" + str(i) + "/" + str(N) + "\"\ndef crunching_zombie(n):\n\treturn n*n\nresult = crunching_zombie("+str(i)+")\nprint result\n"))
        shared_job_q.put("result = sample_path(EvoMPS_TDVP_Generic,"+str(i)+")")

    print "==="
    # Wait until all results are ready in shared_result_q
    numresults = 0
    while numresults < total_iterations:
        result_list.append(shared_result_q.get())
        print "Got result (" + str(numresults+1) + "/" + str(total_iterations) + ")"
        #print numresults
        numresults += 1
        if shared_result_q.empty():
            #print "(Queue empty.)"
            time.sleep(1)
   
    #print result_list         
    print "==="
    print "Computation complete."
    print "Operation took " + str((time.time() - t0)) + " s."
    

def show_help():
    # Show the help
    print "\n\tFor now this only gives the list of commands.\n\n"
    print "\tload: imports the 'tdvp_gen' and 'dissipative_montecarlo' modules when\n\tthey are changed."
    print "\trun: start the computation by putting jobs into the queue."
    print "\tlrun: load new code and start the computation directly after."
    print "\tanalysis: calculate the observables and plot some data."
    print "\thelp: this help screen."
    print "\texit: shut down the computation server.\n\n"
    print "\tThere are more commands but I cannot disclose them. Have fun!\n"
    print "\tYou can also set the system parameters by supplying python Code. A list\n\tof parameters follows.\n"

def runserver():
    # Start a shared manager server and access its queues
    #manager = make_server_manager(port, authkey)
    
    manager = make_server_manager(port,authkey)
     
    shared_job_q = manager.get_job_q()
    shared_result_q = manager.get_result_q()
    codebase = manager.codebase()
    load(manager,codebase)
    #exec(codebase.__str__().__repr__())
    print "done."

    """
    Initialize the shared variable codebase with the current versions
    of the modules tdvp_gen and tdvp_diss
    """
    
    while True:
        try:
            cmd = str(raw_input("\nMPS>"))
            if cmd == "run":
                do_calculations(shared_job_q,shared_result_q,codebase)
                continue
            if cmd == "lrun":
                load(manager,codebase)
                print "done."
                do_calculations(shared_job_q,shared_result_q,codebase)
                continue
            if cmd == "load":
                load(manager,codebase)
                print "done."
                continue
            if cmd == "help":
                show_help()
                continue
            if cmd == "zombie apocalypse":
                print "Sorry, not implemented."
                time.sleep(2)
                print "Yet."
                continue
            if cmd == "analysis":
                cdbs = manager.codebase().__str__() + "\n"
                code = cdbs[12:-3]
                decoded = code.replace("\\n","\n")
                decoded = decoded.replace("\\r","\r")
                execfile("analysis.py")
                continue
            if cmd == "exit":
                break
            print ""
            exec(cmd)
        except Exception, e:
            print "Unexpected error:", sys.exc_info()[0]
            print e
            print "You can type 'help' for a list of commands."
    
    # Sleep a bit before shutting down the server - to give clients time to
    # realize the job queue is empty and exit in an orderly way.
    print "SERVER shutting down."
    time.sleep(5)
    manager.shutdown()
    print "Shutdown complete. Bye bye."

class JobQueueManager(mpm.SyncManager):
    pass

def jq_l():
    return job_q
def rq_l():
    return result_q
def cdbs():
    return codebase

def make_server_manager(port, authkey):
    """ Create a manager for the server, listening on the given port.
        Return a manager object with get_job_q and get_result_q methods.
    """

    # This is based on the examples in the official docs of multiprocessing.
    # get_{job|result}_q return synchronized proxies for the actual Queue
    # objects.

    JobQueueManager.register('get_job_q', callable = jq_l)
    JobQueueManager.register('get_result_q', callable = rq_l)
    JobQueueManager.register('codebase', callable = cdbs)
    #JobQueueManager.register('codebase', callable = cdbs)
    
    manager = JobQueueManager(address=(ip, port), authkey=authkey) 
    
    manager.start()
    
    print 'Server started at port %s \n===' % port
    return manager

class Codebase():
    'Codebase is the class that makes sure the tdvp code gets distributed to the clients'

    codebase = ""
    
    def __init__(self,manager):
        'Constructor of the Codebase class'
        #self.update(manager)
   
    def get_codebase(self):
        return self.codebase

    
def load(manager, codebase):
    'Update method. When the program code of the tdvp suite got changed, call this.'
    
    print "Loading program code into shared memory..."
    
    # Load the current versions of tdvp_diss and tdvp_gen
    
    tdvp_gen_fh = open("tdvp_gen.py","r")
    tdvp_gen = tdvp_gen_fh.read()
    tdvp_gen_fh.close()
    #exec(tdvp_gen)
    
    tdvp_diss_fh = open("tdvp_diss.py","r")
    tdvp_diss = tdvp_diss_fh.read()
    tdvp_diss_fh.close()
    #exec(tdvp_diss)

    codec = tdvp_gen + str("\n\r") + tdvp_diss

    print "Code loaded. Distributing and syncing clients..."
      
    # We succesfully loaded the code base, but we ned to distribute it.
    
    while True:
        try:
            codebase.pop(0)
        except:
            break
        
    # codebase should now be empty :)
    
    codebase.fromstring(codec)
    #print codebase
        
job_q = qu.Queue()
result_q = qu.Queue()
#global codebase
codebase = mpm.Array("c","test")
    
def make_nums(N):
    """ Create N large numbers to factorize.
    """
    nums = [999999999999]
    for i in xrange(N):
        nums.append(nums[-1] + 2)
    return nums    
    
if __name__ == '__main__':
    print "==="
    print "This is the mpsampling distributed computation SERVER."
    print "==="
    runserver()
