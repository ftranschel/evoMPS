#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module is part of
an extension of evoMPS by adding
dissipative dynmaics based on
Monte-Carlo methods.

This part is the server file
for the distributed computing
framework that utilizes parallel
processing to speed up dissipative
dynamics.

@author: F.W.G. Transchel
"""

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
import tdvp_common_diss as tm
import nullspace as ns
import matmul as mm
import matplotlib.pyplot as plt

authkey = "SECRET"
ip = "127.0.0.1"
port = 5678
internal_call = True
global num_results
num_results = 0
global num_started_jobs
num_started_jobs = 0
global total_iterations
global result_list
result_list = []

def do_calculations(shared_job_q,shared_result_q,codebase):

    global num_results
    global num_started_jobs
    
    num_results = 0
    num_started_jobs = 0
            
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
    
    for i in xrange(1,total_iterations+1):
        print "Sending job#" + str(i)
        #shared_job_q.put(str("print \"test: This is process #" + str(i) + "/" + str(N) + "\"\ndef crunching_zombie(n):\n\treturn n*n\nresult = crunching_zombie("+str(i)+")\nprint result\n"))
        shared_job_q.put("result = sample_path(evoMPS_TDVP_Generic_Dissipative,"+str(i)+")")

    print "==="
    # Wait until all results are ready in shared_result_q
    num_results = 0
    num_started_jobs = total_iterations
    
    return do_calculation_prompt(shared_job_q,shared_result_q)

def add_jobs(shared_job_q,shared_result_q):
    global result_list
    global num_results
    global num_started_jobs
    print "Num_started_jobs:", num_started_jobs
    try:
        num_jobs = int(raw_input("\nMPS> How many new job instances?"))                        
        for l in xrange(num_started_jobs+1,num_started_jobs+num_jobs+1):
            #print "Sending job#" + str(l)
            #shared_job_q.put(str("print \"test: This is process #" + str(i) + "/" + str(N) + "\"\ndef crunching_zombie(n):\n\treturn n*n\nresult = crunching_zombie("+str(i)+")\nprint result\n"))
            shared_job_q.put("result = sample_path(evoMPS_TDVP_Generic_Dissipative,"+str(l)+")")
        print "\nPut", num_jobs, "jobs to the queue."
        num_started_jobs+=num_jobs
    except Exception, e:
            print "Error: input does not appear to be integer",e

def show_help():
    # Show the help
    print "\n\tFor now this only gives the list of commands.\n\n"
    print "\tload: imports the 'tdvp_gen' and 'tdvp_gen_diss' modules when\n\tthey are changed."
    print "\tadd: Add jobs to the queue."
    print "\treset: Delete all collected results and purge the queue."
    print "\tanalysis: calculate the observables and plot some data."
    print "\thelp: this help screen."
    print "\texit: shut down the computation server.\n"
    print "\tYou can also set the system parameters by supplying python Code. A list\n\tof parameters follows.\n"

def runserver():
    # Start a shared manager server and access its queues
    #manager = make_server_manager(port, authkey)
    
    manager = make_server_manager(port,authkey)     
    shared_job_q = manager.get_job_q()
    shared_result_q = manager.get_result_q()
    codebase = manager.codebase()
    load(codebase)
    
    global num_results
    global result_list
    
    # Bring up the |MPS> prompt.
    
    while True:        
        if shared_result_q.empty():
            print "\n\nStatus: (" + str(len(result_list)) + "/" + str(num_started_jobs) + ") results collected."
            try:
                cmd = str(raw_input("\nMPS>"))
                if cmd == "add" or cmd == "run":
                    add_jobs(shared_job_q,shared_result_q)
                    continue
                if cmd == "stop" or cmd == "exit":
                    print "==="
                    print "Clearing up..."
                    break
                if cmd == "help":
                    show_help()
                    continue
                if cmd == "zombie apocalypse":
                    print "Sorry, not implemented."
                    time.sleep(2)
                    print "Yet."
                    continue
                if cmd == "reset":
                    print "Not implemented."
                    continue
                if cmd == "analysis" or cmd == "ana":
                    cdbs = manager.codebase().__str__() + "\n"
                    code = cdbs[12:-3]
                    decoded = code.replace("\\n","\n")
                    decoded = decoded.replace("\\r","\r")
                    execfile("../fh_analysis.py")
                    continue
                if cmd == "stop" or cmd == "exit":
                    break
                exec(cmd)
            except Exception, e:
                print "Unexpected error:", sys.exc_info()[0]
                print e
                print "You can type 'help' for a list of commands."
            time.sleep(.1)
            continue
        else:
            result_list.append(shared_result_q.get())
            continue
    
    # Sleep a bit before shutting down the server - to give clients time to
    # realize the job queue is empty and exit in an orderly way.
    print "SERVER shutting down."
    time.sleep(2)
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
    
    def __init__(self):
        pass
   
    def get_codebase(self):
        return self.codebase

    
def load(codebase):
    'Update method. When the program code of the tdvp suite got changed, call this.'
    
    print "Loading program code into shared memory..."
    
    # Load the current versions of tdvp_diss and tdvp_gen
    
    tdvp_com = ""
    #tdvp_com_fh = open("tdvp_common.py","r")
    #tdvp_com = tdvp_com_fh.read()
    #tdvp_com_fh.close()    
    
    tdvp_gen = ""
    #tdvp_gen_fh = open("tdvp_gen.py","r")
    #tdvp_gen = tdvp_gen_fh.read()
    #tdvp_gen_fh.close()
    #exec(tdvp_gen)
    
    #tdvp_diss_fh = open("../tdvp_gen_diss_ash_old.py","r")
    #tdvp_diss_fh = open("tdvp_gen_diss_fermi_hubbard_2d.py","r")
    tdvp_diss_fh = open("tdvp_gen_diss_fermi_hubbard_new.py","r")
    #tdvp_diss_fh = open("D:/promotion/repos/mpsampling/evoMPS/evoMPS/tdvp_gen_diss.py","r")
    tdvp_diss = tdvp_diss_fh.read()
    tdvp_diss_fh.close()
    #exec(tdvp_diss)

    #codec = tdvp_com + str("\n\r") + tdvp_gen + str("\n\r") + tdvp_diss
    codec = tdvp_diss
    
    exec(codec)

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
