#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module is part of
an extension of evoMPS by adding
dissipative dynmaics based on
Monte-Carlo methods.

This part is the client file
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
import traceback
import io
import multiprocessing.sharedctypes
from ast import literal_eval
import string as STR
import pickle as pic
import numpy as np
global np
import scipy as sp
global sp
import scipy.linalg as la
import nullspace as ns
import matmul as m
import tdvp_common_diss as tm
import matmul as mm
import tdvp_gen as TDVP
import scipy.sparse as spp

authkey = "SECRET"
port = 5678
ip = '127.0.0.1'
internal_call = True
# This is used to tell tdvp_gen_diss.py
#(and other dynamically loaded modules)
# to not execute code on its own.

def worker(job_q, result_q,codebase):
    """ A worker function to be launched in a separate process. Takes jobs from
        job_q - each job a list of numbers to factorize. When the job is done,
        the result (dict mapping number -> list of factors) is placed into
        result_q. Runs until job_q is empty.
    """ 
    
    try:
        #print codebase
        exec(codebase,globals(),locals())
    except:
        print "Unexpected error:", sys.exc_info()[0]
        traceback.print_exc()
        raise
    
    while True:
        try:
            job = job_q.get_nowait()
            # job = data for executing something...
            # idea: serialize the code and just have it executed distributively
            result = {}
            try:
                exec(job)
            except:
                print "Unexpected error:", sys.exc_info()[0]
                traceback.print_exc()
                raise
            #print result
            result_q.put(result)
            time.sleep(2)
        except qu.Empty:
            return
        except:
            print "Unexpected error:", sys.exc_info()[0]
            return
            #raise
            
def scheduler(shared_job_q, shared_result_q, codebase, nprocs):
    """ Split the work with jobs in shared_job_q and results in
        shared_result_q into several processes. Launch each process with
        factorizer_worker as the worker function, and wait until all are
        finished.
    """
    #print codebase
    #exec(codebase)
    procs = []
    for i in range(nprocs):
        p = mp.Process(
                target=worker,
                args=(shared_job_q, shared_result_q, codebase))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()
        
          
def runclient():
    """ This is the __main__ function of client.py that connects to a server
        and distributes jobs to the scheduler().
    """
    max_tries = 0
    max_tries_limit = 5
    while True:
        try:
            manager = make_client_manager(ip,port,authkey)
            job_q = manager.get_job_q()
            result_q = manager.get_result_q()
            cdbs = manager.codebase().__str__() + "\n"
            code = cdbs[12:-3] #workaround to get rid of control chars
            decoded = code.replace("\\n","\n")
            decoded = decoded.replace("\\r","\r")
            #exec(decoded,globals(),locals())
            #tdvp_obj = tdvp_diss()
            #print code
            scheduler(job_q, result_q, decoded, (2))
            print "All available jobs finished."
            print "==="
            time.sleep(5)
        except socket.error:
            print "No answer from server. Trying again...."
            max_tries += 1
            time.sleep(5)
            if max_tries >= max_tries_limit:
                break
            else:
                continue
    print "Process aborted from too many failed connection tries. Exiting."
        
#from multiprocessing.managers import BaseManager

class ServerQueueManager(mpm.SyncManager):
    pass
    
def make_client_manager(ip, port, authkey):
    """ Create a manager for a client. This manager connects to a server on the
        given address and exposes the get_job_q and get_result_q methods for
        accessing the shared queues from the server.
        Return a manager object.
    """

    ServerQueueManager.register('get_job_q')
    ServerQueueManager.register('get_result_q')
    ServerQueueManager.register('codebase')   
    
    manager = ServerQueueManager(address=(ip, port), authkey=authkey)
    manager.connect()

    print 'Client connected to %s:%s' % (ip, port)
    return manager
    
def dist_process(n):
    return n    

def importCode(code,name,add_to_sys_modules=0):
    """
    Import dynamically generated code as a module. code is the
    object containing the code (a string, a file handle or an
    actual compiled code object, same types as accepted by an
    exec statement). The name is the name to give to the module,
    and the final argument says wheter to add it to sys.modules
    or not. If it is added, a subsequent import statement using
    name will return this module. If it is not added to sys.modules
    import will try to load it in the normal fashion.

    import foo

    is equivalent to

    foofile = open("/path/to/foo.py")
    foo = importCode(foofile,"foo",1)

    Returns a newly generated module.
    """
    import sys,imp

    module = imp.new_module(name)

    exec code in module.__dict__
    if add_to_sys_modules:
        sys.modules[name] = module

    return module
    
    
if __name__ == '__main__':
    print "==="
    print "This is the mpsampling distributed computation CLIENT."
    print "==="
    print "Using " +  str(mp.cpu_count()) + " cores."
    print "==="
    runclient()
