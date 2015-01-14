#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A demonstration of evoMPS by simulation of quench dynamics
for the transverse Ising model.

@author: Ashley Milsted
"""

import numpy as np
import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import itertools
import sys as sys # If we happen to want interactive output formatting like process indicators later
#import evoMPS.tdvp_gen #Problematic, since the module cannot be distributed in this way

"""
First, we define our Hamiltonian and some observables.
"""

global h_ext

def h_ext(n, s, t):
    """The single-site Hamiltonian representing the external field.
    
    -h * sigmaX_s,t.
    
    The global variable h determines the strength.
    """
    if s == t:
        return 0
    else:
        return -h        

global h_nn

def h_nn(n, s, t, u, v):
    """The nearest neighbour Hamiltonian representing the interaction.

    -J * sigmaZ_n_s,t * sigmaZ_n+1_u,v.
    
    The global variable J determines the strength.
    """
    
    """
    haar = 0
    
    if s != t:
        haar = -h

    if s == u and t == v:
        return -J * (-1)**s * (-1)**t + haar
    else:
        return haar
    """

    """
    The hamiltonian is -sigma_x*sigma_x + lambda (L_1+L_2)
    """

    # The hamiltonian is Sum_i=1^n-1 s_nx*s_n+1x + Lambda s_zn*s_zn+1
    
    value = h    
    
    #if n != qn:
        # could this be qn?
    if n != N:
        value+= (x_ss(n,s,t)*x_ss(n + 1,u,v)) + Lambda * (z_ss(n,s,t)*z_ss(n + 1,u,v))
    if n != 1:
        value+= (x_ss(n - 1,s,t)*x_ss(n,u,v)) + Lambda * (z_ss(n - 1,s,t)*z_ss(n ,u,v))
    
    """
    if s == t:
        return 1
    else:
        return 0
    """
    return value
#def h_nn()    
    
global z_ss
def z_ss(n, s, t):
    """Spin observable: z-direction
    """
    if s == t:
        return (-1)**s
    else:
        return 0
      
global x_ss
def x_ss(n, s, t):
    """Spin observable: x-direction
    """
    if s == t:
        return 0
    else:
        return 1
        
global y_ss
def y_ss(n, s, t):
    """Spin observable: y-direction
    """
    if s == t:
        return 0
    else:
        return 1.j * (-1)**t


"""
 If taken to 0, the whole MC part of the algorithm is skipped
 (e.g. for benchmarking or comparison applications)
"""

global MC
MC = 1

"""
Next, we set up some global variables to be used as parameters to 
the evoMPS class.
"""

global N
N = 16 #The length of the finite spin chain.

global qn
qn = 2 #The site Hilbert space dimension

global h
h =  0

global Lambda
Lambda = 1

global step
step = 0.01

global total_steps
total_steps = 200

global total_iterations
total_iterations = 800

np.random.seed(1111)
# We don't need the global here because the seed actually is global anyway.

global convergence_check
convergence_check = 0

global noise_coeff
noise_coeff = 0

global bond_dim
bond_dim = 16

global l_nn_n
def l_nn_n(N):
    """Lindblad operator for the n-th site.
    That is, it should give the identity operator for all other sites and
    a specified operator for site n  
    """
    
    if N > 5:
        #print N
        pass
        
    def l_nn_N(n,i,j,u,v): 
        
      # i,j: Matrix indices of first site
      # s,t: matrix indices of second site        
      
      value = z_ss(n,i,j)
      
      return value      
      
    return l_nn_N

global l_nns
l_nns = [l_nn_n(1), l_nn_n(2),l_nn_n(3),l_nn_n(4),l_nn_n(5),l_nn_n(6),l_nn_n(7),l_nn_n(8),l_nn_n(9),l_nn_n(10),l_nn_n(11),l_nn_n(12),l_nn_n(13),l_nn_n(14),l_nn_n(15),l_nn_n(16)]

# Lets init the lattice

global D
D = sp.empty(N + 1, dtype=sp.int32)
D.fill(bond_dim)

# And init the observable arrays

global Kx
Kx = sp.zeros((total_iterations,total_steps),dtype=sp.complex128)
global Mx
Mx = sp.zeros((total_iterations,N+1,total_steps),dtype=sp.complex128)

global KMean
KMean = sp.zeros((total_steps),dtype=sp.complex128) 
global MMean
MMean = sp.zeros((total_steps),dtype=sp.complex128)
global MsMean
MsMean = sp.zeros((N,total_steps),dtype=sp.complex128)

global KVar
KVar = sp.zeros((total_steps),dtype=sp.complex128)
global MVar
MVar = sp.zeros((total_steps),dtype=sp.complex128)
global MsVar
MsVar = sp.zeros((N,total_steps),dtype=sp.complex128)

#We're good to go!

def sample_path(evomps,num):
    
    print ">> Starting job #"+str(num)
    
    
    if MC == 0:
        print "Warning: MC part is omitted and you will end up with a pure TDVP evolution."
    
    return_dict = {}
    
    #for it in xrange(total_iterations):
    
    t = 0.    
    
    q = sp.empty(N + 1, dtype=sp.int32)
    q.fill(qn)    
    
    """
    Now we are ready to create an instance of the evoMPS class.
    """
    s = evomps(N, D, q, h_nn)
    
    # We created an MPS instance, but we want to sample from different starting states.
    #s.add_noise(noise_coeff)
    
    """
    Tell evoMPS about our Hamiltonian.
    """
    
    s.h_nn = h_nn    
    
    """
    Print a table header.
    """
    
    #print "Bond dimensions: " + str(s.D)
    #print
    #col_heads = ["Step", "t", "K[1]", "dK[1]", 
    #             "sig_x_3", "sig_y_3", "sig_z_3",
    #             "E_vn_3,4", "M_x", "Next step",
    #             "(itr", "delta", "delta_chk)"] #These last three are for testing the midpoint method.
    #print "\t".join(col_heads)
    #print    
    
    reCF = []
    reNorm = []

    T = sp.zeros((total_steps), dtype=sp.complex128)
    K1 = sp.zeros((total_steps), dtype=sp.complex128)
    lN = sp.zeros((total_steps), dtype=sp.complex128)

    Sx_3 = sp.zeros((total_steps), dtype=sp.complex128) #Observables for site 3.
    Sy_3 = sp.zeros((total_steps), dtype=sp.complex128)
    Sz_3 = sp.zeros((total_steps), dtype=sp.complex128)

    Ms = sp.zeros((N+1,total_steps), dtype=sp.complex128)   #Magnetization in x-direction.
    
    """
    #print("\n\n === Running sampling path " + str(it+1) + " ===")
    """

    for i in xrange(total_steps):
        #print "Noch da! (" + str(i) + ")"
        """
        print >> sys.stdout, "\r"
        print >> sys.stdout, "\r" + "Path number #" + str(it + 1) + "/" + str(total_iterations) + (", Iteration step #" + str(i) + "/" + str(total_steps) + "\n"),
        sys.stdout.flush()
        """
        T[i] = t
        row = [str(i)]
        row.append(str(t))
    
        #print "===BEFORE UPDATE"
        #print s.A
        #print "==="
        s.update()
        #t += 1.j * sp.conj(step) 
        
        """
        for z in xrange(N):
            #print z+1
            redden[z] = s.density_1s(z+1)
        """   
           
        #print "=== AFTER UPDATE / BEFORE RANDOM"
        #b4 = s.A
        #print s.A    
        #print "==="
        
        s.take_step_dissipative(step,MC,l_nns)   
    
        #print "=== AFTER RANDOM"
        #print s.A
        #print b4
        #print "==="
            
        K1[i] = s.K[1][0, 0]    
        row.append("%.15g" % K1[i].real)
    
        if i > 0:        
            dK1 = K1[i].real - K1[i - 1].real
        else:
            dK1 = K1[i]
    
        row.append("%.2e" % (dK1.real))
        
        """
        Compute observables!
        """
    
        Sx_3[i] = s.expect_1s(x_ss, 2) #Spin observables for site 2.
        Sy_3[i] = s.expect_1s(y_ss, 2)
        Sz_3[i] = s.expect_1s(z_ss, 2)
        row.append("%.3g" % Sx_3[i].real)
        row.append("%.3g" % Sy_3[i].real)
        row.append("%.3g" % Sz_3[i].real)
    
        m = 0   #x-Magnetization
        for n in xrange(1, N + 1):
            m += s.expect_1s(x_ss, n) 
            Ms[n,i] = s.expect_1s(x_ss, n)
        
        row.append("%.9g" % m.real)
        Ms[0,i] = m
        
    
        row.append(str(step))
    
        """
        Carry out next step!
        """
        #print "\t".join(row)
        #s.take_step_dissipative(step,l_nns)     
        
        # Real time evolution
        t += 1.j * sp.conj(step) 
        
        #Imaginary time evolution
        #t += 1 * sp.conj(step)     
    
        # We need to find out whether the iteration has converged.
        # To this end, we will compare the last five steps.
    
        #if (abs(sum(K1[-5:])/5 - K1[i]) < (pow(step,2)) and convergence_check == 1):
        #    print "Found converging energy value. Aborting algorithm at step " + str(i)
        #    break
    
        # Just for output of the current step to estimate completion.
        #print "Step: " + str(int(np.lib.type_check.real(-1.j*t/step)))
        #print K1
        #print np.sqrt(K1[i]*np.conjugate(K1[i]))
        
    #Kx[it] = K1[::]
    #for j in xrange(0,N+1):
    #    Mx[it,j] = Ms[j,::]
        
    #print Ks
    return_dict["K1"] = K1
    return_dict["M1"] = Ms
    print "<< Job #"+str(num)+" done."
    return return_dict
    
    """    
When not called from the grid, perform singular operation based on standard
variables.
"""

try:
    internal_call
except NameError:
    internal_call = 0

if(not internal_call):
    execfile('/home/ftransch/repos/mpsampling/evoMPS/evoMPS/tdvp_gen_diss.py') #Problematic, since the module cannot be distributed in this way
    execfile('/home/ftransch/repos/mpsampling/evoMPS/evoMPS/tdvp_diss_analysis.py')
    """    
    print "The dissipative module was not called from the computational grid."
    print "Assuming singular operation..."
    bd = np.ndarray((N+1), dtype=sp.int32)
    bd[:] = bond_dim
    q = np.ndarray((N+1), dtype=sp.int32)
    q[:] = qn
    #for i in xrange(1,20):
    ham_sites = None
    sample_path(EvoMPS_TDVP_Generic_Dissipative,1)
    """
