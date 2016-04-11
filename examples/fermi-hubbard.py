#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An extension of evoMPS by adding
dissipative dynmaics based on
Monte-Carlo methods.

Can be called as stand-alone
evoMPS module or by the distributed
computing framework.

@author: F.W.G. Transchel
"""

import numpy as np
global np
import scipy as sp
global sp
import scipy.linalg as la
import evoMPS.mps_gen as mg
import tdvp_common_diss as tm
import evoMPS.matmul as mm
import evoMPS.tdvp_gen as TDVP
import itertools
import sys as sys # If we happen to want interactive output formatting like process indicators later

"""
First, we set up some global variables to be used as parameters to 
the evoMPS clas.
"""

global N
N = 8 #The length of the finite spin chain.

global qn
qn = 4 #The site Hilbert space dimension

global h
h =  0

# System parameter #1

global Lambda
Lambda = 1

# System parameter #2

global epsilon
epsilon = 0.5 * 2.0

# System parameter #3

global mu
mu = 1


global step
step = 0.005

# Coupling parameter #4

global sqrtgamma1
sqrtgamma1 = 1

# Coupling parameter #5

global sqrtgamma2
sqrtgamma2 = sp.sqrt(2)
# Coupling parameter #6

global U
U = 1

# Coupling parameter #7

global t
t = 1

global sqrtepsilon
sqrtepsilon = np.sqrt(epsilon)

global meas_steps
meas_steps = 100

global total_steps
total_steps = 100

global total_iterations
total_iterations = 16
#total_iterations = 1

np.random.seed(2222)

global convergence_check
convergence_check = 0

global noise_coeff
noise_coeff = 0

global bond_dim
bond_dim = 8          

l_proto = sp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
l_proto_id = sp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);

global l_nns
global l_nns_id
#l_nns = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27, l28, l29, l30, l31, l32, l33, l34, l35, l36, l37, l38, l39, l40, l14, l42, l43, l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55, l56, l57, l58, l59, l60, l61, l62, l63, l64, l65, l66, l67, l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80, l81, l82, l83, l84, l85, l86, l87, l88, l89, l90, l91, l92, l93, l94, l95, l96, l97, l98, l99, l100]
l_nns = [l_proto] * (N)
l_nns_id = [l_proto_id] * (N)

# Lets init the lattice

global D
D = sp.empty(N + 1, dtype=sp.int32)
D.fill(bond_dim)

# And init the observable arrays

global rand
rand = np.random
global p1
p1 = sp.array([[1 + 0.j, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
global p2
p2 = sp.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
global p3
p3 = sp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]);
global p4
p4 = sp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]);

# Define measurement operators

global MP
MP = sp.zeros((meas_steps,4),dtype=sp.complex128)
global DO
DO = sp.zeros((meas_steps),dtype=sp.complex128)
global Ns
Ns = sp.zeros((meas_steps,3),dtype=sp.complex128)
global maf
maf = sp.zeros((meas_steps),dtype=sp.complex128)
global mafabs
mafabs = sp.zeros((meas_steps),dtype=sp.complex128)
global AFB
AFB = sp.zeros((meas_steps),dtype=sp.complex128)
global ENT
ENT = sp.zeros((meas_steps),dtype=sp.complex128)

# Define necessary operators...

p1 = sp.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
p2 = sp.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
p3 = sp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]);
p4 = sp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]);

EINS1 = sp.eye(2);
EINS2 = sp.eye(4);
EINS3 = sp.eye(8);
EINS4 = sp.eye(16);

PLUS = [[0, 0],[1, 0]];
MINUS = [[0, 1],[0, 0]];
PLUSMINUS = [[0, 0],[0, 1]];
MINUSPLUS = [[1, 0],[0, 0]];

Z = [[1, 0],[0, -1]];
X = [[0, 1],[1, 0]];
Y = [[0, -1.j],[1.j, 0]];

C1UP = sp.kron(MINUS, EINS3);
C1DAGUP = sp.kron(PLUS, EINS3);
C1DOWN = sp.kron(Z, sp.kron(MINUS, EINS2));
C1DAGDOWN = sp.kron(Z, sp.kron(PLUS, EINS2));
C2UP = sp.kron(Z, sp.kron(Z, sp.kron(MINUS, EINS1)));
C2DAGUP = sp.kron(Z, sp.kron(Z, sp.kron(PLUS, EINS1)));
C2DOWN = sp.kron(Z, sp.kron(Z, sp.kron(Z, MINUS)));
C2DAGDOWN = sp.kron(Z, sp.kron(Z, sp.kron(Z, PLUS)));

N1UP = sp.dot(C1DAGUP,C1UP);
N1DOWN = sp.dot(C1DAGDOWN,C1DOWN);
N2UP = sp.dot(C2DAGUP,C2UP);
N2DOWN = sp.dot(C2DAGDOWN,C2DOWN);

P1UP = sp.dot(C1DAGUP,C1UP) - sp.dot(C1DAGUP,sp.dot(C1UP,sp.dot(C1DAGDOWN,C1DOWN)));
P1DOWN = sp.dot(C1DAGDOWN,C1DOWN) - sp.dot(C1DAGDOWN,sp.dot(C1DOWN,sp.dot(C1DAGUP,C1UP)));
P2UP = sp.dot(C2DAGUP,C2UP) - sp.dot(C2DAGUP,sp.dot(C2UP,sp.dot(C2DAGDOWN,C2DOWN)));
P2DOWN = sp.dot(C2DAGDOWN,C2DOWN) - sp.dot(C2DAGDOWN,sp.dot(C2DOWN,sp.dot(C2DAGUP,C2UP)));

JEINS12UP = sqrtgamma1 * sp.dot(C1DAGDOWN,sp.dot(C2UP,sp.dot(P1UP,P2UP)));
JEINS12DOWN = sqrtgamma1 * sp.dot(C1DAGUP,sp.dot(C2DOWN,sp.dot(P1DOWN,P2DOWN)));
JEINS21UP = sqrtgamma1 * sp.dot(C2DAGDOWN,sp.dot(C1UP,sp.dot(P2UP,P1UP)));
JEINS21DOWN = sqrtgamma1 * sp.dot(C2DAGUP,sp.dot(C1DOWN,sp.dot(P2DOWN,P1DOWN)));
JZWEI12UP = sqrtgamma1 * sp.conj(JEINS12UP).T;
JZWEI12DOWN = sqrtgamma1 * sp.conj(JEINS12DOWN).T;        
JZWEI21UP = sqrtgamma1 * sp.conj(JEINS21UP).T;
JZWEI21DOWN = sqrtgamma1 * sp.conj(JEINS21DOWN).T;
JDREI12 = sqrtgamma2 * sp.dot(C1DAGDOWN,sp.dot(C2DOWN,N2UP));
JDREI21 = sqrtgamma2 * sp.dot(C2DAGDOWN,sp.dot(C1DOWN,N1UP));

J1UP = JEINS12UP + JEINS21UP
J1DOWN = JEINS12DOWN + JEINS21DOWN
J2UP = JZWEI12UP + JZWEI21UP
J2DOWN = JZWEI12DOWN + JZWEI21DOWN
J3 = JDREI12 + JDREI21

global lindbladians
lindbladians = [J1UP,J1DOWN,J2UP,J2DOWN,J3]

global ham
hsp = sp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
h_proto = sp.kron(hsp, hsp).reshape(4, 4, 4, 4);
ham = [h_proto] * N

assert len(ham) == N

for n in xrange(1,N):
    
    # Add actual Hamiltonian
    
    ham_update = t*(C1DAGUP.dot(C2UP) + C2DAGUP.dot(C1UP) + (C1DAGDOWN.dot(C2DOWN)) + (C2DAGDOWN.dot(C1DOWN))) + U * (C1DAGUP.dot(C1UP.dot(C1DAGDOWN.dot(C1DOWN))))
    
    # Terms are next-neighbour. Thus we have N-1 terms for N sites
    # and need to add the last term "twice".
    if(n == (N-1)):
        ham_update+= U * (C2DAGUP.dot(C2UP.dot(C2DAGDOWN.dot(C2DOWN))))
                
    ham[n] = ham_update.reshape(4,4,4,4)
    
#print "Setup complete. We're good to go!"



def sample_path(tdvp_class,num):
    """
    Performs a complete dissipative evolution for the defined number
    of steps and probes global operators after each step.
    
    Parameters
    ----------
    tdvp_class : tdvp_gen_instance
        class object to work on. Must be derived from
        tdvp_gen_dissipative
    num : int
        (Unique) Job number in the distributed computation framework.
        Also used to seed RNG in case the results shall be deterministically
        reproducible.
    """
    
    # Re-seed the random generator with a number based on the iteration number
    rand.seed((num))
    
    print ">> Starting job #"+str(num)
    #print ">> [" + str(num) + "] seeded with " + str(2222+num)
        
    return_dict = {}
    
    
    t = 0.    
    
    q = sp.empty(N + 1, dtype=sp.int32)
    q.fill(qn)    
    
    """
    Now we are ready to create an instance of the generic evoMPS class extended
    by take_step_dissipative(). In fact, the class to be used comes from the grid,
    so it could in general be a non-generic instance.
    """
    
    s = tdvp_class(N, D, q, ham)
    #s.randomize()
    
    #start = sp.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]) / 4.0; #Totally mixed start
    #start = sp.array([[.9997 + 0.j,0,0,1e-5],[.0,0,.01,0],[0, 0, 1e-5,0],[.0,1e-8,0,0.1]]) #zEROstart
    #start = sp.array([[.01,0,0,1e-5],[.0,0,.01,0],[0, .97 + 0.j, 1e-5,0],[.0,1e-8,0,0.1]]) #UPstart
    start = sp.array([[.01,0,0,1e-5],[0, .97 + 0.j, 1e-5,0],[.0,0,.01,0],[.0,1e-8,0,0.1]]) #DOWNstart
    #start = sp.array([[.01,0,0,1e-5],[0, .0001 + 0.j, 1e-5,0],[.0,0,.01,0],[.0,1e-8,0,0.999]]) #AFstart    
    #start = sp.array([[0.0001, 0.00948, 0.0005, 0.0001],[0.0005, 0.0474, 0.0025, 0.0005], [0.0001, 0.00948, 
  #0.0005, 0.0001],[0.00948, 0.898704, 0.0474, 
  #0.00948]]) # Half Doubly occupied starting state
    
    start = start / sp.trace(start)
    for i in xrange(1,N):
        s.apply_op_1s(start,i)
        #s.apply_op_1s(start_1,i)
        #s.apply_op_1s(start_2,i+1)
    
        
    #s.A[N] = sp.array([[1 + 0.j, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]);

    #print s.A[N-1]
    
    #quit()
    
    for i in xrange(total_steps):
        
        # Define necessary operators...

        #print "Starting step", i

        p1 = sp.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
        p2 = sp.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
        p3 = sp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]);
        p4 = sp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]);
        
        EINS1 = sp.eye(2);
        EINS2 = sp.eye(4);
        EINS3 = sp.eye(8);
        EINS4 = sp.eye(16);
        
        PLUS = [[0, 0],[1, 0]];
        MINUS = [[0, 1],[0, 0]];
        PLUSMINUS = [[0, 0],[0, 1]];
        MINUSPLUS = [[1, 0],[0, 0]];
        
        Z = [[1, 0],[0, -1]];
        X = [[0, 1],[1, 0]];
        Y = [[0, -1.j],[1.j, 0]];
        
        C1UP = sp.kron(MINUS, EINS3);
        C1DAGUP = sp.kron(PLUS, EINS3);
        C1DOWN = sp.kron(Z, sp.kron(MINUS, EINS2));
        C1DAGDOWN = sp.kron(Z, sp.kron(PLUS, EINS2));
        C2UP = sp.kron(Z, sp.kron(Z, sp.kron(MINUS, EINS1)));
        C2DAGUP = sp.kron(Z, sp.kron(Z, sp.kron(PLUS, EINS1)));
        C2DOWN = sp.kron(Z, sp.kron(Z, sp.kron(Z, MINUS)));
        C2DAGDOWN = sp.kron(Z, sp.kron(Z, sp.kron(Z, PLUS)));
        
        N1UP = sp.dot(C1DAGUP,C1UP);
        N1DOWN = sp.dot(C1DAGDOWN,C1DOWN);
        N2UP = sp.dot(C2DAGUP,C2UP);
        N2DOWN = sp.dot(C2DAGDOWN,C2DOWN);
        
        P1UP = sp.dot(C1DAGUP,C1UP) - sp.dot(C1DAGUP,sp.dot(C1UP,sp.dot(C1DAGDOWN,C1DOWN)));
        P1DOWN = sp.dot(C1DAGDOWN,C1DOWN) - sp.dot(C1DAGDOWN,sp.dot(C1DOWN,sp.dot(C1DAGUP,C1UP)));
        P2UP = sp.dot(C2DAGUP,C2UP) - sp.dot(C2DAGUP,sp.dot(C2UP,sp.dot(C2DAGDOWN,C2DOWN)));
        P2DOWN = sp.dot(C2DAGDOWN,C2DOWN) - sp.dot(C2DAGDOWN,sp.dot(C2DOWN,sp.dot(C2DAGUP,C2UP)));
        
        JEINS12UP = sqrtgamma1 * sp.dot(C1DAGDOWN,sp.dot(C2UP,sp.dot(P1UP,P2UP)));
        JEINS12DOWN = sqrtgamma1 * sp.dot(C1DAGUP,sp.dot(C2DOWN,sp.dot(P1DOWN,P2DOWN)));
        JEINS21UP = sqrtgamma1 * sp.dot(C2DAGDOWN,sp.dot(C1UP,sp.dot(P2UP,P1UP)));
        JEINS21DOWN = sqrtgamma1 * sp.dot(C2DAGUP,sp.dot(C1DOWN,sp.dot(P2DOWN,P1DOWN)));
        JZWEI12UP = sqrtgamma1 * sp.conj(JEINS12UP).T;
        JZWEI12DOWN = sqrtgamma1 * sp.conj(JEINS12DOWN).T;        
        JZWEI21UP = sqrtgamma1 * sp.conj(JEINS21UP).T;
        JZWEI21DOWN = sqrtgamma1 * sp.conj(JEINS21DOWN).T;
        JDREI12 = sqrtgamma2 * sp.dot(C1DAGDOWN,sp.dot(C2DOWN,N2UP));
        JDREI21 = sqrtgamma2 * sp.dot(C2DAGDOWN,sp.dot(C1DOWN,N1UP));
            
        H = C1DAGUP.dot(C2UP) + C2DAGUP.dot(C1UP) + (C1DAGDOWN.dot(C2DOWN)) + (C2DAGDOWN.dot(C1DOWN)) + U * ((C1DAGUP.dot(C1UP.dot(C1DAGDOWN.dot(C1DOWN))))+(C2DAGUP.dot(C2UP.dot(C2DAGDOWN.dot(C2DOWN)))))
        
        LHL = (JEINS12UP.conj().T.dot(JEINS12UP) +
              JEINS12DOWN.conj().T.dot(JEINS12DOWN) +
              JEINS21UP.conj().T.dot(JEINS21UP) +
              JEINS21DOWN.conj().T.dot(JEINS21DOWN) +
              JZWEI12UP.conj().T.dot(JZWEI12UP) + 
              JZWEI12DOWN.conj().T.dot(JZWEI12DOWN) + 
              JZWEI21UP.conj().T.dot(JZWEI21UP) + 
              JZWEI21DOWN.conj().T.dot(JZWEI21DOWN) +
              JDREI12.conj().T.dot(JDREI12) + 
              JDREI21.conj().T.dot(JDREI21)
              )
        Q = H - 0.5 * LHL
        L1_up = JEINS12UP + JEINS21UP
        L1_down = JEINS12DOWN + JEINS21DOWN
        L2_up = JZWEI12UP + JZWEI21UP
        L2_down = JZWEI12DOWN + JZWEI21DOWN
        L3 = JDREI12 + JDREI21  
     
        # Integrate exactly.

        #s.update(auto_truncate=True, restore_CF=True)
        #s.take_step_RK4(step)
        
        # Or.... take a dissipative approach.
        s.take_step_dissipative(step, l_nns)
        
        # Real time evolution

        timestep_meas = sp.zeros((4), dtype=sp.complex128);
        NsSum = sp.zeros((3), dtype=sp.complex128);
        DoubleSum = 0
        mafSum = 0
        mafAbsSum = 0
        AFBSum = 0
        entSum = 0
        afbop = sp.kron(p2+p3,p2+p3).reshape(4,4,4,4)
        afbop = afbop/sp.trace(afbop)
        for site in xrange(1,N+1):
            timestep_meas[0]+= np.real(s.expect_1s(p1,site)) / N;
            timestep_meas[1]+= np.real(s.expect_1s(p2,site)) / N;
            timestep_meas[2]+= np.real(s.expect_1s(p3,site)) / N;
            timestep_meas[3]+= np.real(s.expect_1s(p4,site)) / N;
            DoubleSum+= 2 * np.real(s.expect_1s(sp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]),site)) / (N);
            rho1 = s.density_1s(site)
            entSum += - sp.trace(rho1.dot(np.log(rho1))) / N
            if site < N:
                AFBSum+= np.real(s.expect_2s(afbop,site)) / (2*(N-1))
                mafSum+=(np.real(s.expect_2s((N1UP + N2UP).reshape(4,4,4,4),site)) - np.real(s.expect_2s((N1DOWN + N2DOWN).reshape(4,4,4,4),site))) / (2*(N-1)); # Fraction of sites with antiferromagnetic neighbors
                mafAbsSum= np.abs((np.real(s.expect_2s((N1UP+N2UP).reshape(4,4,4,4),site)) + np.real(s.expect_2s((N1DOWN + N2DOWN).reshape(4,4,4,4),site)))) / (2*(N-1));
                nupval = np.real(s.expect_2s((N1UP + N2UP).reshape(4,4,4,4),site)) / (2*(N-1));
                ndownval = np.real(s.expect_2s((N1DOWN + N2DOWN).reshape(4,4,4,4),site)) / (2*(N-1));
                nsumval = np.real(s.expect_2s((N1UP + N2UP + N1DOWN + N2DOWN).reshape(4,4,4,4),site)) / (2*(N-1));                
                NsSum[0]+= nsumval
                NsSum[1]+= nupval
                NsSum[2]+= ndownval
                
        MP[i] = timestep_meas
        DO[i] = DoubleSum
        Ns[i] = NsSum
        maf[i] = mafSum
        mafabs[i] = mafAbsSum
        AFB[i] = AFBSum
        ENT[i] = entSum
        
        #print "Step", i, "completed."        
        
        t += step
        
        if(i == 5* total_steps / 10):
            print "job ",num,": 50% Complete."
    
    return_dict["MP"] = MP
    return_dict["Ns"] = Ns
    return_dict["DO"] = DO
    return_dict["maf"] = maf
    return_dict["mafabs"] = mafabs
    return_dict["AFB"] = AFB
    return_dict["ENT"] = ENT
    
    print "<< Job #"+str(num)+" done."
    return return_dict

class evoMPS_TDVP_Generic_Dissipative(TDVP.EvoMPS_TDVP_Generic):
    """ Class derived from TDVP.EvoMPS_TDVP_Generic.
    Extends it by adding dissipative Monte-Carlo evolution for one-side or
    two-site-lindblad dissipations.
    
    Methods:
    ----------
    take_step_dissipative(dt, l_nns)
        Performs dissipative and unitary evolution according to global
        hamiltonian definition and list of lindblads for single-site lindblads.
    take_step_dissipative_nonlocal(dt, MC, l_nns)
        Performs dissipative and unitary evolution according to global
        hamiltonian definition and list of lindblads for multi-site lindblads.
        WARNING: Implementation incomplete.
    apply_op_1s_diss(op,n)
        Applys a single-site operator to site n.    
    """
    
    def take_step_dissipative(self, dt, l_nns):
        """Performs a complete forward-Euler step of imaginary time dtau.
        
        The unitary operation is A[n] -= dt * B[n] with B[n] from self.calc_B(n).
        
        If dt is itself imaginary, real-time evolution results.
        
        Parameters
        ----------
        dt : complex
            The (imaginary or real) amount of imaginary time (tau) to step.
        """
        
        #print "%%%%% Update started %%%%%"
        
        #K = self.K
        #C = self.C
        self.update()
        #print K - self.K
        #print C - self.C
        
        #print "%%%%%% Update complete %%%%%"
    
        # Calculate Hamiltonian part:
        # Pending...

        #print "HHH | Starting Hamiltonian part..."    
        B_H = [None] #There is no site zero
        for n in xrange(1, self.N + 1):
            B_H.append(self.calc_B(n))         
        #print "==="
        #print B_H
        #print "==="
        #print "HHH | Hamiltonian part finished..."
        
        # Calculate Lindblad part:    
    
        for m in xrange(len(lindbladians)):
            LK = sp.empty((self.N + 1), dtype=sp.ndarray) #Elements 1..N
            LC = sp.empty((self.N), dtype=sp.ndarray) #Elements 1..N-1
            L_expect = sp.zeros((len(lindbladians),self.N + 1), dtype=sp.complex128)
            B_L = sp.empty((len(lindbladians), self.N + 1), dtype=sp.ndarray)
            if not lindbladians[m] is None:
                
                #print "Calling calc_B_2s_diss now..."
                
                # So, das volle Programm. Analog zum Hamiltonian müssen wir
                # sowohl C als auch K erstmal für jeden operator ausrechnen
                
                #print "%%%%%% Starting Calc_C_diss..."
                LC = self.calc_C_diss(lindbladians[m])
                #print "%%%%%% Calc_C_diss finished..."
                #print "%%%%%% Starting Calc_K_diss..."
                LK = self.calc_K_diss(lindbladians[m],LC)
                #print "%%%%%% Calc_K_diss finished..."                
                for u in xrange(1,N):
                    #print "m =",m,"u=",u
                    # Hier liegt der Hase im Pfeffer...
                    ba = self.calc_B_diss(lindbladians[m],LC[u],LK[u],u)
                    #print ba
                    #ba = self.expect_2s_diss(lindbladians[m],LC[u],LK[u],u)
                    if ba is not None:
                        if la.norm(ba) > 1E-10:
                        
                            W = sp.random.normal(0, sp.sqrt(dt), np.shape(self.A[u].ravel())) + 0.j
                            W += 1.j * sp.random.normal(0, sp.sqrt(dt), np.shape(self.A[u].ravel()))
                            #print W
                            #print "W.shape:", W.shape
                            W_ = 1/la.norm(ba) * ba.conj().ravel().dot(W)
                            # QSD: Need to add expectation of L_n             
                            if(u > 1 and u <= N+1):
                                L_expect[m,u] += self.expect_2s(lindbladians[m].reshape(4,4,4,4),u-1)
                            if(B_L[m,u] is not None):
                                B_L[m,u]+= ba * (W_)
                            else:
                                B_L[m,u] = ba * (W_)
        
        for n in xrange(1, self.N):
            if not B_H[n] is None:
                self.A[n] += -step * B_H[n]
            for m in xrange(len(lindbladians)):
                if not B_L[m,n] is None:
                    self.A[n] += B_L[m,n] * (1 + (L_expect[m,n] * step))
        
        return True

    def calc_K_diss(self, LB, LC, n_low=-1, n_high=-1):
        """Generates the K matrices used to calculate the B's.
        
        This is called automatically by self.update().
        
        K[n] is contains the action of the Hamiltonian on sites n to N.
        
        K[n] is recursively defined. It depends on C[m] and A[m] for all m >= n.
        
        It directly depends on A[n], A[n + 1], r[n], r[n + 1], C[n] and K[n + 1].
        
        This is equivalent to K on p. 14 of arXiv:1103.0936v2 [cond-mat.str-el], except 
        that it is for the non-norm-preserving case.
        
        K[1] is, assuming a normalized state, the expectation value H of Ĥ.
        """
        #print "Calc_K_diss started..."
        if LB is None:
            return 0        
        
        if n_low < 1:
            n_low = 1
        if n_high < 1:
            n_high = self.N
            
        # Initialize LK with K and then update it from there.
        LK = self.K
        for n in reversed(xrange(n_low, n_high + 1)):
            if n <= self.N - self.ham_sites + 1:
                if self.ham_sites == 2:
                    """                    
                    print "Conjecture: Error appears here:"
                    print "n is", n
                    print "Printing  shapes of A[n], A[n+1]."
                    print self.A[n].shape
                    print self.A[n + 1].shape
                    #AA = tm.calc_AA(self.A[n], self.A[n + 1])
                    #print AA.shape
                    print "Should be (2,1) or (2,2)..."
                    print "LK[n+1]", LK[n+1]
                    print LK[n+1].shape
                    """
                    LK[n], ex = self.calc_K_common(LK[n + 1], LC[n], self.l[n - 1], 
                                              self.r[n + 1], self.A[n], self.A[n + 1])
                else:
                    assert False, "3-Site interaction detected. Not implemented for the dissipative case!"
                    LK[n], ex = tm.calc_K_3s(self.K[n + 1], LC[n], self.l[n - 1], 
                                              self.r[n + 2], self.A[n], self.AAA[n])

                self.h_expect[n] = ex
            else:
                self.K[n].fill(0)
                
        if n_low == 1:
            self.H_expect = sp.asscalar(LK[1])
        return LK

    def calc_K_common(self, Kp1, C, lm1, rp1, A, Ap1):
        Dm1 = A.shape[1]
        q = A.shape[0]
        qp1 = Ap1.shape[0]
        
        K = sp.zeros((Dm1, Dm1), dtype=A.dtype)
        
        Hr = sp.zeros_like(K)
    
        for s in xrange(q):
            Ash = A[s].conj().T
            for t in xrange(qp1):
                test = Ap1[t]
                Hr += C[t, s].dot(rp1.dot(mm.H(test).dot(Ash)))
            K += A[s].dot(Kp1.dot(Ash))
            
        op_expect = mm.adot(lm1, Hr)
            
        K += Hr
        return K, op_expect

    def calc_C_diss(self, L, n_low=-1, n_high=-1):
        """Generates the C tensors used to calculate the K's and ultimately the B's.
        
        This is called automatically by self.update().
        
        C[n] contains a contraction of the Hamiltonian self.ham with the parameter
        tensors over the local basis indices.
        
        This is prerequisite for calculating the tangent vector parameters B,
        which optimally approximate the exact time evolution.
        
        These are to be used on one side of the super-operator when applying the
        nearest-neighbour Hamiltonian, similarly to C in eqn. (44) of 
        arXiv:1103.0936v2 [cond-mat.str-el], for the non-norm-preserving case.

        Makes use only of the nearest-neighbour Hamiltonian, and of the A's.
        
        C[n] depends on A[n] through A[n + self.ham_sites - 1].
        
        """

        LC = sp.zeros_like(self.C)

        if L is None:
            return 0
        
        if n_low < 1:
            n_low = 1
        if n_high < 1:
            n_high = self.N - self.ham_sites + 1
        
        for n in xrange(n_low, n_high + 1):
            if callable(L):
                ham_n = lambda *args: L(n, *args)
                ham_n = sp.vectorize(ham_n, otypes=[sp.complex128])
                ham_n = sp.fromfunction(ham_n, tuple(self.C[n].shape[:-2] * 2))
            else:
                #print "L shape", L.shape
                ham_n = L.reshape(4,4,4,4)
                
            if self.ham_sites == 2:
                AA = tm.calc_AA(self.A[n], self.A[n + 1])                
                LC[n] = tm.calc_C_mat_op_AA(ham_n, AA)
            else:
                AAA = tm.calc_AAA(self.A[n], self.A[n + 1], self.A[n + 2])
                LC[n] = tm.calc_C_3s_mat_op_AAA(ham_n, AAA)
        return LC
        
    def calc_B_diss(self,op,K,C,n,set_eta=True):
        
        """Generates the TDVP tangent vector parameters for a single site B[n].
        
        A TDVP time step is defined as: A[n] -= dtau * B[n]
        where dtau is an infinitesimal imaginary time step.
        
        In other words, this returns B[n][x*] (equiv. eqn. (47) of 
        arXiv:1103.0936v2 [cond-mat.str-el]) 
        with x* the parameter matrices satisfying the Euler-Lagrange equations
        as closely as possible.
        
        Returns
        -------
            B_n : ndarray or None
                The TDVP tangent vector parameters for site n or None
                if none is defined.
        """
        
        #print "DEBUG: THIS ONE IS CALLED."        
        
        if self.gauge_fixing == "right":
            return self._calc_B_r_diss(op, K, C, n, set_eta=set_eta)
        else:
            return self._calc_B_l_diss(op, K, C, n, set_eta=set_eta)
    
    def _calc_B_r_diss(self, op, K, C, n, set_eta=True):
        if self.q[n] * self.D[n] - self.D[n - 1] > 0:
            l_sqrt, l_sqrt_inv, r_sqrt, r_sqrt_inv = tm.calc_l_r_roots(self.l[n - 1], 
                                                                   self.r[n],
                                                                   sanity_checks=self.sanity_checks,
                                                                   sc_data=("site", n))
            Vsh = tm.calc_Vsh(self.A[n], r_sqrt, sanity_checks=self.sanity_checks)
            x = self.calc_x(n, Vsh, l_sqrt, r_sqrt, l_sqrt_inv, r_sqrt_inv)
            if set_eta:
                self.eta[n] = sp.sqrt(mm.adot(x, x))
    
            B = sp.empty_like(self.A[n])
            for s in xrange(self.q[n]):
                B[s] = mm.mmul(l_sqrt_inv, x, mm.H(Vsh[s]), r_sqrt_inv)
            return B
        else:
            return None

    def _calc_B_l_diss(self, op, K, C, n, set_eta=True):
        if self.q[n] * self.D[n - 1] - self.D[n] > 0:
            l_sqrt, l_sqrt_inv, r_sqrt, r_sqrt_inv = tm.calc_l_r_roots(self.l[n - 1], 
                                                                   self.r[n], 
                                                                   zero_tol=self.zero_tol,
                                                                   sanity_checks=self.sanity_checks,
                                                                   sc_data=("site", n))
            
            Vsh = tm.calc_Vsh_l(self.A[n], l_sqrt, sanity_checks=self.sanity_checks)
            
            x = self.calc_x_l(n, Vsh, l_sqrt, r_sqrt, l_sqrt_inv, r_sqrt_inv)
            
            if set_eta:
                self.eta[n] = sp.sqrt(mm.adot(x, x))
    
            B = sp.empty_like(self.A[n])
            for s in xrange(self.q[n]):
                B[s] = mm.mmul(l_sqrt_inv, mm.H(Vsh[s]), x, r_sqrt_inv)
            return B
        else:
            return None


    def expect_1s_diss(self,op,n):
        """Applies a single-site operator to a single site and returns
        the value after the change. In contrast to
        mps_gen.apply_op_1s, this routine does not change the state itself.
        
        Also, this does not perform self.update().
        
        Parameters
        ----------
        op : ndarray or callable
            The single-site operator. See self.expect_1s().
        n: int
            The site to apply the operator to.
        """
        if callable(op):
            op = sp.vectorize(op, otypes=[sp.complex128])
            op = sp.fromfunction(op, (self.q[n], self.q[n]))
            
        newAn = sp.zeros_like(self.A[n])
        
        for s in xrange(self.q[n]):
            for t in xrange(self.q[n]):
                newAn[s] += self.A[n][t] * op[s, t]
                
        return newAn
        
    def expect_2s_diss(self, op, LC, LK, n, AA=None):
        """Applies a two-site operator to two sites and returns
        the value after the change. In contrast to
        mps_gen.apply_op_2s, this routine does not change the state itself.
        
        Also, this does not perform self.update().
        
        Parameters
        ----------
        op : ndarray or callable
            The two-site operator. See self.expect_2s().
        n: int
            The site to apply the operator to.
            (It's also applied to n-1.)
        """
        #No neighbors, no fun.
        
        if n is 1:
            return 0
        if n is N:
            return 0
            
        A = self.A[n-1]
        Ap1 = self.A[n]
        if AA is None:
            AA = tm.calc_AA(A, Ap1)
            
        if callable(op):
            op = sp.vectorize(op, otypes=[sp.complex128])
            op = sp.fromfunction(op, (A.shape[0], Ap1.shape[0], A.shape[0], Ap1.shape[0]))
        
        op = op.reshape(4,4,4,4)
        C = tm.calc_C_mat_op_AA(op, AA)
        res = tm.eps_r_op_2s_C12_AA34(self.r[n + 1], LC, AA)
        operand = self.l[n-1]
        operand = sp.reshape(operand, (1,16))
        operand = sp.reshape(operand, (2,8))
                
        return mm.mmul(operand,res)
        return mm.adot(self.l[n - 1], res)



"""    
When not called from the grid, perform singular operation based on standard
variables.
"""

try:
    internal_call
except NameError:
    internal_call = False

if(internal_call is False):
    print "The dissipative module was not called from the computational grid."
    print "Assuming singular operation..."
    """
    bd = np.ndarray((N+1), dtype=sp.int32)
    bd[:] = bond_dim
    q = np.ndarray((N+1), dtype=sp.int32)
    q[:] = qn
    """
    #for i in xrange(1,20):
    sample_path(evoMPS_TDVP_Generic_Dissipative,1)
    #sample_path(EvoMPS_TDVP_Generic,1)
    #print Mx
    print "Single instance successfully calculated."
    
    plot_results = False
