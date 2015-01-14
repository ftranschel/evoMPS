# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 09:30:30 2013

@author: ftranschel
"""

"""

Definition of the output datastructure:
    
total_steps: number of time steps
total_iterations: number of sampling paths
N: lattice length

Kx = sp.zeros((total_iterations,total_steps),dtype=sp.complex128)
Mx = sp.zeros((total_iterations,N+1,total_steps),dtype=sp.complex128)

KMean = sp.zeros((total_steps),dtype=sp.complex128) 
MMean = sp.zeros((total_steps),dtype=sp.complex128)
MsMean = sp.zeros((N,total_steps),dtype=sp.complex128)

KVar = sp.zeros((total_steps),dtype=sp.complex128)
MVar = sp.zeros((total_steps),dtype=sp.complex128)
MsVar = sp.zeros((N,total_steps),dtype=sp.complex128)

"""

#print Kx
#print result_list

# task one: reconstruct Kx and Mx from the result_list:

global Kx
Kx = sp.zeros((total_iterations,total_steps),dtype=sp.complex128)
global Mx
Mx = sp.zeros((total_iterations,N+1,total_steps),dtype=sp.complex128)   
   
print "Rebuilding data structure..."

for k in xrange(total_iterations):
    Kx[k] = result_list[k]["K1"]
    Mx[k] = result_list[k]["M1"]
    
print "Kx:"
print Kx
print "Mx:"
print Mx    
    
print str(total_iterations) + " result units processed."

print "Starting postprocessing..."
KMean = float(1.0/float(total_iterations)) * (sum(Kx[::]))
MMean = float(1.0/float(total_iterations)) * (sum(Mx[::,0]))
for i in xrange(1,N+1):
    MsMean[i-1] = float(1.0/float(total_iterations)) * (sum(Mx[::,i]))
    #print Ms.shape
    #print MsMean.shape
    #print i
    for j in xrange(0,total_iterations):
        MsVar[i-1] = float(1/(float(total_iterations) - 1)) * abs(np.subtract(MsMean[i-1],Mx[j,i])*np.subtract(MsMean[i-1],Mx[j,i]))
        
print "done."
print "Plotting now..."

t = sp.zeros((total_steps),dtype=sp.complex128)
i = 0
while i < total_steps:
  t[i] = i*step
  i+=1
  
fig3 = plt.figure(3)
fig4 = plt.figure(4)
fig5 = plt.figure(5)
fig6 = plt.figure(6)

explanation_string = 't [N=' + str(N) + ', dt=' + str(step) + ', Lda=' + str(Lambda) + ', bdim=' + str(bond_dim) + ', tsteps=' + str(total_steps) + ', h=' + str(h) + ', it='+ str(total_iterations) + ']'
K1 = fig3.add_subplot(111)
K1.set_ylabel('K (sum)')
K1.set_xlabel(explanation_string)
M1 = fig4.add_subplot(111)
M1.set_xlabel(explanation_string)
M1.set_ylabel('M_x (sum)')
M2 = fig5.add_subplot(111)
M2.set_xlabel(explanation_string)
M2.set_ylabel('M_x (single site)')
M3 = fig6.add_subplot(111)
M3.set_xlabel(explanation_string)
M3.set_ylabel('M_x (variance)')

#M3.set_yscale('log')


K1.plot(t,np.real(KMean))
M1.plot(t,MMean.real[::])
M2.plot(t,MsMean.real[0,::])
M3.plot(t,np.real(MsVar[1]))

#M1.plot(t,np.real(MMean))
#M2.plot(t,np.real(MsMean[0]),t,np.real(MsMean[1]),t,np.real(MsMean[2]),t,np.real(MsMean[3]),t,np.real(MsMean[4]))
#M3.plot(t,np.real(MsVar[1]))
    #M1.plot(t, MMean.real[::],t,Mx.real[::,0])
    #M2.plot(t, MsMean.real[0,::],t,Mx[0,1,::])
#
#if total_iterations == 2:
#    K1.plot(t,KMean.real[::])
#    M1.plot(t,np.real(MMean))
#    M2.plot(t,np.real(MsMean[0]),t,np.real(MsMean[1]),t,np.real(MsMean[2]),t,np.real(MsMean[3]),t,np.real(MsMean[4]))
#   #M1.plot(t, MMean.real[::],t,Mx.real[::,0])
#    #M2.plot(t, MsMean.real[0,::],t,Mx[0,1,::])
#    M3.plot(t,np.real(MsVar[1]))
#
#if total_iterations == 3:
#    K1.plot(t,KMean.real[::])
#    #M1.plot(t,Ms[0].real[::],t,Ms[1].real[::],t,Ms[2].real[::])
#    M1.plot(t,MMean.real[::])
#    M2.plot(t,MsMean.real[0,::])
#    M3.plot(t,np.real(MsVar[1]))
#
#if total_iterations == 4:
#    #K1.plot(t,Ks[0].real[::],t,Ks[1].real[::],t,Ks[2].real[::],t,Ks[3].real[::])
#    #M1.plot(t,Ms[0].real[::],t,Ms[1].real[::],t,Ms[2].real[::],t,Ms[3].real[::])
#    K1.plot(t,KMean.real[::])
#    M1.plot(t,MMean.real[::])
#    M2.plot(t,MsMean.real[0,::])
#    M3.plot(t,np.real(MsVar[1]))
#
#if total_iterations == 5:
#    #K1.plot(t,Ks[0].real[::],t,Ks[1].real[::],t,Ks[2].real[::],t,Ks[3].real[::],t,Ks[4].real[::],t,Mean.real[::])
#    #K1.plot(t,Mean.real[::])
#    #M1.plot(t,Ms[0].real[::],t,Ms[1].real[::],t,Ms[2].real[::],t,Ms[3].real[::],t,Ms[4].real[::])
#    K1.plot(t,KMean.real[::])    
#    M1.plot(t,MMean.real[::])
#    M2.plot(t,MsMean.real[0,::])
#    M3.plot(t,np.real(MsVar[1]))
#
#if total_iterations == 10:
#    K1.plot(t,KMean.real[::])
#    #K1.plot(t,Mean.real[::])
#    #M1.plot(t,Ms[0].real[::],t,Ms[1].real[::],t,Ms[2].real[::],t,Ms[3].real[::],t,Ms[4].real[::])
#    M1.plot(t,MMean.real[::])
#    M2.plot(t,MsMean.real[0,::])
#    M3.plot(t,np.real(MsVar[1]))
#
#if total_iterations == 25:
#    K1.plot(t,KMean.real[::])
#    M1.plot(t,np.real(MMean))
#    M2.plot(t,np.real(MsMean[0]),t,np.real(MsMean[1]),t,np.real(MsMean[2]),t,np.real(MsMean[3]),t,np.real(MsMean[4]))
#    M3.plot(t,np.real(MsVar[1]))
#    
#if total_iterations == 50:
#    K1.plot(t,KMean.real[::])
#    M1.plot(t,np.real(MMean))
#    M2.plot(t,np.real(MsMean[0]),t,np.real(MsMean[1]),t,np.real(MsMean[2]),t,np.real(MsMean[3]),t,np.real(MsMean[4]))
#    M3.plot(t,np.real(MsVar[1]))
#   
#if total_iterations == 100:
#    K1.plot(t,KMean.real[::])
#    M1.plot(t,np.real(MMean))
#    M2.plot(t,np.real(MsMean[0]),t,np.real(MsMean[1]),t,np.real(MsMean[2]),t,np.real(MsMean[3]),t,np.real(MsMean[4]))
#    M3.plot(t,np.real(MsVar[1]))
#    
#if total_iterations == 1000:
#    K1.plot(t,KMean.real[::])
#    M1.plot(t,np.real(MMean))
#    M2.plot(t,np.real(MsMean[0]),t,np.real(MsMean[1]),t,np.real(MsMean[2]),t,np.real(MsMean[3]),t,np.real(MsMean[4]))
#    M3.plot(t,np.real(MsVar[1]))

plt.show()
