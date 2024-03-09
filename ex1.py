#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:47:14 2024

@author: michaelroop
"""

import numpy as np
import matplotlib.pyplot as plt



def ForwardEuler(h,y0,f):
    y1 = y0+h*f(y0)
    return y1
def ExplicitRK2(h,y0,f):
    y1 = y0+h/2*f(y0)+h/2*f(y0+h*f(y0))
    return y1
def ExplicitRK4(h,y0,f):
    k1 = f(y0)
    k2 = f(y0+h/2*k1)
    k3 = f(y0+h/2*k2)
    k4 = f(y0+h*k3)
    b1 = 1/6
    b2 = 1/3
    b3 = 1/3
    b4 = 1/6
    y1 = y0+h*(b1*k1+b2*k2+b3*k3+b4*k4)
    return y1
def SymplEuler(h,p0,q0):
    p1 = p0-h*np.sin(q0)
    q1 = q0+h*p1
    y1 = np.array([q1,p1])
    return y1
def StromVerlet(h,p0,q0):
    p_half = p0-h/2*np.sin(q0)
    q1 = q0+h/2*(p_half+p_half)
    p1 = p_half-h/2*np.sin(q1)
    y1 = np.array([q1,p1])
    return y1
def f(x):
    rhsP = np.sin(x[0])
    rhsQ = -x[1]
    rhs = np.array([rhsQ,rhsP])
    return rhs
#%% Forward Euler method
h = 0.01 
N = 5000
p = np.zeros(N)
q = np.zeros(N)
t = np.zeros(N)
H = np.zeros(N)
p[0] = 1
q[0] = 0
H0 = p[0]**2/2-np.cos(q[0])
H[0] = H0
y0 = np.array([q[0],p[0]])
for i in range(1,N):
    y1 = ForwardEuler(h,y0,f)
    t[i] = i*h
    q[i] = y1[0]
    p[i] = y1[1]
    H[i] = p[i]**2/2-np.cos(q[i])
    y0 = np.array([y1[0],y1[1]])
plt.figure()
plt.scatter(q,p,marker = '*',color='b')
qrange = np.linspace(-2,2,10000)
prange_plus = np.sqrt(2*(H0+np.cos(qrange)))
prange_minus = -np.sqrt(2*(H0+np.cos(qrange)))
plt.plot(qrange,prange_plus,color='r')
plt.plot(qrange,prange_minus,color='r')
plt.title('Pendulum phase portrait, forward Euler, h ={}, T={}'.format(h,N*h))
plt.xlabel('q')
plt.ylabel('p',rotation=0)
plt.legend(['Numerical flow','Exact flow'],loc='lower left')
plt.savefig('pend_phase_Eu1_middle.eps')
plt.figure()
plt.plot(t,H,'r')
plt.xlabel('t')
plt.ylabel('H',rotation=0)
plt.title('Energy, forward Euler')
plt.savefig('pend_en_Eu1_middle.eps')
#%%Explicit RK2 method

h = 0.01 
N = 500000
p = np.zeros(N)
q = np.zeros(N)
t = np.zeros(N)
H = np.zeros(N)
p[0] = 1
q[0] = 0
H0 = p[0]**2/2-np.cos(q[0])
H[0] = H0
y0 = np.array([q[0],p[0]])
for i in range(1,N):
    y1 = ExplicitRK2(h,y0,f)
    t[i] = i*h
    q[i] = y1[0]
    p[i] = y1[1]
    H[i] = p[i]**2/2-np.cos(q[i])
    y0 = np.array([y1[0],y1[1]])
plt.figure()
plt.scatter(q,p,marker = '*',color='b')
qrange = np.linspace(-2,2,10000)
prange_plus = np.sqrt(2*(H0+np.cos(qrange)))
prange_minus = -np.sqrt(2*(H0+np.cos(qrange)))
plt.plot(qrange,prange_plus,color='r')
plt.plot(qrange,prange_minus,color='r')
plt.title('Pendulum phase portrait, explicit RK2, h ={}, T={}'.format(h,N*h))
plt.xlabel('q')
plt.ylabel('p',rotation=0)
plt.legend(['Numerical flow','Exact flow'],loc='upper right')
plt.savefig('pend_phase_RK2_middle.png')
plt.figure()
plt.plot(t,H,'r')
plt.xlabel('t')
plt.ylabel('H',rotation=0)
plt.title('Energy, explicit RK2')
plt.savefig('pend_en_RK2_middle.png')
#%%Explicit RK4 method

h = 0.1
N = 50000
p = np.zeros(N)
q = np.zeros(N)
t = np.zeros(N)
H = np.zeros(N)
p[0] = 1
q[0] = 0
H0 = p[0]**2/2-np.cos(q[0])
H[0] = H0
y0 = np.array([q[0],p[0]])
for i in range(1,N):
    y1 = ExplicitRK4(h,y0,f)
    t[i] = i*h
    q[i] = y1[0]
    p[i] = y1[1]
    H[i] = p[i]**2/2-np.cos(q[i])
    y0 = np.array([y1[0],y1[1]])
plt.figure()
plt.scatter(q,p,marker = '*',color='b')
qrange = np.linspace(-2,2,10000)
prange_plus = np.sqrt(2*(H0+np.cos(qrange)))
prange_minus = -np.sqrt(2*(H0+np.cos(qrange)))
plt.plot(qrange,prange_plus,color='r')
plt.plot(qrange,prange_minus,color='r')
plt.title('Pendulum phase portrait, explicit RK4, h ={}, T={}'.format(h,N*h))
plt.xlabel('q')
plt.ylabel('p',rotation=0)
plt.legend(['Numerical flow','Exact flow'],loc='upper right')
plt.savefig('pend_phase_RK4_big.png')
plt.figure()
plt.plot(t,H,'r')
plt.xlabel('t')
plt.ylabel('H',rotation=0)
plt.title('Energy, explicit RK4')
plt.savefig('pend_en_RK4_big.png')
#%%Symplectic Euler method

h = 0.1
N = 2000
p = np.zeros(N)
q = np.zeros(N)
t = np.zeros(N)
H = np.zeros(N)
p[0] = 1
q[0] = 0
H0 = p[0]**2/2-np.cos(q[0])
H[0] = H0
y0 = np.array([q[0],p[0]])
for i in range(1,N):
    y1 = SymplEuler(h,p[i-1],q[i-1])
    t[i] = i*h
    q[i] = y1[0]
    p[i] = y1[1]
    H[i] = p[i]**2/2-np.cos(q[i])
plt.figure()
plt.scatter(q,p,marker = '*',color='b')
qrange = np.linspace(-2,2,10000)
prange_plus = np.sqrt(2*(H0+np.cos(qrange)))
prange_minus = -np.sqrt(2*(H0+np.cos(qrange)))
plt.plot(qrange,prange_plus,color='r')
plt.plot(qrange,prange_minus,color='r')
plt.title('Pendulum phase portrait, Symplectic Euler, h ={}, T={}'.format(h,N*h))
plt.xlabel('q')
plt.ylabel('p',rotation=0)
plt.legend(['Numerical flow','Exact flow'],loc='upper right')
plt.savefig('pend_phase_sympl_middle.eps')
plt.figure()
plt.plot(t,H,'r')
plt.xlabel('t')
plt.ylabel('H',rotation=0)
plt.title('Energy, Symplectic Euler')
plt.savefig('pend_en_sympl_middle.eps')
#%%Strömer-Verlet method

h = 0.1
N = 2500
p = np.zeros(N)
q = np.zeros(N)
t = np.zeros(N)
H = np.zeros(N)
p[0] = 1
q[0] = 0
H0 = p[0]**2/2-np.cos(q[0])
H[0] = H0
y0 = np.array([q[0],p[0]])
for i in range(1,N):
    y1 = StromVerlet(h,p[i-1],q[i-1])
    t[i] = i*h
    q[i] = y1[0]
    p[i] = y1[1]
    H[i] = p[i]**2/2-np.cos(q[i])
plt.figure()
plt.scatter(q,p,marker = '*',color='b')
qrange = np.linspace(-2,2,10000)
prange_plus = np.sqrt(2*(H0+np.cos(qrange)))
prange_minus = -np.sqrt(2*(H0+np.cos(qrange)))
plt.plot(qrange,prange_plus,color='r')
plt.plot(qrange,prange_minus,color='r')
plt.title('Pendulum phase portrait, Strömer-Verlet, h ={}, T={}'.format(h,N*h))
plt.xlabel('q')
plt.ylabel('p',rotation=0)
plt.legend(['Numerical flow','Exact flow'],loc='upper right')
plt.savefig('pend_phase_sv_middle.eps')
plt.figure()
plt.plot(t,H,'r')
plt.xlabel('t')
plt.ylabel('H',rotation=0)
plt.title('Energy, Strömer-Verlet')
plt.savefig('pend_en_sv_middle.eps')
