#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 17:03:51 2024

@author: michaelroop
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from celluloid import Camera

def dist(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

def V(r):
    return 1/r**12-2/r**6
def dV(r):
    return 12*(1/r**7-1/r**13)
def H(qx,qy,px,py):
    N = len(qx)
    H0 = 1/2*np.sum(px**2)+1/2*np.sum(py**2)
    for i in range(N):
        for j in range(i+1,N):
            H0 = H0+V(dist(np.array([qx[i],qy[i]]),np.array([qx[j],qy[j]])))
    return H0
def K(qx,qy,px,py):
    H0 = 1/2*np.sum(px**2)+1/2*np.sum(py**2)
    return H0
def rhsQ(px,py):
    return np.concatenate((px,py))
def rhsP(qx,qy):
    N = len(qx)
    rhsQx = np.zeros(N)
    rhsQy = np.zeros(N)
    for k in range(N):
        for i in range(k):
            d = dist(np.array([qx[i],qy[i]]),np.array([qx[k],qy[k]]))
            rhsQx[k] = rhsQx[k]+dV(d)/d*(qx[i]-qx[k])
        for i in range(k+1,N):
            d = dist(np.array([qx[i],qy[i]]),np.array([qx[k],qy[k]]))
            rhsQx[k] = rhsQx[k]+dV(d)/d*(qx[i]-qx[k])
    for k in range(N):
        for i in range(k):
            d = dist(np.array([qx[i],qy[i]]),np.array([qx[k],qy[k]]))
            rhsQy[k] = rhsQy[k]+dV(d)/d*(qy[i]-qy[k])
        for i in range(k+1,N):
            d = dist(np.array([qx[i],qy[i]]),np.array([qx[k],qy[k]]))
            rhsQy[k] = rhsQy[k]+dV(d)/d*(qy[i]-qy[k])
    return np.concatenate((rhsQx,rhsQy))
def StromVerlet(h,p0x,p0y,q0x,q0y):
    N = len(p0x)
    
    p0 = np.concatenate((p0x,p0y))
    
    RP0 = rhsP(q0x,q0y)
    
    
    p_half = p0+h/2*RP0
    p_half_x = p_half[0:N]
    p_half_y = p_half[N:2*N]
    
    RQ_half = rhsQ(p_half_x,p_half_y)
    
    q1x = q0x+h*RQ_half[0:N]
    q1y = q0y+h*RQ_half[N:2*N]
    
    RP1 = rhsP(q1x,q1y)
    
    p1x = p_half_x+h/2*RP1[0:N]
    p1y = p_half_y+h/2*RP1[N:2*N]
    
    return q1x,q1y,p1x,p1y

def ForwardEuler(h,p0x,p0y,q0x,q0y):
    N = len(p0x)
    
    q1x = q0x+h*rhsQ(p0x,p0y)[0:N]
    q1y = q0y+h*rhsQ(p0x,p0y)[N:2*N]
    
    p1x = p0x+h*rhsP(q0x,q0y)[0:N]
    p1y = p0y+h*rhsP(q0x,q0y)[N:2*N]
    
    return q1x,q1y,p1x,p1y

#%%Strömer-Verlet

start_time = time.time()
N = 10
h = 0.02
itermax = 1000
Xmat = np.empty((0,N), float)
for j in range(N):
    col = j*np.ones(N)
    Xmat = np.vstack([Xmat,col])
q0x = np.reshape(Xmat, N**2)
Xmat = np.empty((0,N), float)
for j in range(N):
    col = np.arange(N)
    Xmat = np.vstack([Xmat,col])
q0y = np.reshape(Xmat, N**2)
p0x = np.zeros(N**2)
p0y = np.zeros(N**2)
t = np.zeros(itermax+1)
Hamilt = np.zeros(itermax+1)
Hamilt[0] = H(q0x,q0y,p0x,p0y)
Temp = np.zeros(itermax+1)
Temp[0] = K(q0x,q0y,p0x,p0y)
x1 = np.zeros((N**2,itermax+1))
x1[:,0] = q0x
x2 = np.zeros((N**2,itermax+1))
x2[:,0] = q0y
for i in range(1,itermax+1):
    q1x,q1y,p1x,p1y = StromVerlet(h,p0x,p0y,q0x,q0y)
    x1[:,i] = q1x
    x2[:,i] = q1y
    Hamilt[i] = H(q1x,q1y,p1x,p1y)
    Temp[i] = K(q0x,q0y,p0x,p0y)
    t[i] = h*i
    p0x = p1x
    p0y = p1y
    q0x = q1x
    q0y = q1y
    
plt.figure()
plt.plot(t,Hamilt,'r')
plt.title('Energy, Strömer-Verlet')
plt.xlabel('t')
plt.ylabel('H', rotation = 0)
plt.savefig('energy_sv_N_{}.png'.format(N))

plt.figure()
plt.plot(t,Temp,'r')
plt.title('Temperature, Strömer-Verlet')
plt.xlabel('t')
plt.ylabel('T', rotation = 0)
plt.savefig('temp_sv_N_{}.png'.format(N))

plt.figure()
plt.scatter(x1[:,0],x2[:,0],color='k', s=100)
plt.title('Configuration of particles, t=0')
plt.savefig('particles_t_0')

plt.figure()
plt.scatter(x1[:,400],x2[:,400],color='k', s=100)
plt.title('Configuration of particles, t=8')
plt.savefig('particles_t_8')

plt.figure()
plt.scatter(x1[:,1000],x2[:,1000],color='k', s=100)
plt.title('Configuration of particles, t=20')
plt.savefig('particles_t_20')


end_time = time.time()
stime = (end_time-start_time)/60
print(stime)


camera = Camera(plt.figure())
for i in range(itermax+1):
    plt.scatter(x1[:,i],x2[:,i],color='k', s=100)
    camera.snap()
anim = camera.animate(interval=1,blit=True)
anim.save('scatter.mp4')

#%%
N = 6
h = 0.01
itermax = 1000
Xmat = np.empty((0,N), float)
for j in range(N):
    col = j*np.ones(N)
    Xmat = np.vstack([Xmat,col])
q0x = np.reshape(Xmat, N**2)
Xmat = np.empty((0,N), float)
for j in range(N):
    col = np.arange(N)
    Xmat = np.vstack([Xmat,col])
q0y = np.reshape(Xmat, N**2)
p0x = np.zeros(N**2)
p0y = np.zeros(N**2)
t = np.zeros(itermax+1)
Hamilt = np.zeros(itermax+1)
Hamilt[0] = H(q0x,q0y,p0x,p0y)
Temp = np.zeros(itermax+1)
Temp[0] = K(q0x,q0y,p0x,p0y)
for i in range(1,itermax+1):
    q1x,q1y,p1x,p1y = ForwardEuler(h,p0x,p0y,q0x,q0y)
    Hamilt[i] = H(q1x,q1y,p1x,p1y)
    Temp[i] = K(q0x,q0y,p0x,p0y)
    t[i] = h*i
    p0x = p1x
    p0y = p1y
    q0x = q1x
    q0y = q1y
    
plt.figure()
plt.plot(t,Hamilt,'r')
plt.title('Energy, forward Euler')
plt.xlabel('t')
plt.ylabel('H', rotation = 0)
plt.savefig('energy_FEu.png')
plt.figure()
plt.plot(t,Temp,'r')
plt.title('Temperature, forward Euler')
plt.xlabel('t')
plt.ylabel('T', rotation = 0)
plt.savefig('temp_FEu.png')
    
    
    
    
    
    
    
    
    
    
    
    
    