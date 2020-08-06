"""
Illustration of Oja's rule (Hebbian learning) in Spiking Neural Networks using LIF neuron.
Author: Ali Safa
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import matplotlib.animation as animation

"""
Functions
"""
def sample_spherical(npoints, ndim): #sample a vector of dimension "ndim" from the unit sphere randomly
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec[:,0]

def simulate_neuron_Oja(Tsim, dt, trc, tref, vrest, vth, Jbias, alpha, e, input_vec, tau, eta):
    N = int(np.round(Tsim/dt))
    Vprev = 0
    Jprev = 0
    spike_train = np.zeros(N)
    Vhist = np.zeros(N)
    psc = np.zeros(N)
    W_vec = np.zeros((N, len(e))) 
    W = alpha*e
    W_vec[0,:] = W
    Jbias_vec = np.zeros(N)
    Jbias_vec[0] = Jbias
    mutex = 0 
    for i in range(N):
        J = np.inner(W, input_vec[i,:]) + Jbias
        if mutex == 0:
            V = (J + Jprev - (1-2*trc/dt)*Vprev)/(1+2*trc/dt) #bilinear transform
            if V < vrest:
                V = vrest
            elif V > vth:
                spike_train[i] = 1
                V = vrest
                mutex = np.round(tref/dt)
            Vhist[i] = V 
            Jprev = J
            Vprev = V
        else:
            mutex -= 1
        
        if i > 0:
            psc[i] = (spike_train[i] + spike_train[i-1])*(1/(1+2*tau/dt)) - ((1-2*tau/dt)/(1+2*tau/dt))*psc[i-1]
            #update weights following Oja's rule
            DELW = psc[i]*input_vec[i,:] - (psc[i]**2)*W
            W = W + eta*DELW                
            W_vec[i,:] = W
            Jbias_vec[i] = Jbias
            
    return Vhist, spike_train, W_vec, psc, Jbias_vec

def PSC_filter(Tsim, dt, tau):
    t = np.linspace(0,Tsim,int(np.round(Tsim/dt)))
    h = np.exp(-(t-Tsim/2)/tau)
    h[0:len(h)//2] = 0
    h = (1/dt)*h/np.sum(h)
    return h

def normalize_imges(data): #normalize pixel values to [-1, 1] 
    for i in range(data.shape[0]):
        img = data[i]
        data[i] = 2*(img - min(img))/(max(img) - min(img)) - 1
    return data

"""
Main
"""
np.random.seed(2) #to get reproducable results
plt.close('all')


D = 64 #data dimensions
F_max_l = 100 #100
F_max_h = 200
in_l = -1.0
in_h = 1.0
tref = 0.002 #2ms
trc = 0.02 #20ms
Tsim = 0.2
dt = 0.002
vrest = 0
vth = 1
Tlen = int(np.round(Tsim/dt))

digits = load_digits(n_class=1)
x = normalize_imges(digits.data)
input_vec = x[:Tlen,:]

amax = np.random.uniform(F_max_l,F_max_h,1) # maximum rate uniformly distributed between 100 and 200 HZ
#xi = np.random.uniform(in_l+0.05,in_h-0.05,1) # x-intercept
xi = np.random.uniform(in_l+0.05,in_h-0.05,1) # new idea x-intercept
alpha = (1/(1-np.exp((tref - 1/amax)/trc)) - 1)/(1-xi) #for LIF neuron
Jbias = 1-xi*alpha
e = sample_spherical(1, D)

Vhist, spike_train, W_vec, psc, Jbias_vec = simulate_neuron_Oja(Tsim, dt, trc, tref, vrest, vth, Jbias, alpha, e, input_vec, 0.05, 1)

plt.figure(1)
plt.gray() 
plt.axis('off')
im = np.reshape(W_vec[-1,:], (8,8))
plt.imshow(im) 


#if True:
plt.figure(2)
fig, ax = plt.subplots(1,1)
plt.gray() 
plt.axis('off')
plt.title('On-line and Unsuppervised Pattern Learning', fontsize=15, color='black')

def img_anim(i):
    im = np.reshape(W_vec[i,:], (8,8))
    ax.matshow(im) 
    print("Time step: " + str(i) + "/" + str(Tlen))
        
anim2 = animation.FuncAnimation(fig, img_anim, frames=Tlen, interval=1)
#plt.show() 
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
#anim2.save('Hebbian.mp4', writer=writer)
