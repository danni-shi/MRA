import datetime as dt
import autograd.numpy as np
#import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt, dates
import os
import time

import utils
import optimization
import opt_new

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

with open('../../data/logreturn.npy', 'rb') as f:
    signal = np.load(f)
signal = signal[:60]
# x = np.linspace(0,2*np.pi, 100)
# x = np.sin(x)
# signal = (x-np.mean(x))/np.std(x)

# signal = np.random.randn(50)

# set parameters for generating observations
num_copies = 500
sigma = 0.5
max_shift= 0.3

# generate shifted, noisy version of the signal
start = time.time()
observations, shifts = utils.generate_data(signal, num_copies,  max_shift, sigma, cyclic = False)
print('time to generate data = ', time.time() - start)

# plot selected observations 
N, M = observations.shape
fig, axes = plt.subplots(10,5, figsize=(20,20));
ax = axes.flatten()
for i in range(50):
    lag = shifts[i]
    ax[i].vlines(lag, np.min(observations[:,i]), np.max(observations[:,i]), color = 'red', ls = '-.')
    ax[i].plot(observations[:,i])
plt.savefig('../plots/observations')

# optimization
L = len(signal)
np.random.seed(42)
X0 = np.random.normal(0, 1, L)
X_est = optimization.optimise_manopt(observations, sigma, X0, extra_inits=0)

# align the estimate to original signal    
X_aligned = utils.align_to_ref(X_est.flatten(),signal)
print('relative error = ', np.linalg.norm(X_aligned-signal)/np.linalg.norm(signal))

# save data
with open('../results/data.npy', 'wb') as f:
    np.save(f, observations)
    np.save(f, shifts)
    
with open('../results/visual.npy', 'wb') as f:
    np.save(f, X_est)
    np.save(f, signal)
    np.save(f,X0)