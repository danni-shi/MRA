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

# with open('logreturn.npy', 'rb') as f:
#     signal = pd.Series(np.load(f))

x = np.linspace(0,2*np.pi, 100)
x = np.sin(x)
# x[30:40] = np.sin(np.linspace(0,4*np.pi, 10))
# x[90:100] = np.sin(np.linspace(0,4*np.pi, 10))
signal = (x-np.mean(x))/np.std(x)
# signal = x
# signal = np.linspace(-1, 1, 5)
# x = np.zeros(50)
# x[10:30] = np.arange(20)
# x[30:50] = 20-np.arange(20)
# signal = pd.Series((x-np.mean(x))/np.std(x))

# x = np.random.normal(0,1,50)
# signal = pd.Series(x)

# x = np.linspace(-1, 1, 50)
# signal = pd.Series((x-np.mean(x))/np.std(x))

num_copies = 5000
sigma = 0.1
max_shift= 0.1

start = time.time()
# generate shifted, noisy version of the signal
observations, shifts = utils.generate_data(signal, num_copies,  max_shift, sigma, cyclic = True)
print('time to generate data = ', time.time() - start)

# N, M = observations.shape
# fig, axes = plt.subplots(10,5, figsize=(20,20));
# ax = axes.flatten()
# for i in range(50):
#     lag = shifts[i]
#     ax[i].vlines(lag, np.min(observations[:,i]), np.max(observations[:,i]), color = 'red', ls = '-.')
#     ax[i].plot(observations[:,i])
# plt.savefig('observations')
 
with open('data.npy', 'wb') as f:
    np.save(f, observations)
    np.save(f, shifts)
# signal_dict = {}
# signal_dict['DFT'] = pd.DataFrame(np.fft.fft(observations, axis = 0), 
#                                      columns = observations.columns)
# signal_dict['mean'] = signal_dict['DFT'].iloc[0,:]/len(signal_dict['DFT'])
# signal_dict['power spec'] = signal_dict['DFT'].apply(utils.power_spectrum, axis = 0)
# signal_dict['bispec'] = utils.bispectrum(signal_dict['DFT'])
# print(time.time() - start)
# mean_est, P_est, B_est = utils.invariants_from_data(observations)

L = len(signal)
np.random.seed(42)
X0 = np.random.normal(0, 1, L)
# X0 = np.zeros(L)
X_est = optimization.optimise_manopt(observations, sigma, X0, extra_inits=2)
# X_est = opt_new.optimise_manopt(observations, sigma, X0, extra_inits=0)

with open('test.npy', 'wb') as f:
    np.save(f, X_est)
    np.save(f, signal)
    np.save(f,X0)

# print(X_est)
# print(signal)
print('relative error = ', np.linalg.norm(X_est-signal)/np.linalg.norm(signal))


