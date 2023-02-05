import datetime as dt
import autograd.numpy as np
#import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, dates
import os
import time

import utils
import optimization

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# option: sine wave, random gaussian, log returns
options = ['logreturns', 'sine', 'gaussian']
L = 50
synthetic_data = 'logreturns'
if synthetic_data:
    assert options.count(synthetic_data) > 0, f'{synthetic_data} is not an option'
    
    if synthetic_data == 'logreturns':
        with open('../../data/logreturn.npy', 'rb') as f:
            x = np.load(f)
        x = x[:L]
        signal = (x-np.mean(x))/np.std(x)
    elif synthetic_data == 'sine':
        x = np.linspace(0,2*np.pi, L)
        x = np.sin(x)
        signal = (x-np.mean(x))/np.std(x)
    elif synthetic_data == 'gaussian':
        signal = np.random.randn(L)

    # set parameters for generating observations
    num_copies = 500
    sigma = 1
    max_shift= 0.3
    M = num_copies
    # generate shifted, noisy version of the signal
    start = time.time()
    observations, shifts = utils.generate_data(signal, num_copies,  max_shift, sigma, cyclic = False)
    print('time to generate data = ', time.time() - start)

    # save observations
    with open('../results/data.npy', 'wb') as f:
        np.save(f, observations)
        np.save(f, shifts)
else:
    path = '../../data/OPCL_20000103_20201231.csv'
    data = pd.read_csv(path, index_col=0)
    tickers = ['XLF','XLB','XLK','XLV','XLI','XLU','XLY','XLP','XLE']
    M = 200
    data = data[data.index.isin(tickers)].iloc[:L]

# optimization
L = len(observations)
np.random.seed(42)
X0 = np.random.normal(0, 1, L)
X_est = optimization.optimise_manopt(observations, sigma, X0, extra_inits=0)

# align the estimate to original signal    
X_aligned, lag = utils.align_to_ref(X_est.flatten(),signal)
print('relative error = ', np.linalg.norm(X_aligned-signal)/np.linalg.norm(signal))

    
with open('../results/visual.npy', 'wb') as f:
    np.save(f, X_est)
    np.save(f, X_aligned)
    np.save(f, signal)
    np.save(f,X0)