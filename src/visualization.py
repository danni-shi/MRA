import numpy as np
import matplotlib.pyplot as plt
import os
import utils

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

x = np.arange(5)
y = np.roll(x,1)
x_aligned = utils.align_to_ref(x,y)

with open('../results/visual.npy', 'rb') as f:
    X_est = np.load(f)
    X_aligned = np.load(f)
    signal = np.load(f)
    X0 = np.load(f)


X_est = X_est.flatten()
plt.rcParams['text.usetex'] = True
# use convolution theorem https://en.wikipedia.org/wiki/Cross-correlation
X_est_shifted, lag, ccf = utils.align_to_ref(X_est, signal, return_ccf = True)

fig, ax = plt.subplots(figsize = (15,6))
ax.stem(np.arange(len(X_est)), ccf)
plt.xlabel('Lag, k')
plt.ylabel(r'$corr(X[i], X_{est}[i+k])$')
plt.title(f'Circular CCF, best lag = {lag}')
# 95% UCL / LCL
plt.axhline(-1.96/np.sqrt(len(ccf)), color='r', ls='--') 
plt.axhline(1.96/np.sqrt(len(ccf)), color='r', ls='--')

plt.savefig('../plots/ccf')

L = len(X_est)
fig, ax = plt.subplots(figsize = (15,6))
ax.plot(np.arange(L),signal, label = 'true')
ax.plot(np.arange(L), X_est_shifted, label = 'estimate',linestyle = '--')
ax.plot(np.arange(L), X0, label = 'init',linestyle = ':')
# ax.plot(np.arange(L), X_est, label = 'estimate_no_adjustment')
plt.grid()
plt.legend()
plt.title('Comparison of the Original and Estimated Signals, adjusted for shifts')
plt.savefig('../plots/estimate')

with open('../results/data.npy', 'rb') as f:
    observations = np.load(f)
    shifts = np.load(f)
# plot selected observations 
fig, axes = plt.subplots(10,5, figsize=(20,20));
ax = axes.flatten()
n = 50
assert n <= observations.shape[1]
for i in range(n):
    lag = shifts[i]
    ax[i].vlines(lag, np.min(observations[:,i]), np.max(observations[:,i]), color = 'red', ls = '-.')
    ax[i].plot(observations[:,i])
plt.savefig('../plots/observations')