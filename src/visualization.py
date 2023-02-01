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

with open('visual.npy', 'rb') as f:
    X_est = np.load(f)
    signal = np.load(f)
    X0 = np.load(f)

X_est = X_est.flatten()
plt.rcParams['text.usetex'] = True
# use convolution theorem https://en.wikipedia.org/wiki/Cross-correlation
signal_fft = np.fft.fft(signal)
X_fft = np.fft.fft(X_est)
ccf = np.fft.ifft(X_fft.conj() * signal_fft).real/np.linalg.norm(signal)/np.linalg.norm(X_est)
lag = np.argmax(ccf)
# from scipy.signal import correlate
# ccf = correlate(signal, X_est)
# ccf = smt.ccf(signal, X_est, adjusted = False)
fig, ax = plt.subplots(figsize = (15,6))
ax.stem(np.arange(len(ccf)), ccf)
plt.xlabel('Lag, k')
plt.ylabel(r'$corr(X[i], X_{est}[i+k])$')
plt.title(f'Circular CCF, best lag = {lag}')
# 95% UCL / LCL
# plt.axhline(-1.96/np.sqrt(len(ccf)), color='r', ls='--') 
# plt.axhline(1.96/np.sqrt(len(ccf)), color='r', ls='--')

plt.savefig('ccf')

L = len(X_est)
fig, ax = plt.subplots(figsize = (15,6))
ax.plot(np.arange(L),signal, label = 'true')
ax.plot(np.arange(L), np.roll(X_est,lag), label = 'estimate',linestyle = '--')
ax.plot(np.arange(L), X0, label = 'init',linestyle = ':')
# ax.plot(np.arange(L), X_est, label = 'estimate_no_adjustment')
plt.grid()
plt.legend()
plt.title('Comparison of the Original and Estimated Signals, adjusted for shifts')
plt.savefig('estimate_adjusted')