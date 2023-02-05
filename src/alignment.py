import numpy as np
import matplotlib.pyplot as plt
import utils
import optimization

def get_signal(type, L):
    if type == 'logreturns':
        with open('../../data/logreturn.npy', 'rb') as f:
            signal = np.load(f)
        signal = signal[:L]      
    elif type == 'sine':
        x = np.linspace(0,2*np.pi, L)
        x = np.sin(x)
        signal = (x-np.mean(x))/np.std(x)
    elif type == 'gaussian':
        signal = np.random.randn(L)
    return signal  


sigma = 1 # std of random gaussian noise
max_shift= 0.3 # max proportion of lateral shift
M = 500 # number of copies
L = 50 # length of signal
options = ['logreturns', 'sine', 'gaussian']

def eval_alignment(type, sigma, L):
    # generate signal
    signal = get_signal(type, L)
    # generate shifted, noisy version of the signal
    observations, shifts = utils.generate_data(signal, M,  max_shift, sigma, cyclic = False)
    # estimate and align to signal
    X_est = optimization.optimise_manopt(observations, sigma)
    X_aligned, lag = utils.align_to_ref(X_est.flatten(),signal)
    # calculate lags of observation to the aligned estimate
    shifts_est = np.zeros(M)
    for i in range(M):
        _,lag = utils.align_to_ref(X_aligned, observations[:,i])
        if lag > L//2 + 1:
            lag -= L
        shifts_est[i] = lag
    # evaluate error and accuracy
    mean_error = np.linalg.norm(shifts-shifts_est, 1)/M
    accuracy = np.mean(shifts-shifts_est < 0.1) * 100
    
    return mean_error, accuracy

L = 50
n = 10 # number of points
error_list = np.zeros((n,len(options)))
acc_list = np.zeros((n,len(options)))
sigma_range = np.linspace(1,3,n)
for i in range(len(options)):
    type = options[i]
    for j in range(n):
        sigma = sigma_range[j]
        mean_error, accuracy = eval_alignment(type, sigma, L)
        error_list[j,i] = mean_error
        acc_list[j,i] = accuracy

with open('../results/alignment.npy', 'wb') as f:
    np.save(f, error_list)
    np.save(f, acc_list)
    
fig, ax = plt.subplots(figsize = (15,6))
ax.plot(sigma_range, error_list)
plt.grid()
plt.legend(options)
plt.title(f'Change of Alignment Error with Noise Level')
plt.savefig(f'../plots/align_error')

fig, ax = plt.subplots(figsize = (15,6))
ax.plot(sigma_range, acc_list)
plt.grid()
plt.legend(options)
plt.title(f'Change of Alignment Accuracy with Noise Level')
plt.savefig(f'../plots/align_acc')
    
# with open('../results/data.npy', 'rb') as f:
#     observations = np.load(f)
#     shifts = np.load(f)
# with open('../results/visual.npy', 'rb') as f:
#     X_est = np.load(f)
#     X_aligned = np.load(f)

# L, M = observations.shape
# assert L == len(X_aligned), "lengths of observations and estimated signals are different"
# shifts_est = np.zeros(M)
# for i in range(M):
#     _,lag = utils.align_to_ref(X_aligned, observations[:,i])
#     if lag > L//2 + 1:
#         lag -= L
#     shifts_est[i] = lag
# mean_error = np.linalg.norm(shifts-shifts_est, 1)/M
# accuracy = np.mean(abs(shifts-shifts_est) < 0.1) * 100
# acc = (1 - np.linalg.norm(shifts-shifts_est, 0)/M) * 100
# print(f'mean error = {mean_error:.2f}')
# print(f'accuracy = {accuracy:.2f}%')
# print(f'acc = {acc:.2f}%')