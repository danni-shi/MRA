import numpy as np
import matplotlib.pyplot as plt
import utils
import optimization
import pickle
import time
from tqdm import tqdm
import scipy.io as spio
from scipy.linalg import block_diag

    
    
def get_signal(type, L):
    if type == 'logreturns':
        with open('../../data/logreturn.npy', 'rb') as f:
            signal = np.load(f)
        signal = signal[L:2*L]      
    elif type == 'sine':
        x = np.linspace(0,2*np.pi, L)
        x = np.sin(x)
        signal = (x-np.mean(x))/np.std(x)
    elif type == 'gaussian':
        signal = np.random.randn(L)
    return signal  

def lag_vec_to_mat(vec):
    # note this function is invariant to addition and subtraction 
    # of the same value to every element of vec
    L = len(vec)
    vec = vec.reshape(-1,1)
    ones = np.ones((L,1))
    return vec @ ones.T - ones @ vec.T

assert np.linalg.norm(lag_vec_to_mat(np.array([0,1]))-np.array([[0,-1],[1,0]])) < 1e-10
assert np.linalg.norm(lag_vec_to_mat(np.array([1,2,3]))-np.array([[0,-1,-2],[1,0,-1],[2,1,0]])) < 1e-10

def get_lag_matrix(observations, ref = None):
    """calculate the best lags estimates of a given set of observations, with or without a latent reference signal

    Args:
        observations (np array): L x N matrix with columns consist of time series
        ref (np array, optional): 1-dim length L reference time series signal. Defaults to None.
    """
    L, N = observations.shape
    
    if ref is not None:
        assert len(ref == L)
        shifts_est = np.zeros(N)
        for i in range(N):
            _, lag = utils.align_to_ref(observations[:,i], ref)
            shifts_est[i] = lag
        lag_mat = lag_vec_to_mat(shifts_est)
        for i in range(N):
            for j in range(N):
                if abs(lag_mat[i,j]) >= L//2 + 1:
                    lag_mat[i,j] -= np.sign(lag_mat[i,j]) * L
    else:
        lag_mat = np.zeros((N,N))
        for j in range(N):
            for i in range(j):
                lag = np.argmax((np.correlate(observations[:,i], observations[:,j], 'full'))[L-1:])
                if lag >= L//2 + 1:
                    lag -= L
                lag_mat[i,j] = lag
                lag_mat[j,i] = -lag
    return lag_mat  

def get_lag_mat_het(observations, ref = None, classes = None):
    lag_mat_list = []
    
    for c in np.unique(classes):
        sub_observations = observations[classes == c]
        sub_ref = ref.iloc[:,int(c-1)]
        lag_mat_list.append(get_lag_matrix(sub_observations,sub_ref))
    
    return block_diag(*lag_mat_list)
    
def eval_alignment(observations, shifts, sigma, X_est = None):
    """compare the performance of lead-lag predition using intermidiate latent signal to naive pairwise prediciton

    Args:
        observations (np array): L x N matrix with columns consist of time series
        shifts (np array): 1-dim array that contains the ground true lags of the observations to some unknown signal

    Returns:
        mean_error: error of prediction
        accuracy: accuracy of prediction
        mean_error_0: error of naive approach
        accuracy_0: accuracy of naive approach

    """
    L, N = observations.shape
    lag_mat_true = lag_vec_to_mat(shifts)
    
    if X_est is None:
        # estimate and align to signal
        X_est = optimization.optimise_manopt(observations, sigma)
    
    # calculate lags of observation to the aligned estimate
    lag_mat = get_lag_matrix(observations, X_est)
    lag_mat_0 = get_lag_matrix(observations)
    
    # evaluate error and accuracy
    norm = np.linalg.norm(lag_mat_true)
    mean_error = np.linalg.norm(lag_mat - lag_mat_true)/norm
    accuracy = np.mean(abs(lag_mat - lag_mat_true) < 0.1) * 100
    mean_error_0 = np.linalg.norm(lag_mat_0 - lag_mat_true)/norm
    accuracy_0 = np.mean(abs(lag_mat_0 - lag_mat_true)< 0.1) * 100
    
    return mean_error, accuracy, mean_error_0, accuracy_0, X_est

def assign_classes(observation, X_est):
    dist = []
    for k in range(X_est.shape[1]):
        dist.append(np.linalg.norm(utils.align_to_ref(observation, X_est[:,k])[0]- X_est[:,k])**2)
    return np.argmin(dist) + 1

def eval_alignment_het(observations, shifts, X_est, classes = None):
    """compare the performance of lead-lag predition using intermidiate latent signal to naive pairwise prediciton

    Args:
        observations (np array): L x N matrix with columns consist of time series
        shifts (np array): 1-dim array that contains the ground true lags of the observations to some unknown signal

    Returns:
        mean_error: error of prediction
        accuracy: accuracy of prediction
        mean_error_0: error of naive approach
        accuracy_0: accuracy of naive approach

    """
    L, N = observations.shape
    mean_error = mean_error_0 = accuracy = accuracy_0 = 0
    classes_est = np.apply_along_axis(lambda x: assign_classes(x, X_est), 0, observations)
    class_accuracy = np.mean(classes_est == classes)
    
    for c in np.unique(classes):
        sub_X_est = X_est[:,c-1]
        sub_observations = observations.T[classes == c].T
        sub_shifts = shifts[classes == c]
        
        # calculate lags of observation to the aligned estimate
        lag_mat_true = lag_vec_to_mat(sub_shifts)
        lag_mat = get_lag_matrix(sub_observations, sub_X_est)
        lag_mat_0 = get_lag_matrix(sub_observations)
        
        # evaluate error and accuracy
        norm = np.linalg.norm(lag_mat_true)
        mean_error += np.linalg.norm(lag_mat - lag_mat_true)/norm
        accuracy += np.mean(abs(lag_mat - lag_mat_true) < 0.1) * 100
        mean_error_0 += np.linalg.norm(lag_mat_0 - lag_mat_true)/norm
        accuracy_0 += np.mean(abs(lag_mat_0 - lag_mat_true)< 0.1) * 100
    
    return  mean_error/len(np.unique(classes)),\
            accuracy/len(np.unique(classes)),\
            mean_error_0/len(np.unique(classes)),\
            accuracy_0//len(np.unique(classes)),\
            class_accuracy,\
            X_est



def align_plot():
    # intialise parameters for generating observations
    L = 50 # length of signal
    N = 500 # number of copies
    sigma = 1 # std of random gaussian noise
    max_shift= 0.1 # max proportion of lateral shift

    # intialise parameter for experiments
    n = 10 # number of points
    sigma_range = np.linspace(0.1,3,n) # range of noise level
    options = ['logreturns', 'sine', 'gaussian'] # types of signals

    count = 0
    result = {}
    for i in range(len(options)):
        type = options[i]
        # iniitialise containers
        result[type] = {}
        error_list = np.zeros(n)
        acc_list = np.zeros(n)
        error_list_0 = np.zeros(n)
        acc_list_0 = np.zeros(n)
        
        
        # generate signal
        signal = get_signal(type, L)
        
        for j in range(n):
            sigma = sigma_range[j]
            # generate shifted, noisy version of the signal
            observations, shifts = utils.generate_data(signal, N, max_shift, sigma, cyclic = False)
            mean_error, accuracy, mean_error_0, accuracy_0, X_est = eval_alignment(observations, shifts, sigma)
            X_aligned, lag = utils.align_to_ref(X_est, signal)
            print('relative error = ', np.linalg.norm(X_aligned-signal)/np.linalg.norm(signal))
            error_list[j] = mean_error
            acc_list[j] = accuracy
            error_list_0[j] = mean_error_0
            acc_list_0[j] = accuracy_0
            count += 1
            print(f'{count}/{n*len(options)} steps completed')
        result[type]['accuracy'] = {'intermediate': acc_list,
                                    'pairwise': acc_list_0}        
        result[type]['error'] = {'intermediate': error_list,
                                    'pairwise': error_list_0}        
        
        fig, ax = plt.subplots(figsize = (15,6))
        ax.plot(sigma_range, error_list, label = 'with intermediate')
        ax.plot(sigma_range, error_list_0, label = 'pairwise')
        plt.grid()
        plt.legend()
        plt.title(f'Change of Alignment Error with Noise Level ({type} signal)')
        plt.savefig(f'../plots/align_error_{type}')

        fig, ax = plt.subplots(figsize = (15,6))
        ax.plot(sigma_range, acc_list, label = 'with intermediate')
        ax.plot(sigma_range, acc_list_0, label = 'pairwise')
        plt.grid()
        plt.legend()
        plt.title(f'Change of Alignment Accuracy with Noise Level ({type} signal)')
        plt.savefig(f'../plots/align_acc_{type}')

    with open('../results/alignment.pkl', 'wb') as f:   
        pickle.dump(result, f)



### plot error and accuracy with data and results from MATLAB

# intialise parameters
sigma_range = np.arange(0.1,2.1,0.1) # std of random gaussian noise
max_shift= 0.1 # max proportion of lateral shift
options = ['gaussian'] # types of signals
K_range = [1,2,3,4]
n = 1000 # number of observations we evaluate

# data path
data_path = '/Users/caribbeanbluetin/Desktop/Research/MRA_LeadLag/HeterogeneousMRA/data/'
count = 0
result = {}
type = options[0]
for k in K_range:
    # iniitialise containers
    result[type] = {}
    error_list = []
    acc_list = []
    error_list_0 = []
    acc_list_0 = []
    class_acc_list = []
    
    for sigma in tqdm(sigma_range):
        # read data produced by matlab
        observations_path = data_path + '_'.join(['observations', 
                                      'noise'+f'{sigma:.2g}', 
                                      'shift'+str(max_shift), 
                                      'class'+str(k)+'.mat'])
        results_path = data_path + '_'.join(['results', 
                                      'noise'+f'{sigma:.2g}', 
                                      'shift'+str(max_shift), 
                                      'class'+str(k)+'.mat'])
        observations_mat = spio.loadmat(observations_path)
        results_mat = spio.loadmat(results_path)
        observations = observations_mat['data'][:,:n]
        shifts = observations_mat['shifts'][:n].flatten()
        classes = observations_mat['classes'][:n].flatten()
        X_est = results_mat['x_est']
        
        mean_error, accuracy, mean_error_0, accuracy_0, class_accuracy, X_est = \
            eval_alignment_het(observations, shifts, X_est, classes)
        # X_aligned, lag = utils.align_to_ref(X_est, signal)
        # print('relative error = ', np.linalg.norm(X_aligned-signal)/np.linalg.norm(signal))
        error_list.append(mean_error)
        acc_list.append(accuracy)
        error_list_0.append(mean_error_0)
        acc_list_0.append(accuracy_0)
        class_acc_list.append(class_accuracy)
        
    result[type]['accuracy'] = {'intermediate': acc_list,
                                'pairwise': acc_list_0}        
    result[type]['error'] = {'intermediate': error_list,
                                'pairwise': error_list_0}        
    result[type]['class accuracy'] = class_acc_list
    
    fig, ax = plt.subplots(figsize = (15,6))
    ax.plot(sigma_range, error_list, label = 'with intermediate')
    ax.plot(sigma_range, error_list_0, label = 'pairwise')
    plt.grid()
    plt.legend()
    plt.title(f'Change of Alignment Error with Noise Level ({type} signal)')
    plt.savefig(f'../plots/align_error_{type}_K={k}')

    fig, ax = plt.subplots(figsize = (15,6))
    ax.plot(sigma_range, acc_list, label = 'with intermediate')
    ax.plot(sigma_range, acc_list_0, label = 'pairwise')
    plt.grid()
    plt.legend()
    plt.title(f'Change of Alignment Accuracy with Noise Level ({type} signal)')
    plt.savefig(f'../plots/align_acc_{type}_K={k}')
    
    fig, ax = plt.subplots(figsize = (15,6))
    ax.plot(sigma_range, class_acc_list)
    plt.grid()
    plt.title(f'Change of Class Assignment Accuracy with Noise Level ({type} signal)')
    plt.savefig(f'../plots/class_acc_{type}_K={k}')

with open('../results/alignment.pkl', 'wb') as f:   
    pickle.dump(result, f)
    
    
    
# L = 50 # length of signal
# N = 500 # number of copies
# sigma = 1 # std of random gaussian noise
# max_shift= 0.1
# signal = get_signal('sine', L)

# with open('../results/data.npy', 'rb') as f:
#     observations = np.load(f)
#     shifts = np.load(f)
    
# with open('../results/visual.npy', 'rb') as f:
#     X_est = np.load(f)
#     X_aligned = np.load(f)

# # calculate lags of observation to the aligned estimate
# lag_mat = get_lag_matrix(observations, X_est)
# lag_mat_0 = get_lag_matrix(observations)
# lag_mat_true = lag_vec_to_mat(shifts)

# # evaluate error and accuracy
# mean_error = np.linalg.norm(lag_mat - lag_mat_true)/np.linalg.norm(lag_mat_true)
# accuracy = np.mean(abs(lag_mat - lag_mat_true) < 0.1) * 100
# mean_error_0 = np.linalg.norm(lag_mat_0 - lag_mat_true)/np.linalg.norm(lag_mat_true)
# accuracy_0 = np.mean(abs(lag_mat_0 - lag_mat_true)< 0.1) * 100

# print(f'accuracy of lag predictions: experiment: {accuracy:.2f}%; benchmark: {accuracy_0:.2f}%')
# print(f'relative error of lag predictions: experiment: {mean_error:.2f}; benchmark: {mean_error_0:.2f}')

