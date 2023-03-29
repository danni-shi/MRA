import numpy as np
import matplotlib.pyplot as plt
import utils
import optimization
import pickle
import time
from tqdm import tqdm
import scipy.io as spio
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
    
    
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

def lag_mat_het(lags, classes, return_block_mat = False):
    """arrange lags vector or lags matrix into block-diagonal form based on the given class labels. 

    Args:
        lags (np array): lags vector or matrix
        classes (np array): class labels of each observation
        return_block_mat (bool, optional): if True, return the list of matrices in block-diagonal form; else return the list. Defaults to False.

    Returns:
        _type_: _description_
    """
    lag_mat_list = []

    for c in np.unique(classes):
        if lags.ndim == 2 and lags.shape[0] == lags.shape[1]:
            sub_lags = lags[classes == c, classes == c]
        else:
            sub_lags = lag_vec_to_mat(lags[classes == c])
        lag_mat_list.append(sub_lags)
    
    if return_block_mat:
        return block_diag(*lag_mat_list)
    else:
        return lag_mat_list

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
                # lag = np.argmax((np.correlate(observations[:,i], observations[:,j], 'full'))[L-1:])
                # if lag >= L//2 + 1:
                #     lag -= L
                # norm_factor = np.array(list(range(L))+list(range(L,0,-1)))
                # lag = np.argmax((np.correlate(observations[:,i], observations[:,j], 'full'))/norm_factor)
                _, lag = utils.align_to_ref(observations[:,i], observations[:,j])
                if lag >= L//2 + 1:
                    lag -= L
                lag_mat[i,j] = lag
                lag_mat[j,i] = -lag
    return lag_mat  

def get_lag_mat_het(observations, ref = None, classes = None):
    lag_mat_list = []
    
    # for c in np.unique(classes):
    #     sub_observations = observations[classes == c]
    #     sub_ref = ref.iloc[:,int(c-1)]
    #     lag_mat_list.append(get_lag_matrix(sub_observations,sub_ref))
    
    # return block_diag(*lag_mat_list)
    pass

def eval_lag_mat(lag_mat, lag_mat_true):
    """compute the relative error and accuracy of a lag matrix wrt to a ground truth lag matrix

    Args:
        lag_mat (_type_): _description_
        lag_mat_true (_type_): _description_

    Returns:
        _type_: _description_
    """
    if lag_mat_true.ndim == 1 or \
        np.count_nonzero(np.array(lag_mat_true.shape) != 1) == 1:
        lag_mat_true = lag_vec_to_mat(lag_mat_true)
        
    rel_error = np.linalg.norm(lag_mat - lag_mat_true,1)/np.linalg.norm(lag_mat_true,1)
    accuracy = np.mean(abs(lag_mat - lag_mat_true) < 0.1) * 100
    
    return rel_error, accuracy

def lag_mat_post_clustering(lag_mat, classes):
    """mask the i-j entry of the lag matrix if sample i,j are not in the same cluster.

    Args:
        lag_mat (_type_): _description_
        classes (_type_): _description_

    Returns:
        _type_: _description_
    """
    for c in np.unique(classes):
        mask = (classes==c)[:,None] * (classes!=c)[None,:]
        lag_mat[mask] = 0
    return lag_mat


def eval_lag_mat_het(lag_mat, lag_mat_true, classes, classes_true):
    """evaluate the relative error and accurcy of a lag matrix if there are more than one class of samples.

    Args:
        lag_mat (_type_): _description_
        lag_mat_true (_type_): _description_
        classes (_type_): _description_
        classes_true (_type_): _description_

    Returns:
        _type_: _description_
    """
    lag_mat = lag_mat_post_clustering(lag_mat, classes)
    lag_mat_true = lag_mat_post_clustering(lag_mat_true, classes_true)
    rel_error, accuracy = eval_lag_mat(lag_mat, lag_mat_true)
    
    return rel_error, accuracy

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
    norm = np.linalg.norm(lag_mat_true,1)
    mean_error = np.linalg.norm(lag_mat - lag_mat_true,1)/norm
    accuracy = np.mean(abs(lag_mat - lag_mat_true) < 0.1) * 100
    mean_error_0 = np.linalg.norm(lag_mat_0 - lag_mat_true,1)/norm
    accuracy_0 = np.mean(abs(lag_mat_0 - lag_mat_true)< 0.1) * 100
    
    return mean_error, accuracy, mean_error_0, accuracy_0, X_est



def eval_alignment_het(observations, shifts, classes_est = None, X_est = None, sigma = None):
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
    # initialization
    mean_error = accuracy = 0
    # assign observations to the closest cluster centre
    if classes_est is None:
        assert X_est != None, 'Cannot assign classes without cluster signals'
        classes_est = np.apply_along_axis(lambda x: utils.assign_classes(x, X_est), 0, observations)
    
    if X_est is None:
        X_est_list = []
    # evaluate the lag estimation for each cluster
    for c in np.unique(classes_est):
        
        # calculate true lags
        sub_shifts = shifts[classes_est == c]
        lag_mat_true = lag_vec_to_mat(sub_shifts)
        
        # estimate lags from data
        sub_observations = observations.T[classes_est == c].T
        if X_est is None:
            # estimate and align to signal
            sub_X_est, _, _ = optimization.optimise_matlab(sub_observations, sigma, 1)
            X_est_list.append(sub_X_est.reshape(-1,1))
        else:
            sub_X_est = X_est[:,c]
        lag_mat = get_lag_matrix(sub_observations, sub_X_est)
        
        # lag_mat_0 = get_lag_matrix(sub_observations)
        
        # evaluate error and accuracy, weighted by cluster size
        norm = np.linalg.norm(lag_mat_true,1)
        mean_error += sub_observations.shape[1]/N * np.linalg.norm(lag_mat - lag_mat_true, 1)/norm 
        accuracy += sub_observations.shape[1]/N * np.mean(abs(lag_mat - lag_mat_true) < 0.1) * 100
    
    
    if X_est is None:
        X_est = np.concatenate(X_est_list,axis=1)
        return mean_error, accuracy, X_est
    else:
        return  mean_error, accuracy

#---- Implementation of SVD-Synchronization ----#
def reconcile_score_signs(H,r, G=None):
    L = H.shape[0]
    ones = np.ones((L,1))
     
    # G is the (true) underlying measurement graph
    if G is None:
        G = np.ones((L,L)) - np.eye(L)
   
    const_on_rows = np.outer(r,ones) 
    const_on_cols = np.outer(ones,r) 
    recompH = const_on_rows - const_on_cols
    recompH = recompH * G
    
    # difMtx{1,2} have entries in {-1,0,1}
    difMtx1 = np.sign(recompH) - np.sign(H)    
    difMtx2 = np.sign(recompH) - np.sign(H.T)
    
    # Compute number of upsets:
    upset_difMtx_1 = np.sum(abs(difMtx1))/2
    upset_difMtx_2 = np.sum(abs(difMtx2))/2
    
    if upset_difMtx_1 > upset_difMtx_2:
        r = -r 
        
    return r

def SVD_NRS(H, scale_estimator = 'median'):
    """perform SVD normalised ranking and synchronization on a pairwise score matrix H to obtain a vector of lags

    Args:
        H (_type_): _description_
    """
    L = H.shape[0]
    ones = np.ones((L,1))
    
    D_inv_sqrt = np.diag(np.sqrt(abs(H).sum(axis = 1))) # diagonal matrix of sqrt of column sum of abs
    H_ss = D_inv_sqrt @ H @ D_inv_sqrt
    U, S, _ = np.linalg.svd(H_ss) # S already sorted in descending order, U are orthonormal basis
    assert np.all(S[:-1] >= S[1:]), 'Singular values are not sorted in desceding order'
    
    u1_hat = U[:,0]; u2_hat = U[:,1]
    u1 = D_inv_sqrt @ ones
    u1 /= np.linalg.norm(u1) # normalize

    u1_bar = (U[:,:2] @ U[:,:2].T @ u1).flatten()
    u1_bar /= np.linalg.norm(u1_bar) # normalize
    u2_tilde = u1_hat - np.dot(u1_hat,u1_bar)*u1_bar # same as proposed method
    u2_tilde /= np.linalg.norm(u2_tilde) # normalize
    # test
    T = np.array([np.dot(u2_hat,u1_bar),-np.dot(u1_hat,u1_bar)])
    u2_tilde_test = U[:,:2] @ T 
    
    u2_tilde_test /= np.linalg.norm(u1_bar)

    assert np.linalg.norm(u2_tilde_test.flatten()-u2_tilde) <1e-8 or np.linalg.norm(u2_tilde_test.flatten()+u2_tilde) <1e-8
    pi = D_inv_sqrt @ u2_tilde.reshape(-1,1)
    pi = reconcile_score_signs(H, pi)
    S = lag_vec_to_mat(pi)
    
    # median 
    if scale_estimator == 'median':
        
        offset = np.divide(H, S, out=np.zeros(H.shape, dtype=float), where=S!=0)
        tau = np.median(offset)
        
    # regression
    if scale_estimator == 'regression':    
        tau = np.sum(abs(np.triu(H,k=1)))/np.sum(abs(np.triu(S,k=1)))

    r = tau * pi - tau * np.dot(ones.flatten(), pi.flatten()) * ones / L
    r_test = tau * pi
    # test
    r_test = r_test - np.mean(r_test)
    assert np.linalg.norm(r_test.flatten()-r.flatten()) <1e-8 or np.linalg.norm(r_test.flatten()+r.flatten()) <1e-8
    return pi.flatten(), r.flatten(), tau


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


"""
### plot error and accuracy with data and results from MATLAB

# intialise parameters
sigma_range = np.arange(0.1,2.1,0.1) # std of random gaussian noise
max_shift= 0.1 # max proportion of lateral shift
options = ['gaussian'] # types of signals
K_range = [1]
# n = 2000 # number of observations we evaluate

# data path
data_path = '/Users/caribbeanbluetin/Desktop/Research/MRA_LeadLag/HeterogeneousMRA/data/'
count = 0
result = {}
type = options[0]
# iniitialise containers
result[type] = {}
error_list = []
acc_list = []
error_list_0 = []
acc_list_0 = []
# class_acc_list = []
k = 1  
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
    observations = observations_mat['data']
    shifts = observations_mat['shifts'].flatten()
    classes = observations_mat['classes'].flatten()
    X_est = results_mat['x_est']
    
    mean_error, accuracy, mean_error_0, accuracy_0, X_est = \
        eval_alignment(observations, shifts, sigma, X_est)
        
    
    # X_aligned, lag = utils.align_to_ref(X_est, signal)
    # print('relative error = ', np.linalg.norm(X_aligned-signal)/np.linalg.norm(signal))
    error_list.append(mean_error)
    acc_list.append(accuracy)
    error_list_0.append(mean_error_0)
    acc_list_0.append(accuracy_0)
    # class_acc_list.append(class_accuracy)
    
result[type]['accuracy'] = {'intermediate': acc_list,
                            'pairwise': acc_list_0}        
result[type]['error'] = {'intermediate': error_list,
                            'pairwise': error_list_0}        
# result[type]['class accuracy'] = class_acc_list

fig, ax = plt.subplots(figsize = (15,6))
ax.plot(sigma_range, error_list, label = 'with intermediate')
ax.plot(sigma_range, error_list_0, label = 'pairwise')
plt.grid()
plt.legend()
plt.title(f'Change of Alignment Error with Noise Level ({type} signal)')
plt.savefig(f'../plots/align_error_{type}_K=1')

fig, ax = plt.subplots(figsize = (15,6))
ax.plot(sigma_range, acc_list, label = 'with intermediate')
ax.plot(sigma_range, acc_list_0, label = 'pairwise')
plt.grid()
plt.legend()
plt.title(f'Change of Alignment Accuracy with Noise Level ({type} signal)')
plt.savefig(f'../plots/align_acc_{type}_K=1')
    
    # fig, ax = plt.subplots(figsize = (15,6))
    # ax.plot(sigma_range, class_acc_list)
    # plt.grid()
    # plt.title(f'Change of Class Assignment Accuracy with Noise Level ({type} signal)')
    # plt.savefig(f'../plots/class_acc_{type}_K={k}_0')

with open('../results/alignment_homo.pkl', 'wb') as f:   
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



# baseline: matrix entry as the residual of aligned observations, cluster them and everage the observaton in each class to recover the latent signal , then we compute the lags of observations against the signals

# 
"""