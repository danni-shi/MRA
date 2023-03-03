"""compare clustering performance of heterogeneous optimization of MRA and spectral clustering;
Compare the accuracy of lag recovery between the two methods
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import scipy.io as spio
from scipy.linalg import block_diag

import utils
import optimization

# intialise parameters
sigma_range = np.arange(0.1,2.1,0.1) # std of random gaussian noise
max_shift= 0.1 # max proportion of lateral shift
K_range = [2,3,4]
n = 2000 # number of observations we evaluate

# data path
data_path = '/Users/caribbeanbluetin/Desktop/Research/MRA_LeadLag/HeterogeneousMRA/data/'
count = 0
result = {}
for k in K_range:
    # iniitialise containers
    ARI_list = []
    ARI_list_spc = []
    
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
        classes_true = observations_mat['classes'][:n].flatten()
        X_est = results_mat['x_est']
        
        # baseline clustering method

        residuals, lags = utils.residual_lag_mat(observations)
        delta = 1
        affinity_matrix = np.exp(- residuals ** 2 / (2. * delta ** 2))

        SPC = SpectralClustering(n_clusters=k,
                                affinity = 'precomputed',
                                random_state=0).fit(affinity_matrix)
        classes_spc = SPC.labels_
        classes_est = np.apply_along_axis(lambda x: utils.assign_classes(x, X_est), 0, observations)
        ARI_list_spc.append(adjusted_rand_score(classes_true, classes_spc))
        ARI_list.append(adjusted_rand_score(classes_true, classes_est))
 
        # store results
        result[f'K={k}'] = {'classes':{'spc': classes_spc,
                                       'het': classes_est,
                                       'true': classes_true},
                            'ARI'    : {'spc': ARI_list_spc,
                                        'het': ARI_list}
                                }
    
    
    fig, ax = plt.subplots(figsize = (15,6))
    ax.plot(sigma_range, ARI_list_spc, label = 'Spectral Clustering')
    ax.plot(sigma_range, ARI_list, label = 'Assignment to Heterogeneous Signals')
    plt.grid()
    plt.legend()
    plt.title(f'Ajusted Rand Index of clustering against noise level, K = {k}')
    plt.savefig(f'../plots/ARI_K={k}')
    
with open('../results/clustering.pkl', 'wb') as f:   
        pickle.dump(result, f)
    
with open('../results/clustering.pkl', 'rb') as f:   
    x = pickle.load(f)