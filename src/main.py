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
from alignment import eval_alignment_het

# intialise parameters
sigma_range = np.arange(0.1,2.1,1) # std of random gaussian noise
# sigma_range = [0.3,0.4,1.3]
max_shift= 0.1 # max proportion of lateral shift
K_range = [2]
n = 200 # number of observations we evaluate

# data path
data_path = '/Users/caribbeanbluetin/Desktop/Research/MRA_LeadLag/HeterogeneousMRA/data_n=500/'
results_save_dir = utils.save_to_folder('../plots/SPC_cluster', '')
result = {}
for k in K_range:
    # iniitialise containers
    ARI_list = []
    ARI_list_spc = []
    error_list = []
    acc_list = []
    error_list_spc = []
    acc_list_spc = []
    j = 0
    
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
        classes_true = observations_mat['classes'].flatten() - 1
        X_est = results_mat['x_est']
        P_est = results_mat['p_est'].flatten()
        X_true = results_mat['x_true']

        # baseline clustering method
        # residuals, lags = utils.score_lag_mat(observations, utils.alignment_residual)
        # delta = 1
        # affinity_matrix = np.exp(- residuals ** 2 / (2. * delta ** 2))
        
        affinity_matrix, lags = utils.score_lag_mat(observations)

        SPC = SpectralClustering(n_clusters=k,
                                affinity = 'precomputed',
                                random_state=0).fit(affinity_matrix)
        classes_spc = SPC.labels_
        classes_est = np.apply_along_axis(lambda x: utils.assign_classes(x, X_est), 0, observations)
        ARI_list_spc.append(adjusted_rand_score(classes_true, classes_spc))
        ARI_list.append(adjusted_rand_score(classes_true, classes_est))

        # evaluate the estimation of lags
        
        # heterogeneous optimization
        mean_error, accuracy = \
            eval_alignment_het(observations, shifts, classes_est, X_est)
        # baseline
        mean_error_spc, accuracy_spc, X_est_spc = \
            eval_alignment_het(observations, shifts, classes_spc, sigma = sigma)
        
        error_list.append(mean_error)
        acc_list.append(accuracy)
        error_list_spc.append(mean_error_spc)
        acc_list_spc.append(accuracy_spc)

        if sigma % 0.5 < 1:
            # plot the difference of estimated signals

            X_spc_aligned, perm = utils.align_to_ref_het(X_est_spc, X_true)
            fig, ax = plt.subplots(k, 1, figsize = (10,5*k))
            for i in range(k):
                rel_err_hetero = np.linalg.norm(X_est[:,i]-X_true[:,i])/np.linalg.norm(X_true)
                rel_err_spc = np.linalg.norm(X_spc_aligned[:,i]-X_true[:,i])/np.linalg.norm(X_true)
                p_true = np.sum(classes_true==i)/observations.shape[1]
                ax[i].plot(X_true[:,i], label = 'true')
                ax[i].plot(X_est[:,i], label = 'hetero', linestyle = '--')
                ax[i].plot(X_spc_aligned[:,i], label = 'spc', linestyle = ':')
                ax[i].set_title(f'rel. err.:  hetero {rel_err_hetero:.2f}; '\
                                f'spc {rel_err_spc:.2f}; '\
                                f'true p: {p_true:.2f}, '\
                                f'est. p: {P_est[i]:.2f}')
                ax[i].grid()
                ax[i].legend()
            
            fig.suptitle(f'Comparison of the True and Estimated Signals, K = {k}, noise = {sigma:.2g}')
            plt.savefig(results_save_dir + f'/signals_K={k}_{j}')
            j += 1
    
    # store results
    result[f'K={k}'] = {'classes':{'spc': classes_spc,
                                    'het': classes_est,
                                    'true': classes_true},
                        'ARI'    : {'spc': ARI_list_spc,
                                    'het': ARI_list},
                        'error'    : {'spc': error_list_spc,
                                    'het': error_list},
                        'accuracy'    : {'spc': acc_list_spc,
                                    'het': acc_list}
                        
                            }
    
    fig, axes = plt.subplots(3,1, figsize = (15,18), squeeze=False)
    ax = axes.flatten()
    ax[0].plot(sigma_range, ARI_list, label = 'Assignment to Heterogeneous Signals')
    ax[0].plot(sigma_range, ARI_list_spc, label = 'Spectral Clustering')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_title(f'Ajusted Rand Index of clustering against noise level, K = {k}')
    
    ax[1].plot(sigma_range, error_list, label = 'Heterogeneous optimization')
    ax[1].plot(sigma_range, error_list_spc, label = 'Spectral Clustering')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_title(f'Change of Alignment Error with Noise Level')

    ax[2].plot(sigma_range, acc_list, label = 'Heterogeneous optimization')
    ax[2].plot(sigma_range, acc_list_spc, label = 'Spectral Clustering')
    ax[2].grid()
    ax[2].legend()
    ax[2].set_title(f'Change of Alignment Accuracy with Noise Level')
    plt.savefig(results_save_dir + f'/acc_err_ARI_K={k}_0')
    
with open('../results/clustering.pkl', 'wb') as f:   
        pickle.dump(result, f)


### If we only want clustering results 

# for k in K_range:
#     # iniitialise containers
#     ARI_list = []
#     ARI_list_spc = []
    
#     for sigma in tqdm(sigma_range):
#         # read data produced by matlab
#         observations_path = data_path + '_'.join(['observations', 
#                                       'noise'+f'{sigma:.2g}', 
#                                       'shift'+str(max_shift), 
#                                       'class'+str(k)+'.mat'])
#         results_path = data_path + '_'.join(['results', 
#                                       'noise'+f'{sigma:.2g}', 
#                                       'shift'+str(max_shift), 
#                                       'class'+str(k)+'.mat'])
#         observations_mat = spio.loadmat(observations_path)
#         results_mat = spio.loadmat(results_path)
#         observations = observations_mat['data']
#         shifts = observations_mat['shifts'].flatten()
#         classes_true = observations_mat['classes'].flatten() - 1
#         X_est = results_mat['x_est']
#         P_est = results_mat['p_est'].flatten()

#         # baseline clustering method
#         residuals, lags = utils.residual_lag_mat(observations)
#         delta = 1
#         affinity_matrix = np.exp(- residuals ** 2 / (2. * delta ** 2))

#         SPC = SpectralClustering(n_clusters=k,
#                                 affinity = 'precomputed',
#                                 random_state=0).fit(affinity_matrix)
#         classes_spc = SPC.labels_
#         classes_est = np.apply_along_axis(lambda x: utils.assign_classes(x, X_est), 0, observations)
#         ARI_list_spc.append(adjusted_rand_score(classes_true, classes_spc))
#         ARI_list.append(adjusted_rand_score(classes_true, classes_est))

#     fig, axes = plt.subplots( figsize = (15,6))
#     ax = axes.flatten()
#     ax.plot(sigma_range, ARI_list, label = 'Assignment to Heterogeneous Signals')
#     ax.plot(sigma_range, ARI_list_spc, label = 'Spectral Clustering')
#     ax.grid()
#     ax.legend()
#     ax.set_title(f'Ajusted Rand Index of clustering against noise level, K = {k}')
#     plt.savefig(results_save_dir + f'/ARI_K={k}_0')