"""compare clustering performance of heterogeneous optimization of MRA and spectral clustering;
Compare the accuracy of lag recovery between the two methods
"""
#======== imports ===========#

import numpy as np
import pickle
from tqdm import tqdm
import os
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import scipy.io as spio
# from scipy.linalg import block_diag

import utils
import alignment

def read_data(data_path, sigma, max_shift, k, n = None):
    # read data produced by matlab
    observations_path = data_path + '_'.join(['observations',
                                              'noise' + f'{sigma:.2g}',
                                              'shift' + str(max_shift),
                                              'class' + str(k) + '.mat'])
    results_path = data_path + '_'.join(['results',
                                         'noise' + f'{sigma:.2g}',
                                         'shift' + str(max_shift),
                                         'class' + str(k) + '.mat'])
    observations_mat = spio.loadmat(observations_path)
    results_mat = spio.loadmat(results_path)
    observations = observations_mat['data'][:, :n]
    shifts = observations_mat['shifts'].flatten()[:n]
    classes_true = observations_mat['classes'].flatten()[:n] - 1
    X_est = results_mat['x_est']
    P_est = results_mat['p_est'].flatten()
    X_true = results_mat['x_true']

    return observations, shifts, classes_true, X_est, P_est, X_true

def clustering(observations, k, assumed_max_lag, score_fn):
    # --------- Clustering ----------#

    # baseline clustering method, obtain lag matrix from pairwise CCF
    affinity_matrix, lag_matrix = alignment.score_lag_mat(observations, max_lag=assumed_max_lag,
                                                          score_fn=alignment.alignment_similarity)
    SPC = SpectralClustering(n_clusters=k,
                             affinity='precomputed',
                             random_state=0).fit(affinity_matrix)

    # compare baseline and IVF clustering
    classes_spc = SPC.labels_
    classes_est = np.apply_along_axis(lambda x: utils.assign_classes(x, X_est), 0, observations)
    classes_spc_aligned = utils.align_classes(classes_spc, classes_true)
    classes_est_aligned = utils.align_classes(classes_est, classes_true)
    assert np.sum(classes_spc_aligned == classes_true) >= np.sum(classes_spc == classes_true)
    assert np.sum(classes_est_aligned == classes_true) >= np.sum(classes_est == classes_true)
    classes_spc = classes_spc_aligned
    classes_est = classes_est_aligned

    ARI_dict = {'spc': adjusted_rand_score(classes_true, classes_spc),
                'het': adjusted_rand_score(classes_true, classes_est)}

    return {}
def eval_models(models = ['pairwise', 'sync', 'spc-homo', 'het']):
    results_dict = {}
    # ----- Evaluate the lag estimation methods -----#

    # ground truth pairwise lag matrix
    lag_mat_true = alignment.lag_vec_to_mat(shifts)
    # error_penalty = int(observations.shape[0]/2)
    error_penalty = 0

    if 'pairwise' in models:
        # SPC + pairwaise correlation-based lags
        results_pair = alignment.eval_lag_mat_het(lag_matrix, lag_mat_true, \
                                                  classes_spc, classes_true, \
                                                  error_penalty)
        results_dict['pairwise'] = results_pair

    if 'sync' in models:
        # SPC + synchronization
        X_est_sync = alignment.get_synchronized_signals(observations, classes_spc, lag_matrix)
        results_sync = \
            alignment.eval_alignment_het(observations, lag_mat_true, \
                                         classes_spc, classes_true, \
                                         X_est_sync, penalty=error_penalty, \
                                         max_lag=assumed_max_lag)
        results_dict['sync'] = results_sync

    if 'spc-homo' in models:
        # SPC + homogeneous optimization
        results_spc = \
            alignment.eval_alignment_het(observations, lag_mat_true, \
                                         classes_spc, classes_true, \
                                         sigma=sigma, penalty=error_penalty, \
                                         max_lag=assumed_max_lag)
        X_est_spc = results_spc[-1]
        results_spc = results_spc[:-1]
        results_dict['spc-homo'] = results_spc

    if 'het' in models:
        # heterogeneous optimization
        results_het = \
            alignment.eval_alignment_het(observations, lag_mat_true, \
                                         classes_est, classes_true, \
                                         X_est, penalty=error_penalty, \
                                         max_lag=assumed_max_lag)`
        results_dict['het'] = results_dict
