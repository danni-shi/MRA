"""compare clustering performance of heterogeneous optimization of MRA and spectral clustering;
Compare the accuracy of lag recovery between the two methods
"""
# ======== imports ===========#

import numpy as np
import pickle
from tqdm import tqdm
import os
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import scipy.io as spio
import multiprocessing
import time
import shutil

import utils
import alignment


def read_data(data_path, sigma, max_shift, k, n=None):
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


def clustering(observations, k, classes_true, assumed_max_lag, X_est, score_fn=alignment.alignment_similarity):
    # --------- Clustering ----------#

    # baseline clustering method, obtain lag matrix from pairwise CCF
    affinity_matrix, lag_matrix = alignment.score_lag_mat(observations, max_lag=assumed_max_lag,
                                                          score_fn=score_fn)
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

    return classes_spc, classes_est, lag_matrix, ARI_dict


def eval_models(lag_matrix, shifts, assumed_max_lag, \
                models=['pairwise', 'sync', 'spc-homo', 'het'],
                observations=None,
                classes_true=None,
                classes_spc=None,
                classes_est=None,
                X_est=None,
                sigma=None,
                return_lag_mat=False):
    results_dict = {}
    signal_dict = {}
    lag_mat_dict = {}

    # ----- Evaluate the lag estimation methods -----#

    # ground truth pairwise lag matrix
    lag_mat_true = alignment.lag_vec_to_mat(shifts)
    # error_penalty = int(observations.shape[0]/2)
    error_penalty = 0

    if 'pairwise' in models:
        # SPC + pairwise correlation-based lags
        results_pair = alignment.eval_lag_mat_het(lag_matrix,
                                                  lag_mat_true,
                                                  classes_spc,
                                                  classes_true,
                                                  error_penalty)
        results_dict['pairwise'] = results_pair
        lag_mat_dict['pairwise'] = lag_matrix

    if 'sync' in models:
        # SPC + synchronization
        X_est_sync = alignment.get_synchronized_signals(observations,
                                                        classes_spc,
                                                        lag_matrix)
        lag_mat_sync = alignment.get_lag_matrix_het(observations, classes_spc, X_est_sync, assumed_max_lag)
        results_sync = alignment.eval_lag_mat_het(lag_mat_sync, lag_mat_true, classes_spc, classes_true,
                                                  error_penalty)
        # results_sync = \
        #     alignment.eval_alignment_het(observations,
        #                                  lag_mat_true,
        #                                  classes_spc,
        #                                  classes_true,
        #                                  X_est_sync,
        #                                  penalty=error_penalty,
        #                                  max_lag=assumed_max_lag)
        results_dict['sync'] = results_sync
        signal_dict['sync'] = X_est_sync
        lag_mat_dict['sync'] = lag_mat_sync

    if 'spc-homo' in models:
        # SPC + homogeneous optimization
        X_est_spc_homo = alignment.latent_signal_homo(observations, classes_spc, sigma)
        lag_mat_spc_homo = alignment.get_lag_matrix_het(observations, classes_spc, X_est_spc_homo, assumed_max_lag)
        results_spc_homo = alignment.eval_lag_mat_het(lag_mat_spc_homo, lag_mat_true, classes_spc, classes_true,
                                                      error_penalty)
        # results_spc = \
        #     alignment.eval_alignment_het(observations,
        #                                  lag_mat_true,
        #                                  classes_spc,
        #                                  classes_true,
        #                                  X_est_spc_homo,
        #                                  penalty=error_penalty,
        #                                  max_lag=assumed_max_lag)

        results_dict['spc-homo'] = results_spc_homo
        signal_dict['spc-homo'] = X_est_spc_homo
        lag_mat_dict['spc-homo'] = lag_mat_spc_homo

    if 'het' in models:
        # classes = np.apply_along_axis(lambda x: utils.assign_classes(x, kwargs['X_est']), 0, kwargs['observations'])
        #
        # lag_mat_het = alignment.get_lag_matrix_het(kwargs['observations'],
        #                                            classes=classes,
        #                                            X_est=kwargs['X_est'],
        #                                            max_lag=assumed_max_lag)
        #
        # results_het = alignment.eval_lag_mat_het(lag_mat_het,
        #                                           lag_mat_true,
        #                                           classes,
        #                                           kwargs['classes_true'],
        #                                           error_penalty)

        # heterogeneous optimization
        lag_mat_het = alignment.get_lag_matrix_het(observations, classes_est, X_est, assumed_max_lag)
        results_het = alignment.eval_lag_mat_het(lag_mat_het, lag_mat_true, classes_est, classes_true, error_penalty)
        # results_het = \
        #     alignment.eval_alignment_het(observations,
        #                                  lag_mat_true,
        #                                  classes_est,
        #                                  classes_true,
        #                                  X_est,
        #                                  penalty=error_penalty,
        #                                  max_lag=assumed_max_lag)
        results_dict['het'] = results_het
        signal_dict['het'] = X_est
        lag_mat_dict['het'] = lag_mat_het

    if return_lag_mat:
        return results_dict, signal_dict, lag_mat_dict
    else:
        return results_dict, signal_dict


def align_all_signals(X_est_sync, X_est_spc, X_true, classes_spc, classes_est, classes_true, k, X_est, P_est):
    # aligned the estimated signals and mixing probabilities to the ground truth
    X_est_sync_aligned, perm = utils.align_to_ref_het(X_est_sync, X_true)

    X_est_spc_aligned, perm = utils.align_to_ref_het(X_est_spc, X_true)
    prob_spc = utils.mixing_prob(classes_spc, k)
    prob_spc = np.array([prob_spc[i] for i in perm])

    # reminder that the estimations from heterogeneous IVF method is already aligned with the truth
    prob_het_reassigned = utils.mixing_prob(classes_est, k)

    # true mixing probabilities
    P_true = [np.mean(classes_true == c) for c in np.unique(classes_true)]

    # record estimations for each K and sigma
    signal_class_prob = {
        'signals': {'true': X_true,
                    'sync': X_est_sync_aligned,
                    'spc-homo': X_est_spc_aligned,
                    'het': X_est
                    },
        'classes': {'true': classes_true,
                    'spc': classes_spc,
                    'het': classes_est
                    },
        'probabilities': {'true': P_true,
                          'spc-homo': prob_spc,
                          'het reassigned': prob_het_reassigned,
                          'het': P_est
                          }
    }
    return signal_class_prob


def initialise_containers(K_range, models):
    metrics = ['error', 'error_sign', 'accuracy', 'errors_quantile']

    performance = {}
    estimates = {}

    for k in K_range:
        performance[f'K={k}'] = {'ARI': {'spc': [],
                                         'het': []}}
        for metric in metrics:
            performance[f'K={k}'][metric] = {model: [] for model in models}

        estimates[f'K={k}'] = {}

    return performance, estimates


"""
model labels: ['pairwise', 'sync', 'spc-homo', 'het']
"""

def empty_folders():
    if os.path.exists('../results/performance'):
        shutil.rmtree('../results/performance')
    os.mkdir('../results/performance')
    if os.path.exists('../results/signal_estimates'):
        shutil.rmtree('../results/signal_estimates')
    os.mkdir('../results/signal_estimates')
    if os.path.exists('results/lag_matrices'):
        shutil.rmtree('results/lag_matrices')
    os.mkdir('../results/lag_matrices')


def run(sigma_range=np.arange(0.1, 2.1, 0.1),
        K_range=None,
        n=None,
        test=False,
        max_shift=0.04,
        assumed_max_lag=5,
        models=None,
        data_path='../../data/data500_shift0.04_pvCLCL_init2/',
        return_signals=False,
        return_lag_mat=False,
        round=1):
    if models is None:
        models = ['pairwise', 'sync', 'spc-homo', 'het']
    if K_range is None:
        K_range = [2, 3, 4]
    if test:
        sigma_range = np.arange(0.1, 2.0, 0.5)
        K_range = [2]
    metrics = ['error', 'error_sign', 'accuracy', 'errors_quantile']

    # initialise containers
    performance = {}
    estimates = {}
    lag_matrices = {}
    for k in tqdm(K_range):
        performance[f'K={k}'] = {'ARI': {'spc': [],
                                         'het': []}}
        for metric in metrics:
            performance[f'K={k}'][metric] = {model: [] for model in models}

        estimates[f'K={k}'] = {}
        lag_matrices[f'K={k}'] = {}

        for sigma in tqdm(sigma_range):

            # read data produced from matlab code base
            observations, shifts, classes_true, X_est, P_est, X_true = read_data(
                data_path=data_path + str(round) + '/',
                sigma=sigma,
                max_shift=max_shift,
                k=k,
                n=n
            )
            # args_dict = {'observations'}

            # calculate clustering and pairwise lag matrix
            classes_spc, classes_est, lag_matrix, ARI_dict = clustering(observations=observations,
                                                                        k=k,
                                                                        classes_true=classes_true,
                                                                        assumed_max_lag=assumed_max_lag,
                                                                        X_est=X_est
                                                                        )

            # evaluate model performance in lag predictions
            results = eval_models(lag_matrix=lag_matrix,
                                  shifts=shifts,
                                  assumed_max_lag=assumed_max_lag,
                                  classes_true=classes_true,
                                  models=models,
                                  observations=observations,
                                  classes_spc=classes_spc,
                                  classes_est=classes_est,
                                  X_est=X_est,
                                  sigma=sigma,
                                  return_lag_mat=return_lag_mat
                                  )
            # store model performance results in dictionaries

            # clustering performance
            for label, value in ARI_dict.items():
                performance[f'K={k}']['ARI'][label].append(value)
            # prediction performance
            results_dict = results[0]
            for i in range(len(metrics)):
                for model in models:
                    metric = metrics[i]
                    metric_result = results_dict[model][i]
                    performance[f'K={k}'][metric][model].append(metric_result)

            if return_signals:
                signal_dict = results[1]
                # organize signal estimates, classes estimates and mixing prob estimates
                signal_class_prob = align_all_signals(X_est_sync=signal_dict['sync'],
                                                      X_est_spc=signal_dict['spc-homo'],
                                                      X_true=X_true,
                                                      classes_spc=classes_spc,
                                                      classes_est=classes_est,
                                                      classes_true=classes_true,
                                                      k=k,
                                                      X_est=X_est,
                                                      P_est=P_est
                                                      )
                # store the signal estimates, classes estimates and mixing prob estimates
                estimates[f'K={k}'][f'sigma={sigma:.2g}'] = signal_class_prob

            # store the  lag matrices predicted by the models
            if return_lag_mat:
                lag_mat_dict = results[2]
                lag_matrices[f'K={k}'][f'sigma={sigma:.2g}'] = lag_mat_dict


    # save the results to folder
    empty_folders()
    with open(f'../results/performance/{round}.pkl', 'wb') as f:
        pickle.dump(performance, f)

    if return_signals:
        with open(f'../results/signal_estimates/{round}.pkl', 'wb') as f:
            pickle.dump(estimates, f)
    if return_lag_mat:
        with open(f'../results/lag_matrices/{round}.pkl', 'wb') as f:
            pickle.dump(lag_matrices, f)

def run_wrapper(round):
    run(test=False, return_signals=True, round=round)


if __name__ == "__main__":
    # remember to untick 'Run with Python console' in config
    # rounds = 4
    # inputs = range(1, 1+rounds)
    # start = time.time()
    # with multiprocessing.Pool() as pool:
    #     # use the pool to apply the worker function to each input in parallel
    #     pool.map(run_wrapper, inputs)
    #     pool.close()
    # print(f'time taken to run {rounds} rounds: {time}')
    run(max_shift=0.04, test=True, return_signals=True, return_lag_mat=True, round=1)

    # TODO: parallelize main()
    # TODO: modify align_all_signals to process only the outputs of the selected models
    # note the current maximum shift is 4
