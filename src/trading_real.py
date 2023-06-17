import numpy as np
import pandas as pd
import alignment
import warnings
import time
import scipy.io as spio
import pickle
from trading import *
from tqdm import tqdm




def strategy_lag_groups(returns, trading_start, trading_end,
                        days_advanced, lag_vector, lags,
                        watch_period=1, hold_period=1,
                        market_index=None, hedge=True):
    # including returns from dates before trading takes place to construct trading signals
    # pi, r, _ = alignment.SVD_NRS(lag_matrix)
    # lag_vector = np.array(np.round(r), dtype=int)
    sub_returns = returns.iloc[:, trading_start - days_advanced:trading_end]
    market_index = market_index.iloc[trading_start - days_advanced:trading_end]
    PnL = {}
    signals = {}
    PnL_excess = {}

    sub_returns = np.array(sub_returns)
    for l1 in lags:
        for l2 in lags:
            if l1 < l2:
                leaders = np.where(lag_vector == l1)[0]
                laggers = np.where(lag_vector == l2)[0]
                lag = l2 - l1
                pnl, signal = PnL_two_groups(sub_returns, leaders, laggers, lag, watch_period, hold_period)
                PnL[f'{l1}->{l2}'] = pnl[days_advanced:]
                signals[f'{l1}->{l2}'] = signal[days_advanced:]
                if hedge:
                    pnl_excess = pnl - np.sign(signal) * np.array(market_index)
                    PnL_excess[f'{l1}->{l2}'] = pnl_excess[days_advanced:]

        # calculate the simple average of PnL of each group pair
        PnL['average'] = np.nanmean(np.stack(list(PnL.values())), axis=0)
        PnL_excess['average'] = np.nanmean(np.stack(list(PnL_excess.values())), axis=0)

        # fill nans with 0 for every value in the results dictionary
        for values in PnL.values():
            values[np.isnan(values)] = 0
        for values in PnL_excess.values():
            values[np.isnan(values)] = 0

        results_dict = group_performance(PnL, signals)
        results_excess_dict = group_performance(PnL_excess, signals)

    return {'raw returns': results_dict,
            'excess returns': results_excess_dict}


def group_performance(PnL, signals):
    # results contains a dictionary for each metric which contains metrics of each group
    results_dict = {}
    results_dict['PnL'] = PnL
    fin_stats_by_group = {}
    # fin_dict = {}
    # for group, returns in results.items():
    #     signal = signals[group]
    #     fin_stats = financial_stats(returns,signal)
    #     for metric, value in fin_stats.items():
    #         metric_dict = fin_dict.get(metric,{})
    #         metric_dict[group] = value
    #     fin_dict[metric] =
    for group in signals.keys():
        lag_pair = tuple(int(a) for a in group.split('->'))
        assert lag_pair[0] < lag_pair[1]
        lag = lag_pair[1] - lag_pair[0]
        fin_stats = financial_stats(PnL[group][lag:], signals[group][lag:])
        fin_stats_by_group[group] = fin_stats
    # fin_stats_by_group = {group:financial_stats(PnL[group],signals[group]) for group in signals.keys()}
    fin_metrics = ['annualized SR', 'corr SP',
                   'corr SP p-value', 'hit ratio',
                   'long ratio', 'reg R2']
    fin_stats_by_metric = {metric: {group: fin_stats_by_group[group][metric]
                                    for group in fin_stats_by_group}
                           for metric in fin_metrics}
    fin_stats_by_metric['annualized SR']['average'] = annualized_sharpe_ratio(PnL['average'])
    results_dict['financial_stats'] = fin_stats_by_metric

    return results_dict


def strategy_multiple_lags(returns, lag_matrix, watch_period=1, hold_period=1, leader_prop=0.2, lagger_prop=0.2,
                           rank='plain',
                           hedge='no'):
    L, N = returns.shape
    returns = returns.T
    # lag_ij = shift_i - shift_j
    # positive means i lag j
    ranking = np.mean(np.sign(lag_matrix), axis=1)  # vanilla
    ranking1 = np.mean(lag_matrix, axis=1)  # plain
    ranking2 = alignment.SVD_NRS(lag_matrix)[0]  # synchro
    sort_index = np.argsort(ranking)  # ascending
    lag_ind = sort_index[-int(lagger_prop * N):]
    # sort_index1 = np.argsort(ranking1)  # ascending
    # lag_ind1 = sort_index1[-int(lagger_prop * N):]
    # sort_index2 = np.argsort(ranking2)  # ascending
    # lag_ind2 = sort_index2[-int(lagger_prop * N):]
    # lag_mat_nan = lag_matrix.copy()
    # np.fill_diagonal(lag_mat_nan, np.nan)
    # leaders = lag_mat_nan[lead_ind,:]
    laggers = lag_matrix[lag_ind, :].astype(int)
    # for every lagger we trade, find all the leaders and respective lags
    leaders_list_by_laggers = [[(i, lag) for i, lag in enumerate(row) if lag > 0] for row in laggers]
    lagger_returns = returns[lag_ind, :]

    portfolio_returns = []
    for t in range(watch_period, L - hold_period):
        signals_by_leader = [np.sum([returns[p[0], t - p[1]] for p in l if (t - p[1] >= 0)]) \
                             for l in leaders_list_by_laggers]
        # weights proportional to the strength of signals, sum of absolute values equal to 1
        weights = winsorize(signals_by_leader)
        weights = weights / np.sum(np.abs(weights) + 1e-9)
        # total returns of the weighted portfolio
        alpha = np.dot(weights, np.sum(lagger_returns[:, t:t + hold_period], axis=1))
        # alpha1 = np.average(np.sum(lagger_returns[:,t:t+hold_period],axis=1),weights=signals_by_leader)
        # assert abs(alpha - alpha1) < 1e-8
        if hedge == 'no':
            portfolio_returns.append(alpha)
        elif hedge == 'mkt':
            alpha2 = alpha - np.sum(signals_by_leader) * (
                returns.loc['SPY'][returns.columns[t:t + hold_period]].sum(axis=0))
            portfolio_returns.append(alpha2)

    return portfolio_returns


def strategy_plain(returns, lag_matrix, shifts, watch_period=1, hold_period=1, leader_prop=0.2, lagger_prop=0.2,
                   rank='plain',
                   hedge='no'):
    """

    Args:
        returns:
        lag_matrix:
        watch_period:
        hold_period:
        leader_prop:
        lagger_prop:
        rank:
        hedge:

    Returns: the simple returns of the asset at each time step

    """
    result = []
    signs = []
    df = pd.DataFrame(lag_matrix)
    L, N = returns.shape
    returns = pd.DataFrame(returns.T)

    # df = pd.read_csv(matrix[i]) # NxN matrix where each element is a pairwise lag
    # df = df.set_index('Unnamed: 0')
    # df.columns = df1.columns
    # df.index = df1.index

    # date = daily_return.columns[i]

    if rank == 'plain':
        ranking = np.mean(lag_matrix, axis=1)

    elif rank == 'Synchro':
        ranking = alignment.SVD_NRS(lag_matrix)[0]

    sort_index = np.argsort(ranking)
    lead_ind = sort_index[:int(leader_prop * N)]
    lag_ind = sort_index[-int(lagger_prop * N):]

    # calculate the average lag between the leader and lagger groups
    lag_mat_nan = lag_matrix.copy()
    np.fill_diagonal(lag_mat_nan, np.nan)
    leaders = lag_mat_nan[:, lead_ind]
    laggers = lag_mat_nan[:, lag_ind]
    ahead = np.mean(leaders[~np.isnan(leaders)]) - np.mean(laggers[~np.isnan(laggers)])
    ahead = max(0, round(ahead) - 1)
    # 选取leader和lagger

    # 找到leader和lagger的return
    leader_returns = returns.iloc[lead_ind]
    lagger_returns = returns.iloc[lag_ind]

    size = len(lagger_returns.columns)
    for i in range(watch_period, L - hold_period):
        # this part is written based on simple returns
        signal = np.sign(np.mean(leader_returns[leader_returns.columns[i - watch_period:i]].sum(axis=1), axis=0))
        # hold period denotes the number of consecutive days we trade the laggers close to close
        alpha = signal * np.mean(lagger_returns[lagger_returns.columns[ahead + i: ahead + i + hold_period]].sum(axis=1),
                                 axis=0)
        if hedge == 'no':
            result.append(alpha)
        elif hedge == 'mkt':
            alpha2 = alpha - signal * (returns.loc['SPY'][returns.columns[i:i + hold_period]].sum(axis=0))
            result.append(alpha2)
        elif hedge == 'lead':
            alpha2 = alpha - signal * np.mean(leader_returns[leader_returns.columns[i]])
            result.append(alpha2)
        signs.append(int(signal))
        # print(i)
        # print(alpha2)
    return result, signs


def strategy_het_real(df_returns, trading_start, trading_end,
                      lag_matrix, classes, watch_period=1,
                      hold_period=1, class_threshold=None,
                      assumed_max_lag=5, hedge=False,
                      ):
    """
    returns: NxT np array. N is the number of instruments and T is the number of time points

    Returns: the simple returns of the asset at each time step, averaged over different classes

    """

    results_dict = {}
    class_labels = np.unique(classes)
    if not class_threshold:
        class_threshold = int(0.2 * df_returns.shape[0] / len(class_labels))
    class_counts = []
    market_index = df_returns.loc['SPY']
    for c in class_labels:
        # count number of samples in class c
        count = np.count_nonzero(classes == c)
        if count > class_threshold:  # ignore classes with size below a certain threshold
            sub_lag_matrix = lag_matrix[classes == c][:, classes == c]
            lag_vector = lag_mat_to_vec(sub_lag_matrix)
            # pi, r, _ = alignment.SVD_NRS(lag_matrix)
            # lag_vector = np.array(np.round(r), dtype=int)
            lags, counts = np.unique(lag_vector, return_counts=True)
            min_group_size = 0.1 * int(len(sub_lag_matrix) / assumed_max_lag + 1)
            lags = lags[counts >= min_group_size]

            if len(lags) > 1:
                min_lag = np.min(lags)
                lags -= min_lag
                lag_vector -= min_lag
                days_advanced = min(max(lags) + watch_period - 1, trading_start)
                sub_returns = df_returns.iloc[classes == c]
                results = strategy_lag_groups(
                    sub_returns, trading_start, trading_end,
                    days_advanced, lag_vector, lags,
                    watch_period, hold_period,
                    market_index=market_index, hedge=hedge)

                assert len(results['raw returns']) > 0, 'empty results'
                results_dict[f'class {c}'] = results
                class_counts.append(count)

    # average PnL of each group across all classes, if any valid results are produced
    if len(results_dict) > 0:
        results_dict['portfolio average'] = {}
        for key in ['raw returns', 'excess returns']:
            PnL_group_list = [{i: results[key]['PnL'][i] for i in results[key]['PnL'] \
                               if i != 'average'} for group, results in
                              results_dict.items() if group != 'portfolio average']
            PnL = class_average_returns_each_group(PnL_group_list)

            # average return weighted by class size
            pnl_average_list = [results[key]['PnL']['average'] for group, results in
                                results_dict.items() if group != 'portfolio average']
            pnl_average = np.average(pnl_average_list, axis=0, weights=class_counts)
            PnL['average'] = pnl_average
            SR = {group: annualized_sharpe_ratio(returns) for group, returns in PnL.items()}
            results_dict['portfolio average'][key] = {'PnL': PnL,
                                                      'annualized SR': SR}

    return results_dict

def trading_single(df_returns, lag_matrices, estimates, k, sigma,
                   model, trading_period_start, trading_period_end,
                   assumed_max_lag, hedge,  **trading_kwargs):

    # load estimates of lags
    lag_mat = lag_matrices[f'K={k}'][f'sigma={sigma:.2g}'][model]

    if model == 'het':
        classes = estimates[f'K={k}'][f'sigma={sigma:.2g}']['classes']['het']
    else:
        classes = estimates[f'K={k}'][f'sigma={sigma:.2g}']['classes']['spc']
    trading_results = strategy_het_real(
        df_returns,
        trading_period_start, trading_period_end,
        lag_mat, classes, assumed_max_lag=assumed_max_lag,
        hedge=hedge, **trading_kwargs)

    return trading_results
def trading_real_data_multiple(data_path, K_range, sigma_range,
                      train_period_start=0,
                      train_period_end=500,
                      out_of_sample=True,
                      trading_period=50,
                      assumed_max_lag=5,
                      hedge=False,
                      **trading_kwargs):
    """
    run trading strategy on real stocks returns data.

    """
    trading = {f'K={k}': {f'sigma={sigma:.2g}': {} for sigma in sigma_range} for k in K_range}
    with open(f'../results/signal_estimates_real/start{train_period_start}end{train_period_end}.pkl', 'rb') as f:
        estimates = pickle.load(f)
    with open(f'../results/lag_matrices_real/start{train_period_start}end{train_period_end}.pkl', 'rb') as f:
        lag_matrices = pickle.load(f)

    # load returns data
    df_returns = pd.read_csv(data_path, index_col=0)

    # # compute signals and returns based on excess returns (hedged with SPY)
    # if hedge:
    #     df_returns = df_returns - df_returns.loc['SPY']

    # trade on untrained data if out_of_sample is True
    if out_of_sample:
        # returns = df_returns.iloc[:, train_period_end:train_period_end + trading_period]
        trading_period_start = train_period_end
        trading_period_end = train_period_end + trading_period
    else:
        # returns = df_returns.iloc[:, train_period_start:train_period_end]
        trading_period_start = train_period_start
        trading_period_end = train_period_end
    models = ['pairwise', 'sync', 'spc-homo', 'het']
    for k in K_range:
        for sigma in sigma_range:
            for model in models:
                trading[f'K={k}'][f'sigma={sigma:.2g}'][model] = \
                trading_single(df_returns, lag_matrices, estimates,
                               k, sigma, model,
                               trading_period_start, trading_period_end,
                               assumed_max_lag, hedge, **trading_kwargs)

    if out_of_sample:
        file_name = f'start{train_period_start}end{train_period_end}trade{trading_period}'
    else:
        file_name = f'start{train_period_start}end{train_period_end}_insample'
    if hedge:
        file_name = file_name + 'excess'
    with open('../results/PnL_real_excess/' + file_name + '.pkl', 'wb') as f:
        pickle.dump(trading, f)

def trading_real_data(data_path, k, sigma,
                      model,
                      train_period_start=0,
                      train_period_end=50,
                      out_of_sample=True,
                      trading_period=10,
                      assumed_max_lag=2,
                      hedge=True,
                      **trading_kwargs):
    """
    run trading strategy on real stocks returns data.

    """
    # load lag prediction
    with open(f'../results/signal_estimates_real/start{train_period_start}end{train_period_end}.pkl', 'rb') as f:
        estimates = pickle.load(f)
    with open(f'../results/lag_matrices_real/start{train_period_start}end{train_period_end}.pkl', 'rb') as f:
        lag_matrices = pickle.load(f)

    # load returns data
    df_returns = pd.read_csv(data_path, index_col=0)

    # trade on untrained data if out_of_sample is True
    if out_of_sample:
        trading_period_start = train_period_end
        trading_period_end = train_period_end + trading_period
    else:
        trading_period_start = train_period_start
        trading_period_end = train_period_end

    trading = trading_single(df_returns, lag_matrices, estimates,
                       k, sigma, model,
                       trading_period_start, trading_period_end,
                       assumed_max_lag, hedge, **trading_kwargs)
    return trading
    # if out_of_sample:
    #     file_name = f'start{train_period_start}end{train_period_end}trade{trading_period}'
    # else:
    #     file_name = f'start{train_period_start}end{train_period_end}_insample'
    #
    # with open('../results/PnL_real_single/' + file_name + '.pkl', 'wb') as f:
    #     pickle.dump(trading, f)

def best_K_and_sigma(df_returns, K_range, sigma_range,
                     model,
                      train_period_start=0,
                      train_period_end=50,
                    assumed_max_lag=5,
                      criterion='raw returns'):
    trading = {f'K={k}': {f'sigma={sigma:.2g}': {} for sigma in sigma_range} for k in K_range}
    with open(f'../results/signal_estimates_real/start{train_period_start}end{train_period_end}.pkl', 'rb') as f:
        estimates = pickle.load(f)
    with open(f'../results/lag_matrices_real/start{train_period_start}end{train_period_end}.pkl', 'rb') as f:
        lag_matrices = pickle.load(f)

    trading_period_start = train_period_start
    trading_period_end = train_period_end

    if model in ['pairwise', 'sync']:
        sigma_range = [sigma_range[0]]
    score = np.zeros((len(K_range),len(sigma_range)))
    for i, k in enumerate(K_range):
        for j, sigma in enumerate(sigma_range):
            # load estimates of lags
            trading_results = trading_single(df_returns, lag_matrices, estimates,
                           k, sigma, model,
                           trading_period_start, trading_period_end,
                           assumed_max_lag, hedge=True)

            return_type, metric = criterion.split(' ')
            name_extension = {'raw': 'raw returns',
                              'excess': 'excess returns',
                              'returns': 'PnL',
                              'SR': 'annualized SR'}
            return_type = name_extension[return_type]
            metric = name_extension[metric]
            try:
                result = trading_results['portfolio average'][return_type][metric]['average']
                score[i, j] = np.sum(result)
            except:
                print(f'no result at period {trading_period_start} to {trading_period_end} with model {model} at K {k}, sigma {sigma}')
                score[i, j] = -np.Inf
    ind_k, ind_s = np.unravel_index(score.argmax(), score.shape)

    return K_range[ind_k], sigma_range[ind_s]

def best_K_and_sigma_for_all(df_returns,
                             K_range, sigma_range,
                            models,
                          start_indices, signal_length,
                            assumed_max_lag,
                          criterion='raw returns'):

    end_indices = np.array(start_indices) + signal_length
    result = np.zeros((len(start_indices), len(models)), dtype=tuple)
    for i in range(len(start_indices)):
        train_period_start = start_indices[i]
        train_period_end = end_indices[i]
        for j in range(len(models)):
            model = models[j]
            K_sigma = best_K_and_sigma(df_returns,K_range,sigma_range,model,
                                        train_period_start,train_period_end,assumed_max_lag, criterion)
            result[i,j] = K_sigma

    start_end_indices = [(start_indices[t], end_indices[t]) for t in range(len(start_indices))]
    df_results = pd.DataFrame(result, index=start_end_indices, columns=models)
    return df_results

def concat_PnL_real(K, sigma, model,
                    start, end, signal_length,
                    trading_period,
                    return_excess=True,
                    return_SR=True):
    """
    concatenate PnL simulation on out-of-sample data from multiple retrained experiments
    Args:
        train_start:
        train_end:
        retrain_period:
        return_SR:

    Returns: PnL and SR (optional)

    """

    start_indices = range(start, end, retrain_period)

    return_types = ['raw returns']
    if return_excess:
        return_types.append('excess returns')
    PnL_list_dict = {type: [] for type in return_types}

    for train_start in start_indices:
        train_end = train_start + signal_length
        file_name = f'start{train_start}end{train_end}trade{trading_period}excess'
        folder_name = 'PnL_real_excess'
        with open(f'../results/{folder_name}/{file_name}.pkl', 'rb') as f:
            trading = pickle.load(f)
        with open(f'../results/PnL_real/{file_name}.pkl', 'rb') as f:
            trading_test = pickle.load(f)

        for return_type in return_types:
            try:
                pnl = trading[f'K={K}'][f'sigma={sigma:.2g}'][model]['portfolio average'][return_type]['PnL']['average']
                assert len(pnl) == trading_period
            except:
                pnl = np.empty((trading_period))
                pnl[:] = np.nan

            PnL_list_dict[return_type].append(pnl)

    PnL = {return_type: np.concatenate(PnL_list_dict[return_type]) for return_type in return_types}

    if return_SR:
        SR = {type: annualized_sharpe_ratio(returns[~np.isnan(returns)]) for type, returns in PnL.items()}
        return PnL, SR
    else:
        return PnL

def concat_PnL_real2(model,
                    start, end, signal_length,
                    trading_period,
                    return_excess=True,
                    return_SR=True):
    """
    concatenate PnL simulation on out-of-sample data from multiple retrained experiments
    Args:
        train_start:
        train_end:
        retrain_period:
        return_SR:

    Returns: PnL and SR (optional)

    """

    start_indices = range(start, end, retrain_period)

    return_types = ['raw returns']
    if return_excess:
        return_types.append('excess returns')
    PnL_list_dict = {type: [] for type in return_types}

    for train_start in start_indices:
        train_end = train_start + signal_length
        file_name = f'start{train_start}end{train_end}trade{trading_period}'
        folder_name = 'PnL_real_single'
        with open(f'../results/{folder_name}/{file_name}.pkl', 'rb') as f:
            trading = pickle.load(f)

        for return_type in return_types:
            try:
                pnl = trading[model]['portfolio average'][return_type]['PnL']['average']
                assert len(pnl) == trading_period
            except:
                pnl = np.empty((trading_period))
                pnl[:] = np.nan

            PnL_list_dict[return_type].append(pnl)

    PnL = {return_type: np.concatenate(PnL_list_dict[return_type]) for return_type in return_types}

    if return_SR:
        SR = {type: annualized_sharpe_ratio(returns[~np.isnan(returns)]) for type, returns in PnL.items()}
        return PnL, SR
    else:
        return PnL
def string_to_int(string):
    # string format '(x,y)'
    l = string.split(',')
    return int(l[0][1:]), int(l[1][:-1])
if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    # for testing run without parallelization
    data_path = '../../data/pvCLCL_clean.csv'
    K_range = [1, 2, 3]
    sigma_range = np.arange(0.2, 2.1, 0.2)
    start = 5;
    end = 500
    retrain_period = 10
    signal_length = 50
    start_indices = range(start, end, retrain_period)
    models = ['pairwise', 'sync', 'spc-homo', 'het']
    df_returns = pd.read_csv(data_path, index_col=0)
    df_results = best_K_and_sigma_for_all(df_returns,
                             K_range, sigma_range,
                             models,
                             start_indices, signal_length,
                                          assumed_max_lag=2,
                             criterion='raw returns')
    best_K_sigma_path = '../results/PnL_real_single/best_k_sigma.csv'
    # df_results.to_csv(best_K_sigma_path)
    # df_results = pd.read_csv(best_K_sigma_path, index_col=0).applymap(eval)
    # for index in tqdm(df_results.index):
    #     train_period_start, train_period_end = string_to_int(index)
    #     trading_results_models = {}
    #     for model in df_results.columns:
    #         K, sigma = df_results.loc[index, model]
    #
    #         trading = trading_real_data(data_path, K, sigma, model,
    #                           train_period_start=train_period_start,
    #                           train_period_end=train_period_start + signal_length,
    #                           out_of_sample=True,
    #                           trading_period=retrain_period,
    #                           assumed_max_lag=2,
    #                           hedge=True)
    #         trading_results_models[model] = trading
    #     file_name = f'start{train_period_start}end{train_period_end}trade{retrain_period}'
    #     with open('../results/PnL_real_single/' + file_name + '.pkl', 'wb') as f:
    #         pickle.dump(trading_results_models, f)
    #
    # PnL_concat_dict = {}
    # for model in models:
    #
    #     PnL, SR = concat_PnL_real2(
    #         model, start, end,
    #         signal_length, retrain_period,
    #         return_excess=True)
    #
    #     PnL_concat_dict[model] = \
    #         {'PnL': PnL,
    #         'annualized SR': SR}
    #
    # file_name = f'start{start}end{end}_length{signal_length}_trade{retrain_period}'
    # with open('../results/PnL_real_single/' + file_name + '.pkl', 'wb') as f:
    #     pickle.dump(PnL_concat_dict, f)


    # df_returns = pd.read_csv(data_path, index_col=0)
    # k_list = []
    # sigma_list = []
    # for model in models:
    #     k, sigma = best_K_and_sigma(df_returns, K_range, sigma_range,
    #              model,
    #               train_period_start=5,
    #               train_period_end=55,
    #               criterion='raw returns')
    #     k_list.append(k)
    #     sigma_list.append(sigma)
    # print(k_list)
    # print(sigma_list)






"""

n = None
test = False
max_shift = 0.1
assumed_max_lag = 10
models = None
data_path = '../../data/data500_OPCLreturns_init3/'
return_signals = False
round = 1
sigma = 0.1
k = 2
cum_pnl = []
sigma_range = np.arange(0.1, 2.1, 0.5)
return_type = 'simple'

lags = 1

for sigma in sigma_range:
    # read data produced from matlab code base
    observations, shifts, classes_true, X_est, P_est, X_true = main.read_data(
        data_path=data_path + str(round) + '/',
        sigma=sigma,
        max_shift=max_shift,
        k=k,
        n=n
    )

    # calculate clustering and pairwise lag matrix
    classes_spc, classes_est, lag_matrix, ARI_dict = main.clustering(observations=observations,
                                                                     k=k,
                                                                     classes_true=classes_true,
                                                                     assumed_max_lag=assumed_max_lag,
                                                                     X_est=X_est
                                                                     )

    sub_observations = observations[:, shifts < 2]
    sub_lag_matrix = lag_matrix[shifts < 2][:, shifts < 2]
    sub_classes_true = classes_true[shifts < 2]
    #
    results = strategy_het(sub_observations, sub_lag_matrix, sub_classes_true)

    cum_pnl.append(results)
# cmap = {1:'green',-1:'red'}
# colors_mapped = [cmap[c] for c in signs]
# sns.barplot(x=np.arange(len(results)),y=results,palette=colors_mapped)

fig, axes = plt.subplots(len(sigma_range), 1, figsize=(10, 5 * len(sigma_range)))
n = 15
for i in range(len(sigma_range)):
    results = cum_pnl[i]
    cum_returns = np.cumsum(results)
    sns.lineplot(x=np.arange(n), y=cum_returns[:n], ax=axes[i], label=f'sigma = {sigma_range[i]:.1g}')
    axes[i].set_xlabel('day')
    axes[i].set_ylabel('cumulative return')
    axes[i].legend()
    # axes[i].set_title(f'sigma = {sigma_range[i]:.1g}')
plt.show()
"""
