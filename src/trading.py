import numpy as np
import pandas as pd
import alignment
import multiprocessing
import time
import scipy.io as spio
import pickle


"""
这是通过为股票进行全局排序从而进行选股择的策略
t：回看窗口
modulo：更新leader的频率
matrix：相关性矩阵所在文件夹的路径的list, at different days
daily_return：每日return的dataFrame
Tlag: roll-over从而控制TVR的系数 （0-1）
leader_prop: 选取leader的百分比（默认20%）
lagger_prop：选取lagger的百分比（默认50%）
neutralize: 选择是否进行neutralization
"""
def cum_returns(returns, return_type):
    """"
    returns: list like, the returns of an asset
    return_type: string denoting the type of returns given by the list 'returns'. 'simple', 'percentage' or 'log'

    return: list like, the cumulative returns of the asset at each time step
    """
    if return_type == 'simple':
        cum_returns = np.cumsum(returns)
    if return_type == 'percentage':
        cum_returns = np.cumprod(returns)
    if return_type == 'log':
        cum_returns = np.cumprod(np.exp(returns))

    return cum_returns

def strategy_het(returns, lag_matrix, classes, shifts,watch_period=1, hold_period=1, leader_prop=0.2, lagger_prop=0.2, rank='plain',
                   hedge='no'):
    """

    Returns: the simple returns of the asset at each time step, averaged over different classes

    """
    results_het = []
    for c in np.unique(classes):
        sub_returns  = returns[:, classes == c]
        if sub_returns.shape[1] > 10: # ignore classes with size below a certain threshold
            sub_lag_matrix = lag_matrix[classes==c][:, classes == c]
            sub_results = strategy_plain(sub_returns, sub_lag_matrix, shifts[classes==c], watch_period,
                                   hold_period, leader_prop, lagger_prop, rank,
                                   hedge)[0]
            results_het.append(sub_results)

    return np.array(results_het).mean(axis=0)

def strategy_plain(returns, lag_matrix, shifts,watch_period=1, hold_period=1, leader_prop=0.2, lagger_prop=0.2, rank='plain',
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
        ranking = np.mean(lag_matrix,axis=1)

    elif rank == 'Synchro':
        ranking = alignment.SVD_NRS(lag_matrix)[0]

    sort_index = np.argsort(ranking)
    lead_ind = sort_index[:int(leader_prop * N)]
    lag_ind = sort_index[-int(lagger_prop * N):]

    # calculate the average lag between the leader and lagger groups
    lag_mat_nan = lag_matrix.copy()
    np.fill_diagonal(lag_mat_nan, np.nan)
    leaders = lag_mat_nan[:,lead_ind]
    laggers = lag_mat_nan[:,lag_ind]
    ahead = np.mean(leaders[~np.isnan(leaders)]) - np.mean(laggers[~np.isnan(laggers)])
    ahead = max(0,round(ahead)-1)
    # 选取leader和lagger

    # 找到leader和lagger的return
    leader_returns = returns.iloc[lead_ind]
    lagger_returns = returns.iloc[lag_ind]

    size = len(lagger_returns.columns)
    for i in range(watch_period, L - hold_period):
        # this part is written based on simple returns
        signal = np.sign(np.mean(leader_returns[leader_returns.columns[i - watch_period:i]].sum(axis=1), axis=0))
        # hold period denotes the number of consecutive days we trade the laggers close to close
        alpha = signal * np.mean(lagger_returns[lagger_returns.columns[ahead + i: ahead + i + hold_period]].sum(axis=1), axis=0)
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

def run_trading(data_path, K_range, sigma_range, max_shift = 0.04, round = 1, **trading_kwargs):
    PnL = {f'K={k}': {f'sigma={sigma:.2g}': {} for sigma in sigma_range} for k in K_range}

    with open(f'../results/signal_estimates/{round}.pkl', 'rb') as f:
        estimates = pickle.load(f)
    with open(f'../results/lag_matrices/{round}.pkl', 'rb') as f:
        lag_matrices = pickle.load(f)

    for k in K_range:
        for sigma in sigma_range:
            observations_path = data_path + '_'.join(['observations',
                                              'noise' + f'{sigma:.2g}',
                                              'shift' + str(max_shift),
                                              'class' + str(k) + '.mat'])
            # load returns
            observations = spio.loadmat(observations_path)['data']
            shifts = spio.loadmat(observations_path)['shifts'].flatten()
            lag_mat_dict = lag_matrices[f'K={k}'][f'sigma={sigma:.2g}']
            classes_spc = estimates[f'K={k}'][f'sigma={sigma:.2g}']['classes']['spc']
            classes_est = estimates[f'K={k}'][f'sigma={sigma:.2g}']['classes']['het']

            PnL_dict = {}
            for model, lag_mat in lag_mat_dict.items():
                if model == 'het':
                    classes = classes_est
                else:
                    classes = classes_spc
                PnL[f'K={k}'][f'sigma={sigma:.2g}'][model] = strategy_het(observations, lag_mat, classes,shifts = shifts, **trading_kwargs)
    with open(f'../results/PnL/{round}.pkl', 'wb') as f:
        pickle.dump(PnL, f)

def run_wrapper(round):
    data_path = '../../data/data500_shift0.04_pvCLCL_init2/' + str(round) + '/'
    K_range = [2,3,4]
    sigma_range = np.arange(0.4, 2.1, 0.5)
    run_trading(data_path=data_path, K_range=K_range,
                sigma_range=sigma_range,round=round,rank = 'plain', lagger_prop=0.5)

if __name__ == '__main__':
    # for testing run without parallelization
    test = False
    if test:
        run_wrapper(round=1)
    else:
        rounds = 4
        inputs = range(1, 1+rounds)
        start = time.time()
        with multiprocessing.Pool() as pool:
            # use the pool to apply the worker function to each input in parallel
            pool.map(run_wrapper, inputs)
            pool.close()
        print(f'time taken to run {rounds} rounds: {time.time() - start}')

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