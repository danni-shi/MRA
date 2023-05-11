import numpy as np
import pandas as pd
import main
import matplotlib.pyplot as plt
import seaborn as sns
import alignment

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

# def lag_vec_to_mat(vec):
#     # note this function is invariant to addition and subtraction
#     # of the same value to every element of vec
#     L = len(vec)
#     vec = vec.reshape(-1,1)
#     ones = np.ones((L,1))
#     return vec @ ones.T - ones @ vec.T
#
# def reconcile_score_signs(H,r, G=None):
#     L = H.shape[0]
#     ones = np.ones((L,1))
#
#     # G is the (true) underlying measurement graph
#     if G is None:
#         G = np.ones((L,L)) - np.eye(L)
#
#     const_on_rows = np.outer(r,ones)
#     const_on_cols = np.outer(ones,r)
#     recompH = const_on_rows - const_on_cols
#     recompH = recompH * G
#
#     # difMtx{1,2} have entries in {-1,0,1}
#     difMtx1 = np.sign(recompH) - np.sign(H)
#     difMtx2 = np.sign(recompH) - np.sign(H.T)
#
#     # Compute number of upsets:
#     upset_difMtx_1 = np.sum(abs(difMtx1))/2
#     upset_difMtx_2 = np.sum(abs(difMtx2))/2
#
#     if upset_difMtx_1 > upset_difMtx_2:
#         r = -r
#
#     return r
#
# def SVD_NRS(H, scale_estimator = 'median'):
#     """perform SVD normalised ranking and synchronization on a pairwise score matrix H to obtain a vector of lags
#
#     Args:
#         H (_type_): _description_
#     """
#     L = H.shape[0]
#     ones = np.ones((L,1))
#
#     D_inv_sqrt = np.diag(np.sqrt(abs(H).sum(axis = 1))) # diagonal matrix of sqrt of column sum of abs
#     H_ss = D_inv_sqrt @ H @ D_inv_sqrt
#     U, S, _ = np.linalg.svd(H_ss) # S already sorted in descending order, U are orthonormal basis
#     assert np.all(S[:-1] >= S[1:]), 'Singular values are not sorted in desceding order'
#
#     u1_hat = U[:,0]; u2_hat = U[:,1]
#     u1 = D_inv_sqrt @ ones
#     u1 /= np.linalg.norm(u1) # normalize
#
#     u1_bar = (U[:,:2] @ U[:,:2].T @ u1).flatten()
#     # u1_bar /= np.linalg.norm(u1_bar) # normalize
#     # u2_tilde = u1_hat - np.dot(u1_hat,u1_bar)*u1_bar # same as proposed method
#     # u2_tilde /= np.linalg.norm(u2_tilde) # normalize
#     # test
#     T = np.array([np.dot(u2_hat,u1_bar),-np.dot(u1_hat,u1_bar)])
#     u2_tilde_test = U[:,:2] @ T
#
#     u2_tilde_test /= np.linalg.norm(u1_bar)
#
#     # assert np.linalg.norm(u2_tilde_test.flatten()-u2_tilde) <1e-8 or np.linalg.norm(u2_tilde_test.flatten()+u2_tilde) <1e-8
#     pi = D_inv_sqrt @ u2_tilde_test.reshape(-1,1)
#     pi = alignment.reconcile_score_signs(H, pi)
#     S = alignment.lag_vec_to_mat(pi)
#
#     # median
#     if scale_estimator == 'median':
#
#         offset = np.divide(H, (S+1e-9), out=np.zeros(H.shape, dtype=float), where=np.eye(H.shape[0])==0)
#         tau = np.median(offset[np.where(~np.eye(S.shape[0],dtype=bool))])
#         if tau == 0:
#             tau = np.sum(abs(np.triu(H,k=1)))/np.sum(abs(np.triu(S,k=1)))
#
#     if scale_estimator == 'regression':
#         tau = np.sum(abs(np.triu(H,k=1)))/np.sum(abs(np.triu(S,k=1)))
#
#     r = tau * pi - tau * np.dot(ones.flatten(), pi.flatten()) * ones / L
#     r_test = tau * pi
#     # test
#     r_test = r_test - np.mean(r_test)
#     assert np.linalg.norm(r_test.flatten()-r.flatten()) <1e-8 or np.linalg.norm(r_test.flatten()+r.flatten()) <1e-8
#
#     return pi.flatten(), r.flatten(), tau
#

def strategy_plain(returns, lag_matrix, watch_period=1, hold_period=1, leader_prop=0.2, lagger_prop=0.5, rank='plain',
                   hedge='no'):
    # 创建数据模版
    # df1 = pd.read_csv(matrix[1])
    # df1 = df1.set_index('Unnamed: 0')
    result = []
    sign = 0
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
        b = pd.DataFrame(df.mean())
        b.columns = ['avg']
        # 排序
        b = b.sort_values(by='avg', ascending=False)
        lead = b[0:int(leader_prop * N)]
        lag = b[int(-lagger_prop * N):]
        # 查找leader和lagger
        lead = lead.index
        lag = lag.index

    elif rank == 'Synchro':
        sort_index = np.argsort(alignment.SVD_NRS(lag_matrix)[0])[::-1]
        lead = sort_index[0:int(leader_prop * N)]
        lag = sort_index[int(-lagger_prop * N):]
    stk_list = df.columns
    # 选取leader和lagger

    # 找到leader和lagger的return
    leader_returns = returns.iloc[lead]
    lagger_returns = returns.iloc[lag]

    size = len(lagger_returns.columns)
    for i in range(watch_period, L - hold_period):
        # this part is written based on simple returns
        signal = np.sign(np.mean(leader_returns[leader_returns.columns[i - watch_period:i]].sum(axis=1), axis=0))
        ahead = 0
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
sigma_range = np.arange(0.1, 2.1, 0.2)
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

    # sub_observations = observations[:, (classes_true == 0) * (shifts < 2)]
    # sub_lag_matrix = lag_matrix[(classes_true == 0) * (shifts < 2)][:, (classes_true == 0) * (shifts < 2)]
    sub_observations = observations[:, (classes_true == 0) * (shifts < lags + 1)]
    sub_lag_matrix = lag_matrix[(classes_true == 0) * (shifts < lags + 1)][:, (classes_true == 0) * (shifts < lags + 1)]
    results, signs = strategy_plain(returns=sub_observations, lag_matrix=sub_lag_matrix, lagger_prop=0.2,
                                    watch_period=1, hold_period=2)

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
