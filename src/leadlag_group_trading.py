import numpy as np
import pandas as pd


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

def lag_vec_to_mat(vec):
    # note this function is invariant to addition and subtraction
    # of the same value to every element of vec
    L = len(vec)
    vec = vec.reshape(-1,1)
    ones = np.ones((L,1))
    return vec @ ones.T - ones @ vec.T

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
    # u1_bar /= np.linalg.norm(u1_bar) # normalize
    # u2_tilde = u1_hat - np.dot(u1_hat,u1_bar)*u1_bar # same as proposed method
    # u2_tilde /= np.linalg.norm(u2_tilde) # normalize
    # test
    T = np.array([np.dot(u2_hat,u1_bar),-np.dot(u1_hat,u1_bar)])
    u2_tilde_test = U[:,:2] @ T

    u2_tilde_test /= np.linalg.norm(u1_bar)

    # assert np.linalg.norm(u2_tilde_test.flatten()-u2_tilde) <1e-8 or np.linalg.norm(u2_tilde_test.flatten()+u2_tilde) <1e-8
    pi = D_inv_sqrt @ u2_tilde_test.reshape(-1,1)
    pi = reconcile_score_signs(H, pi)
    S = lag_vec_to_mat(pi)

    # median
    if scale_estimator == 'median':

        offset = np.divide(H, (S+1e-9), out=np.zeros(H.shape, dtype=float), where=np.eye(H.shape[0])==0)
        tau = np.median(offset[np.where(~np.eye(S.shape[0],dtype=bool))])
        if tau == 0:
            tau = np.sum(abs(np.triu(H,k=1)))/np.sum(abs(np.triu(S,k=1)))

    if scale_estimator == 'regression':
        tau = np.sum(abs(np.triu(H,k=1)))/np.sum(abs(np.triu(S,k=1)))

    r = tau * pi - tau * np.dot(ones.flatten(), pi.flatten()) * ones / L
    r_test = tau * pi
    # test
    r_test = r_test - np.mean(r_test)
    assert np.linalg.norm(r_test.flatten()-r.flatten()) <1e-8 or np.linalg.norm(r_test.flatten()+r.flatten()) <1e-8

    return pi.flatten(), r.flatten(), tau

def strategy_plain(lag_matrix, period=1, leader_prop=0.2, lagger_prop=0.5, rank='plain', hedge='no'):
    # 创建数据模版
    df1 = pd.read_csv(matrix[1])
    df1 = df1.set_index('Unnamed: 0')
    result = []
    sign = 0
    signs = []
    for i in range(31, len(matrix) - period, period):

        df = pd.read_csv(matrix[i]) # NxN matrix where each element is a pairwise lag
        df = df.set_index('Unnamed: 0')
        df.columns = df1.columns
        df.index = df1.index

        date = daily_return.columns[i]

        length = len(df.columns)
        if rank == 'plain':
            b = pd.DataFrame(df.mean())
            b.columns = ['avg']
            # 排序
            b = b.sort_values(by='avg', ascending=False)

        elif rank == 'Synchro':
            b = synchro(df)
        stk_list = df.columns
        # 选取leader和lagger
        lead = b[0:int(leader_prop * length)]
        lag = b[int(-lagger_prop * length):]
        # 查找leader和lagger
        lead = lead.index
        lag = lag.index
        # 找到leader和lagger的return
        leader = daily_return.loc[lead]
        lagger = daily_return.loc[lag]

        size = len(lagger.columns)

        signal = np.sign(np.mean(leader[leader.columns[i - period:i]].sum(axis=1), axis=0))

        alpha = signal * np.mean(lagger[lagger.columns[i:i + period]].sum(axis=1), axis=0)
        if hedge == 'no':
            result.append(alpha)
        elif hedge == 'mkt':
            alpha2 = alpha - signal * (daily_return.loc['SPY'][daily_return.columns[i:i + period]].sum(axis=0))
            result.append(alpha2)
        elif hedge == 'lead':
            alpha2 = alpha - signal * np.mean(leader[leader.columns[i]])
            result.append(alpha2)
        signs.append(signal)
        print(i)
        print(alpha2)
    return result