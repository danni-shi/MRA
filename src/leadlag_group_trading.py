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


def strategy_plain(matrix, period=1, leader_prop=0.2, lagger_prop=0.5, rank='plain', hedge='no'):
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
        elif rank == 'Serial':
            b = serial_rank(df)
        elif rank == 'Spring':
            b = pd.DataFrame(SpringRank.SpringRank(df, alpha=0.3), index=df.columns)
            b = b.sort_values(by=0, ascending=True)
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