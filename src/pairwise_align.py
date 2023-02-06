import numpy as np
import matplotlib.pyplot as plt
import utils
import seaborn as sns
import optimization


def pairwise_align(observations):
    """Calculate pairwise cross-correlations and find the best lag

    Args:
        observations (np array): observations

    Returns:
        np array: the i,j th element represents the lag k such that corr(Xi(t), Xj(t+k)) is maximum. i.e. shifting Xi rightwards by k timescales leads to the best alignment.
    """
    L, M = observations.shape
    lag_matrix = np.eye(M)
    for m in range(M):
        for i in range(1,m):
            _, lag = utils.align_to_ref(observations[:,i], observations[:,m])    
            if lag > L//2 + 1:
                lag -= L
            lag_matrix[i,m] = lag
    return lag_matrix

with open('../results/data.npy', 'rb') as f:
    observations = np.load(f)
    shifts = np.load(f)
    
lag_matrix = pairwise_align(observations)

fig, ax = plt.subplots(figsize = (15,6))
ax = sns.heatmap(lag_matrix[:100,:100])
plt.show()