from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt

def fn_gp_smooth(X_train,y_train, sigma):
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=sigma**2, n_restarts_optimizer=2
    )
    gp.fit(X_train, y_train)
    
    return gp.predict(X_train)

k = 2
sigma = 0.5
data_path = '../data_n=500/'
max_shift= 0.1
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

x = np.arange(len(observations))
X_train = np.array([x]).T
num_cand = 5
observations_smoothed = np.array([fn_gp_smooth(X_train,observations[:,i],sigma) for i in range(num_cand)]).T

#plot all the candidate signals along with their class labels (depicted as colors)
fig0, ax = plt.subplots(num_cand,1,sharex=True, figsize=(20,3*num_cand))
for i in range(num_cand):
    ax[i].plot(x,observations[:,i],label='Original')
    ax[i].plot(x,observations_smoothed[:,i],color=[.5,.5,.5],linestyle='--',label='GP smoothed')
    ax[i].plot(x,np.roll(X_true[:,classes_true[i]],shifts[i]),linestyle = ':', label = 'True ref signal')
    ax[i].set_ylabel("c"+str(i)+"(t)")
    ax[i].legend()
ax[i].set_xlabel("t")   
plt.savefig('smoothed_observation.png')