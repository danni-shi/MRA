import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import seaborn as sns 
import pickle 

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#----- Check these parameters are in sync with main.py -----#
sigma_range = np.arange(0.1,2.1,0.1) # std of random gaussian noise
K_range = [2,3,4]
# sigma_range = [1.1]
# K_range = [2]
with open('../results/performance.pkl', 'rb') as f:   
        performance = pickle.load(f)

with open('../results/estimates.pkl', 'rb') as f:   
    estimates = pickle.load(f)
    

#======== plot the results ==================

labels = {'pairwise': 'SPC',
          'sync': 'SPC-Sync',
            'spc': 'SPC-IVF',
            'het': 'IVF',
            'het reassigned': 'IVF (regroup)',
            'true': 'True'}
color_labels = labels.keys()
col_values = sns.color_palette('Set2')
color_map = dict(zip(color_labels, col_values))

lty_map = {'sync': 'dotted',
            'spc': 'dashdot',
            'het': 'dashed',
            'true': 'solid'}

#--- plots of performance of different methods for lag estimatiopn ---#
results_save_dir = utils.save_to_folder('../plots/SPC_cluster', '')
for k in K_range: 
    fig, axes = plt.subplots(3,1, figsize = (15,18), squeeze=False)
    ax = axes.flatten()
    plot_list = ['ARI', 'error', 'accuracy']
    
    for i in range(len(plot_list)):
        for key, result_list in performance[f'K={k}'][plot_list[i]].items():
            ax[i].plot(sigma_range, result_list, label = labels[key], color = color_map[key])
        ax[i].grid()
        ax[i].legend()
        ax[i].set_xlabel('std of added noise')
    ax[0].set_title(f'Ajusted Rand Index of clustering against noise level, K = {k}')
    ax[1].set_title(f'Change of Alignment Error with Noise Level')
    ax[2].set_title(f'Change of Alignment Accuracy with Noise Level')
    plt.savefig(results_save_dir + f'/acc_err_ARI_K={k}')

#--- plots of signal estimates of different methods ---#
results_save_dir_2 = results_save_dir + '/signal_estimates'
os.makedirs(results_save_dir_2)

# align the estimated signals to ground truth signals
for k in K_range: 
    os.makedirs(results_save_dir_2 + f'/K={k}')
    i = 0
    for sigma in sigma_range:
        # # calculate the estimated mixing probabilities

        # for key, classes in classes_estimates[f'K={k}'][f'sigma={sigma:.2g}'].items():
        #     p_est[key] = np.zeros(k)
        #     for c in classes.unique():
        #         p_est[key][c] = np.mean(classes==c)
        estimates_i = estimates[f'K={k}'][f'sigma={sigma:.2g}']
        fig, ax = plt.subplots(k, 1, figsize = (10,5*k))
        X_true =estimates_i['signals']['true']
        # for key, X_estimates in estimates_i['signals'].items():
        #     if key != 'true':
        #         X_estimates, perm = utils.align_to_ref_het(X_estimates, X_true)
        #         signal_estimates[f'K={k}'][f'sigma={sigma:.2g}'][key] = X_estimates
        #         rel_errors = {}
        #         rel_errors_str = ''
        #         p_est_str = ''
        for j in range(k):
            rel_errors_str = []
            p_est_str = []
            for key, X_estimates in estimates_i['signals'].items():
                ax[j].plot(X_estimates[:,j], \
                            label = labels[key], \
                            color = color_map[key], \
                            linestyle = lty_map[key])
                if key != 'true':
                    rel_err = np.linalg.norm(X_estimates[:,j]-X_true[:,j])/np.linalg.norm(X_true[:,j])
                    rel_errors_str.append( f'{labels[key]} {rel_err:.3f}')
                    
            for key_p, value_p in estimates_i['probabilities'].items():
                p_est_str.append(f'{labels[key_p]} {value_p[j]:.3g}')
                
            title = 'rel. err.: ' + ';'.join(rel_errors_str) + '\nmix. prob.: ' + '; '.join(p_est_str)
            ax[j].set_title(title)
            ax[j].grid()
            ax[j].legend()

        fig.suptitle(f'Comparison of the True and Estimated Signals, K = {k}, noise = {sigma:.2g}')
        plt.savefig(results_save_dir_2 + f'/K={k}/{int(i)}')
        i+=1
    
    