from matplotlib import pyplot as plt
import utils
from optimization import create_cost_function
import autograd.numpy as np
# import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers

# np.random.seed(42)
# print(np.random.normal(0,1,3))
# x = np.random.normal(0,1,5)

# # DFT and invariants tested
# xhat = np.fft.fft(x)
# mean, p, bis = utils.invariants_from_data(x)
# print('x = ', x)
# print('x DFT = ', xhat)
# print('mean of x = ', mean)
# print('power spectrum of x = ', p)
# print('bispectrum of x = ', bis)

sigma = 1
mean_est =  0.04164962198820365
P_est = np.array([ 5.69322879, 10.34707154,  7.28589838,  7.28589838, 10.34707154])
B_est = np.array([[ 3.86926394+0.00000000e+00j,  3.75393181-1.35167485e-16j,
        0.76951695+8.18746113e-17j,  0.76951695-8.18746113e-17j,
        3.75393181+1.35167485e-16j],
    [ 3.75393181+1.35167485e-16j,  3.75393181+0.00000000e+00j,
        1.10766601+9.11979295e+00j, -0.38176641+1.35677705e+00j,
        1.10766601+9.11979295e+00j],
    [ 0.76951695-8.18746113e-17j,  1.10766601-9.11979295e+00j,
        0.76951695+0.00000000e+00j, -0.38176641+1.35677705e+00j,
    -0.38176641+1.35677705e+00j],
    [ 0.76951695+8.18746113e-17j, -0.38176641-1.35677705e+00j,
    -0.38176641-1.35677705e+00j,  0.76951695+0.00000000e+00j,
        1.10766601+9.11979295e+00j],
    [ 3.75393181-1.35167485e-16j,  1.10766601-9.11979295e+00j,
    -0.38176641-1.35677705e+00j,  1.10766601-9.11979295e+00j,
        3.75393181+0.00000000e+00j]])
L = 5
manifold = pymanopt.manifolds.Euclidean(L)
cost, euclidean_gradient, euclidean_hessian = create_cost_function(mean_est, P_est, B_est, sigma, manifold)
x = np.array([[1,2,3,4,5],
              [0,1,0,1,0],
              [1,1,1,1,1]])
for i in range(x.shape[0]):
    y = cost(x[i])
    print(y)