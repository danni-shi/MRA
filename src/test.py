from matplotlib import pyplot as plt
import utils
from optimization import create_cost_function
import autograd.numpy as np
import random
# import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers

sigma = 1
X0 = np.arange(1,6).astype('float64')
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
with open('test.npy', 'wb') as f:
    np.save(f, sigma)
    np.save(f, X0)
    np.save(f, mean_est)
    np.save(f, P_est)
    np.save(f, B_est)

with open('test.npy', 'rb') as f:
    sigma = float(np.load(f))
    X0 = np.load(f) 
    mean_est = float(np.load(f))
    P_est = np.load(f) 
    B_est = np.load(f)  
    
    
def test1():
    
    L = 5
    manifold = pymanopt.manifolds.Euclidean(L)
    cost, grad, euclidean_hessian = create_cost_function(mean_est, P_est, B_est, sigma, manifold)
    x = np.array([[1,2,3,4,5],
                [0,1,0,1,0],
                [1,1,1,1,1]])
    for i in range(x.shape[0]):
        y = cost(x[i])
        z = grad(x[i])
     


def test2(num=10, num_copies = 500, sigma = 0.1):
    max_shift = 0
    for i in range(num):
        L = random.randint(5,50)
        X = np.random.normal(0,1,L)
        X = (X-np.mean(X))/np.std(X)
        
        observations, shifts = utils.generate_data(X, num_copies,  max_shift, sigma, cyclic = True)
        
        mean_est, P_est, B_est = utils.invariants_from_data(observations)
        manifold = pymanopt.manifolds.Euclidean(L,1)
        cost, grad, euclidean_hessian = create_cost_function(mean_est, P_est, B_est, sigma, manifold)
        X = X.reshape(-1,1)
        print('singal length: ', L)
        print('cost at solution: ', cost(X)/L)
        print('grad norm at solution: ', np.linalg.norm(grad(X),2)/L)
        print('/n')
