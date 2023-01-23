import autograd.numpy as np
# import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers

import utils

def make_A(L):
    A = np.eye(L, dtype = 'complex_') 
    A[0,:] += np.ones(L)
    A[:,0] += np.ones(L)
    return A

def create_cost_function(mean_est, P_est, B_est, sigma, manifold):
    euclidean_gradient = euclidean_hessian = None
    
    @pymanopt.function.autograd(manifold)
    def cost(X):
        L = len(X)
        # print(type(X))
        FX = np.fft.fft(X)
        A = make_A(L)
        
        M1 = np.mean(X)
        M2 = abs(FX)**2 + L * sigma**2 * np.ones(L)
        # x = np.array(X)[:,i]
        mat1 = np.array([np.roll(FX,k) for k in range(L)])
        mat2 = np.outer(FX, np.conjugate(FX))
        matmul = mat1 * mat2
        M3 = matmul + mean_est * (sigma**2) * (L**2) * A
            
        # M3 = utils.bispectrum(X.reshape(L,1))[0] + mean_est * (sigma**2) * (L**2) * A
    
        M3_min_Best = M3 - B_est
        
        # compute coefficients
        a1 = L**2
        a2 = 1/(L*(2+sigma**2) )
        a3 = 1/(L**2*(3+sigma**4))
        
        scale = 3 + sigma**4
        assert M2.shape == P_est.shape
        f = scale * 0.5 * \
            (a1*(M1-mean_est)**2 + \
                a2*np.linalg.norm(M2 - P_est,2)**2 + \
                    a3*np.linalg.norm(M3_min_Best, 'fro')**2 
            )
                
        return f
    return cost, euclidean_gradient, euclidean_hessian
    
    
    
# matrix = anp.random.normal(size=(dim, dim))
# matrix = 0.5 * (matrix + matrix.T)

# @pymanopt.function.autograd(manifold)
# def cost(point):
#     return -point @ matrix @ point
    
    
    
def optimise_manopt(data, sigma, X0, extra_inits = 0):
    assert isinstance(extra_inits, int)
    L, N = data.shape
    mean_est, P_est, B_est = utils.invariants_from_data(data)
    
    manifold = pymanopt.manifolds.Euclidean(L)
    cost, euclidean_gradient, euclidean_hessian = create_cost_function(mean_est, P_est, B_est, sigma, manifold)
    # print('EG: ', euclidean_gradient, ' EH: ', euclidean_hessian)
    # print(type(cost))
    problem = pymanopt.Problem(manifold, cost)
    optimizer = pymanopt.optimizers.TrustRegions(min_gradient_norm = 1e-6, max_iterations = 200)
    result = optimizer.run(problem, initial_point=X0)
    # optimizer = pymanopt.optimizers.SteepestDescent()
    # result = optimizer.run(problem)
    X_est = result.point
    result_cost = result.cost
    
    if extra_inits > 0:
        for i in range(extra_inits):
            result = optimizer.run(problem)
            if result.cost < result_cost:
                result_cost = result.cost
                X_est = result.point
            
    return X_est