import autograd.numpy as np
# import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers

import utils

def optimise_manopt(data, sigma, X0, extra_inits = 0):
    assert isinstance(extra_inits, int)
    L, N = data.shape
    mean_est, P_est, B_est = utils.invariants_from_data(data)
    
    # manifold = pymanopt.manifolds.Euclidean(L,1)
    manifold = pymanopt.manifolds.Euclidean(L,1)
    cost, grad, euclidean_hessian = create_cost_function(mean_est, P_est, B_est, sigma, manifold)
    # problem = pymanopt.Problem(manifold, cost, euclidean_gradient = grad, euclidean_hessian=euclidean_hessian)
    problem = pymanopt.Problem(manifold, cost, riemannian_gradient=grad)
    optimizer = pymanopt.optimizers.TrustRegions(min_gradient_norm = 1e-7, max_iterations = 200, verbosity = 1)
    # optimizer = pymanopt.optimizers.NelderMead()
    if X0.ndim == 1:
        X0 = X0.reshape(-1,1)
    result = optimizer.run(problem, initial_point=X0)
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



def create_cost_function(mean_est, P_est, B_est, sigma, manifold):
    euclidean_gradient = euclidean_hessian = None
    
    # @pymanopt.function.autograd(manifold)
    # def cost(X):
    #     L = len(X)
    #     # print(type(X))
    #     FX = np.fft.fft(X)
    #     A = make_A(L)
        
    #     M1 = np.mean(X)
    #     M2 = abs(FX)**2 + L * sigma**2 * np.ones(L)
    #     # x = np.array(X)[:,i]
    #     mat1 = np.array([np.roll(FX,k) for k in range(L)])
    #     mat2 = np.outer(FX, np.conjugate(FX))
    #     matmul = mat1 * mat2
    #     M3 = matmul + mean_est * (sigma**2) * (L**2) * A
            
    #     # M3 = utils.bispectrum(X.reshape(L,1))[0] + mean_est * (sigma**2) * (L**2) * A
    
    #     M3_min_Best = M3 - B_est
        
    #     # compute coefficients
    #     a1 = L**2
    #     a2 = 1/(L*(2+sigma**2) )
    #     a3 = 1/(L**2*(3+sigma**4))
        
    #     scale = 3 + sigma**4
    #     assert M2.shape == P_est.shape
    #     f = scale * 0.5 * \
    #         (a1*(M1-mean_est)**2 + \
    #             a2*np.linalg.norm(M2 - P_est,2)**2 + \
    #                 a3*np.linalg.norm(M3_min_Best, 'fro')**2 
    #         )
                
    #     return f
    
    @pymanopt.function.autograd(manifold)
    def cost(X):
        if X.ndim == 1:
            X = X.reshape(-1,1)
        L, K = X.shape
        assert K == 1
        # print(type(X))
        FX = np.fft.fft(X, axis = 0)
        A = make_A(L)
        
        # limits
        M1 = np.mean(X)
        M2 = abs(FX)**2 + L * sigma**2 * np.ones((L,K))
        M3 = mean_est * (sigma**2) * (L**2) * A
        # matmul = np.zeros((L,L,K))
        for k in range(K):
            y =FX[:,k]
            mat1 = utils.circulant(y)
            mat2 = np.outer(y, np.conjugate(y))
            M3 += mat1 * mat2
            
        M3_min_Best = M3 - B_est
        
        # compute coefficients
        a1 = L**2
        a2 = 1/(L*(2+sigma**2) )
        a3 = 1/(L**2*(3+sigma**4))
        
        scale = 3 + sigma**4
        # assert M2.shape == P_est.shape
        f = scale * 0.5 * \
            (a1*(M1-mean_est)**2 + \
                a2*np.linalg.norm(M2 - P_est.reshape(-1,1))**2 + \
                    a3*np.linalg.norm(M3_min_Best, 'fro')**2 
            )
                
        return f
    
    
    @pymanopt.function.autograd(manifold)
    def grad(X):
        if X.ndim == 1:
            X = X.reshape(-1,1)
        L, K = X.shape
        assert K == 1
        # print(type(X))
        FX = np.fft.fft(X, axis = 0)
        A = make_A(L)
        
        # limits
        M1 = np.mean(X)
        M2 = abs(FX)**2 + L * sigma**2 * np.ones((L,K))
        M3 = mean_est * (sigma**2) * (L**2) * A
        for k in range(K):
            y =FX[:,k]
            mat1 = utils.circulant(y)
            mat2 = np.outer(y, np.conjugate(y)) 
            M3 += mat1 * mat2
            
        M3_min_Best = M3 - B_est
        
        # compute coefficients
        a1 = L**2
        a2 = 1/(L*(2+sigma**2) )
        a3 = 1/(L**2*(3+sigma**4))
        gradX = (a1/L) * (M1 - mean_est) * np.ones((L,K)) + \
            2 * L * a2 * np.fft.ifft((M2-P_est.reshape(-1,1))*FX, axis = 0) 
        for k in range(K):
            gradX[:,k] += a3 * DBx_adj(FX, M3_min_Best)
        
        scale = 3 + sigma**4
        gradX = scale * gradX.real
        gradX = manifold.euclidean_to_riemannian_gradient(X, gradX)
        # print('used defined grad')
        return gradX
    
    return cost, grad, euclidean_hessian
    # @pymanopt.function.numpy(manifold)
    # def cost(X):
    #     return -np.trace(X.T @ B_est @ X)

    # @pymanopt.function.numpy(manifold)
    # def euclidean_gradient(X):
    #     return -2 * B_est @ X

    # @pymanopt.function.numpy(manifold)
    # def euclidean_hessian(X, H):
    #     return -2 * B_est @ H
    

    #return cost, grad, euclidean_hessian
    

def make_A(L):
    A = np.eye(L, dtype = 'complex_') 
    A[0,:] += np.ones(L)
    A[:,0] += np.ones(L)
    return A
    
    
# matrix = anp.random.normal(size=(dim, dim))
# matrix = 0.5 * (matrix + matrix.T)

# @pymanopt.function.autograd(manifold)
# def cost(point):
#     return -point @ matrix @ point
    
def DBx_adj(y, W):
    y = y.reshape(-1,1)
    L = y.shape[0]   
    H = W * utils.circulant(y.conj())
    z = L * np.fft.ifft(utils.circulantadj(W * (y @ y.conj().T).conj()) + (H + H.conj().T) @ y.reshape(-1,1),axis=0)
    return z.flatten()

