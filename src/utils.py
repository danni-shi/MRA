import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt, dates
import random
import time
import numpy as np
import os
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import jacobian
from scipy.optimize import linear_sum_assignment

def get_dataset(type, ticker, date, nlevels, start = '34200000', end = '57600000'):
    """ Return LOBSTER intra-day dataset based on the default naming style

    Args:
        type (string): 'message' or 'orderbook 
        ticker (string): ticker of the stock    
        date (string): date of data in format'yyyy-mm-dd'
        nlevels (string): number of levels of the LOB data
        start (string): start time of data in seconds after midnight
        end (string): end time of data in seconds after midnight
        

    Returns:
        dataframe: LOBSTER data
    """
    assert (type == 'orderbook' or type == 'message' )
    
    message = '_'.join((ticker, date, start, end, 'message', nlevels))   
    msg_col = ['Time', 'EventType', 'OrderID', 'Size','Price','Direction']
    df_message = pd.read_csv('../data/' + str(ticker) + '/' + message +'.csv', names = msg_col)
    
    if type == 'orderbook':
        orderbook = '_'.join((ticker, date, start, end, 'orderbook', nlevels))
        ob_col = [item for sublist in [['AskPrice' + str(i), 'AskSize' + str(i), 'BidPrice' + str(i), 'BidSize' +str(i)] 
                            for i in range(1, int(nlevels) + 1)] for item in sublist]
        df_orderbook = pd.read_csv('../data/' + str(ticker) + '/' + orderbook + '.csv', names = ob_col)
        df_orderbook['Time'] = df_message['Time'].copy()
        return df_orderbook
    else:
        return df_message
    
def midprice(df):
    """
    Args:
        df (dataframe): orderbook data set

    Returns:
        dataframe: mid-price at each message time
    """
    mid_price = pd.Series(df[['AskPrice1', 'BidPrice1']].mean(axis = 1), name = 'MidPrice')
    time = df.Time
    return pd.concat([time, mid_price], axis = 1)

def downsample(df, freq = 60):
    """downsample the data set to a given time interval. The lateset value in the interval is taken as the value at the interval's right edge.

    Args:
        df (dataframe): message or orderbook data whose 'Time' column is written as number of seconds after midnight
        freq (int, optional): interval length in number of seconds. Defaults to 60, i.e. 1 min.

    Returns:
        dataframe: downsampled data set at regular intervals
    """
    bins = np.arange(34200, 57600 + 0.5*freq, freq)
    
    return df.groupby(pd.cut(df['Time'], bins)).max().set_index(bins[1:len(bins)])

def seconds_to_time(second, date = '2012-06-21'):
    """convert time value from number of seconds after midnight to datetime format, up to the accuracy of 1 microsecond

    Args:
        second (float): number of seconds after midnight, up to the accuracy of 1e-9
        date (str, optional): date. Defaults to '2012-06-21'.

    Returns:
        datetime: time value in datetime format
    """
    td = str(dt.timedelta(seconds = int(second), microseconds = 1e6 * (second - int(second))))
    # when the microsecond component is 0, reformat the string
    if td[-7] != '.':
        td = td + '.000000'
    time = dt.datetime.strptime("{} {}".format(date, td), "%Y-%m-%d %H:%M:%S.%f")
    return time

def random_shift(x, num_copies, cyclic = True, max_shift = 0.1, seed = 42):
    """ create shifted copies of given signal

    Args:
        x (numpy array): input signal
        num_copies (int): number of randomly shifted copies to be generated
        cyclic (bool, optional): shift type, if 'False', shift is parallel. Defaults to True.
        max_shift (float, optional): Maximum extend of shift as a proportion of the length of the input signal . Defaults to 0.1.
        seed (int, optional): random seed. Defaults to 42.

    Returns:
        numpy array: copies of shifted signals of size (len(x) by num_copies). 
        numpy array: shifts of each observation 
    """
    
    max_shift_positions = max(1,int(max_shift * len(x)))
    shifts = np.random.randint(0, max_shift_positions + 1, num_copies)
    data = np.zeros((len(x), num_copies))
    
    for i in range(num_copies):
        k = shifts[i]
        y = np.roll(x,k)
        if not cyclic:
            # y[:k] = np.random.normal(0, 1, size = k)
            y[:k] = np.zeros(k)
        data[:,i] = y
    return data, shifts

def random_shift_het(X, num_copies, cyclic = True, max_shift = 0.1, seed = 42):
    L,K = X.shape
    assert K == len(num_copies)
    M = int(sum(num_copies))
    cumsum = np.append([0],np.cumsum(num_copies)).astype(int)
    data = np.zeros((L,M))
    shifts = np.zeros(M)
    classes = np.zeros(M)
    for k in range(K):
        x = X[:,k]
        m = num_copies[k]
        # fill the zeros array from left to right
        r = range(cumsum[k],cumsum[k+1])
        data[:,r], shifts[r] = random_shift(x,m,cyclic,max_shift,seed)
        classes[r] = k
    # permute the data
    perm = np.random.permutation(M)
    data = data[:,perm]
    shifts = shifts[perm]
    classes = classes[perm]
    
    return data, shifts, classes
    
def random_noise(x, sigma = 0.1 ,seed = 42):
    """add iid gaussian noise to a signal

    Args:
        x (numpy array): input signal
        sigma (float, optional): standard deviation of the added gaussain white noise. Defaults to 0.1.
        seed (int, optional): random seed. Defaults to 42.

    Returns:
        numpy array: output signal
    """
    # random.seed(seed)
    noise = np.random.normal(0, sigma, x.shape)
    y = x + noise
    return y

def generate_data(x, num_copies, max_shift = 0.1, sigma = 0.1, cyclic = True, seed = 42):
    data, shifts = random_shift(x, num_copies, cyclic, max_shift, seed)
    data = random_noise(data, sigma, seed)
    return data, shifts

def generate_data_het(x, num_copies, max_shift = 0.1, sigma = 0.1, cyclic = True, seed = 42):
    data, shifts, classes = random_shift_het(x, num_copies, cyclic, max_shift, seed)
    data = random_noise(data, sigma, seed)
    return data, shifts, classes

def power_spectrum(x):
    """return the power spectrum of a signal

    Args:
        x (numpy array): the complexed-valued discrete fourier transform of a single input signal

    Returns:
        numpy array: real-valued power spectrum 
    """
    return abs(x)**2

def circulant(X):
    X = X.flatten()
    L = len(X)
    mat = np.array([np.roll(X,k) for k in range(L)])
    return mat

def circulantadj(mat):
    N,M = mat.shape
    assert N == M
    A = np.zeros((N, N**2))
    I = np.identity(N)
    for n in range(N):
        en =I[:,n]
        Cn = circulant(en)
        A[n,:] = Cn.flatten('F') # flatten in column-major
    return A @ mat.flatten('F').reshape(-1,1)
  
def bispectrum(X):
    """return the bispectrum of a signal

    Args:
        X (m by n array): each column of the df is the discrete fourier transform of a single input signal

    Returns:
        ndarray: n x m x m array. The ith m x m matrix corresponds to the bispectrum of the ith signal
    """
    
    if X.ndim == 1:
        X = X.reshape(-1,1)
    
    assert X.ndim == 2, "dimension of input signal is wrong"
    # m: length of signal; n: number of copies
    m,n = X.shape
    if n != 1:
        output = np.zeros((n,m,m),dtype = 'complex_')
    for i in range(n):
        x = np.array(X)[:,i]
        mat1 = circulant(x)
        mat2 = np.outer(x, np.conjugate(x))
        matmul = mat1 * mat2
        if n == 1 :
            return matmul
        else:
            output[i] = (mat1 * mat2)
    return output

def simulate_data(signal, n_copies = 800, cyclic = False, sigma = 0.1):
    """simulate randomly shifted and noisy copies of an input signal, calculate the mean, power spectrum of biscpectrum of each of the copies

    Args:
        signal (series): a real-valued financial time series
        n_copies (int, optional): number of randomly shifted and noisy copies of the signal. Defaults to 800.
        cyclic (bool, optional): whether the shifts are cyclic. Defaults to False.
        sigma (float, optional): standard deviation of the gaussian white noise. Defaults to 0.1.
        
    Returns: 
        dict: { 'original': the standardised input signal,
                'shifted': shifted copies of the standard signal,
                'shifted+noise': noisy copies of the shifted series
                'DFT': discrete fourier transforms of the shifted, noisy copies
                'mean': mean of each copy,
                'power spec': power spectrum of each copy,
                'bispec': bispectrum of each copy}
    """
    signal_dict = {}
    # standardised signals
    standard_signal = (signal - signal.mean())/ signal.std()
    signal_dict['original'] = standard_signal
    # create shifted, noisy version of the signal
    signal_dict['shifted'] = random_shift(signal_dict['original'], n_copies, cyclic = cyclic)
    signal_dict['shifted+noise'] = random_noise(signal_dict['shifted'], sigma)
    signal_dict['DFT'] = pd.DataFrame(np.fft.fft(signal_dict['shifted+noise'], axis = 0), 
                                     columns = signal_dict['shifted+noise'].columns)
    signal_dict['mean'] = signal_dict['DFT'].iloc[0,:]/len(signal_dict['DFT'])
    signal_dict['power spec'] = signal_dict['DFT'].apply(power_spectrum, axis = 0)
    signal_dict['bispec'] = bispectrum(signal_dict['DFT'])
    
    return signal_dict

def invariants_from_data(X, sigma = None, debias = False, verbose = False):
    """estimates the invariant features from data by averaging the features over all observations

    Args:
        X (numpy array): L x N, each column contains an observation
        debias(bool): whether the estimates are debiased
        
    Returns:
        mean_est(float): estimate of the mean of the signal
        
    """
    start = time.time()
    if X.ndim == 1:
        X = X.reshape(-1,1)
        
    L, N = X.shape
    
    mean_est = X.mean().mean()
    
    if debias:
        X = X - mean_est
        
    X_fft = np.fft.fft(X, axis = 0)
    
    P = np.apply_along_axis(power_spectrum, 0, X_fft)
    P_est = np.mean(P, axis = 1)
    
    if debias:
        if sigma == None:
            sigma = np.std(X.sum(axis =2))/np.sqrt(L)
        P_est = max(0, P_est - L*sigma**2)
    
    B_est = np.mean(bispectrum(X_fft), axis = 0)
    
    if verbose:
        print('time to estimate invariants from data = ', time.time() - start)
    return mean_est, P_est, B_est

def invariants_from_data_het(X, w = None, sigma = None, debias = False, verbose = False):
    """estimates the invariant features from data by averaging the features over all observations, when number of classes K >= 1

    Args:
        X (numpy array): L x N, each column contains an observation
        debias(bool): whether the estimates are debiased
        
    Returns:
        mean_est(float): estimate of the mean of the signal
        
    """
    
    start = time.time()
    if X.ndim == 1:
        X = X.reshape(-1,1)
        
    L, N = X.shape
    if  w is None:
        w = np.ones(N)/N
        
    mean_est = X.mean().mean()
    
    if debias:
        X = X - mean_est
        
    X_fft = np.fft.fft(X, axis = 0)
    
    P = np.apply_along_axis(power_spectrum, 0, X_fft)
    P_est = np.mean(P, axis = 1)
    
    if debias:
        if sigma == None:
            sigma = np.std(X.sum(axis =2))/np.sqrt(L)
        P_est = max(0, P_est - L*sigma**2)
    
    B_est = np.mean(bispectrum(X_fft), axis = 0)
    
    if verbose:
        print('time to estimate invariants from data = ', time.time() - start)
    return mean_est, P_est, B_est

def align_to_ref(X, X_ref, return_ccf = False):
    """align the vector x after circularly shifting it such that it is optimally aligned with X_ref in 2-norm

    Args:
        X (np array): vector to align
        X_ref (np array): reference vector 
    """
    X_ref = X_ref.flatten()
    X = X.flatten()
    
    X_ref_fft = np.fft.fft(X_ref)
    X_fft = np.fft.fft(X)
    ccf = np.fft.ifft(X_ref_fft.conj() * X_fft).real
    lag = np.argmax(ccf)
    X_aligned = np.roll(X, -lag)
    
    if return_ccf:
        return X_aligned, lag, ccf/np.linalg.norm(X_ref)/np.linalg.norm(X)
    else:
        return X_aligned, lag

def align_to_ref_het(X, X_ref):
    """permuted and circularly shifted (individually) to match xref as closely as possible (in some sense defined by the code.

    Args:
        X (np array): L x K. Each column contains a signal recovered from data
        X_ref (np array): L x K. Each column contains a true signal
    """
    assert X.shape == X_ref.shape
    
    L, K =  X.shape
    dist_mat = np.zeros((K,K))
    for k1 in range(K):
        for k2 in range(K):
            dist_mat[k1,k2] = np.linalg.norm(align_to_ref(X[:,k2], X_ref[:,k1])[0]- X_ref[:,k1])**2
    row_ind, col_ind = linear_sum_assignment(dist_mat)
    X_aligned = X[:, col_ind]
    for k in range(K):
        X_aligned[:,k] = align_to_ref(X_aligned[:,k], X_ref[:,k])[0]
        
    return X_aligned, col_ind

def assign_classes(observation, X_est):
    dist = []
    for k in range(X_est.shape[1]):
        # dist.append(np.linalg.norm(utils.align_to_ref(observation, X_est[:,k])[0]- X_est[:,k])**2)
        # dist.append(np.corrcoef(utils.align_to_ref(observation, X_est[:,k])[0], X_est[:,k])[0,1])
        _, lag, corrcoef = align_to_ref(observation, X_est[:,k], True)
        dist.append(corrcoef[lag])
    return np.argmax(dist)

def alignment_residual(x1, x2):
    """align the vector x1 after circularly shifting it such that it is optimally aligned with x2 in 2-norm. Calculate the 

    Args:
        x1 (np array): 
        x2 (np array): 

    Returns:
        relative_residual (float): normalized residual between the aligned vector and x2.
        lag (int): lag of best alignment
    """
    # align x1 to x2
    x1_aligned, lag = align_to_ref(x1,x2)
    relative_residual = np.linalg.norm(x1_aligned-x2)/np.linalg.norm(x1_aligned)/np.linalg.norm(x2)
    
    return relative_residual, lag

def residual_lag_mat(observations):
    L, N = observations.shape
    residuals = np.zeros((N,N))
    lags = np.zeros((N,N))
    for j in range(N):
        for i in range(j):
            res, lag = alignment_residual(observations[:,i], observations[:,j])
            residuals[i,j] = residuals[j,i] = res
            lags[i,j] = lag
            lags[j,i] = -lag
    
    return residuals, lags

def save_to_folder(directory, folder_name):
    # given a dictionary of errors: error_dict[str_Learner_object_name] = prediction_loss_of_that_object
    # save such files in the \recorded_results folder

    now = dt.datetime.now()
    date_string = now.strftime("%Y-%m-%d-%Hh%Mmin%Ss")  # use formatted datestring to name directory where results are saved

    print('Recording results in directory: ' + directory, date_string + '_' + folder_name)
    str_write_dir = os.path.join(directory + '/', date_string + '_' + folder_name)
    os.makedirs(str_write_dir)
    
    return str_write_dir


def hess_from_grad(grad):
    """return the hessian function as the jacobian of a gradient function from autograd. 
    
    Args:
        grad (function): (np array) -> (np array). Gradient of a cost function at a given point x

    Returns: 
        function: returns the directional derivative of gradient (at point x) in the direction of a tangent vector (y).
    """
    def hess(x,y):
        assert x.shape == y.shape
        H_f = jacobian(grad)
        H = H_f(x)
        return H @ y
    return hess