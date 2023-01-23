import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt, dates
import random
import time
# import numpy as np
import autograd.numpy as np


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
    """_summary_

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
    
    random.seed(seed)
    max_shift_positions = max(1,int(max_shift * len(x)))
    shifts = np.random.randint(0, max_shift_positions, num_copies)
    data = np.zeros((len(x), num_copies))
    
    for i in range(num_copies):
        k = shifts[i]
        y = np.roll(x,k)
        if not cyclic:
            # y[:k] = np.random.normal(0, 1, size = k)
            y[:k] = np.zeros(k)
        data[:,i] = y
    return data, shifts

def random_noise(x, sigma = 0.1 ,seed = 42):
    """add iid gaussian noise to a signal

    Args:
        x (numpy array): input signal
        sigma (float, optional): standard deviation of the added gaussain white noise. Defaults to 0.1.
        seed (int, optional): random seed. Defaults to 42.

    Returns:
        numpy array: output signal
    """
    random.seed(seed)
    noise = np.random.normal(0, sigma, x.shape)
    y = x + noise
    return y

def generate_data(x, num_copies, max_shift = 0.1, sigma = 0.1, cyclic = True, seed = 42):
    data, shifts = random_shift(x, num_copies, cyclic, max_shift, seed)
    data = random_noise(data, sigma, seed)
    return data, shifts

def power_spectrum(x):
    """return the power spectrum of a signal

    Args:
        x (numpy array): the complexed-valued discrete fourier transform of a single input signal

    Returns:
        numpy array: real-valued power spectrum 
    """
    return abs(x)**2

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
        mat1 = np.array([np.roll(x,k) for k in range(m)])
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

def invariants_from_data(X, sigma = None, debias = False):
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
    
    print('time to estimate invariants from data = ', time.time() - start)
    return mean_est, P_est, B_est

def align_to_ref(X, X_ref):
    """align the vector x after circularly shifting it such that it is optimally aligned with X_ref inn 2-norm

    Args:
        X (np array): vector to align
        X_ref (np array): reference vector 
    """
    X_ref_fft = np.fft.fft(X_ref)
    X_fft = np.fft.fft(X)
    ccf = np.fft.ifft(X_fft.conj() * X_ref_fft).real
    shift = np.argmax(ccf)
    X_aligned = np.roll(X, shift)
    
    return X_aligned