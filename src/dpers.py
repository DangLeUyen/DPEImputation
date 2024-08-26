import numpy as np
from funcs import solving

def sig_estimate(X, mus0, mus1):
  """
    Estimates the variances and covariance for the given dataset.

    Parameters:
    X (np.ndarray): A 2D numpy array where each column is a data sample [x0, x1].
    mus0 (float): The mean of the first variable (x0).
    mus1 (float): The mean of the second variable (x1).

    Returns:
    tuple: A tuple containing:
        - sig11 (float): The variance of x0.
        - sig22 (float): The variance of x1.
        - sig12 (float): The covariance between x0 and x1.
    """
  # Initialize counters and summation variables
  m = n = l = 0         # Counters for valid, x0-valid and x1-valid cases
  s11 = s12 = s22 = 0   # Sum of squares and product for valid cases
  sig11 = sig22 = 0     # Variances
  del_case = 0          # Sum of products for valid cases
  
  # Iterate over each sample (transposed columns of X)
  for i in X.T:
    x0, x1 = i[0], i[1]
    # Case 1: Both x0 and x1 are finite
    if np.isfinite(x0) and np.isfinite(x1):
      m += 1
      s11 += (x0 - mus0) ** 2
      s22 += (x1 - mus1) ** 2
      s12 += (x0 - mus0) * (x1 - mus1)
      sig11 += (x0 - mus0) ** 2
      sig22 += (x1 - mus1) ** 2
      del_case += (x0 - mus0) * (x1 - mus1)      
    # Case 2: Only x0 is finite
    elif np.isfinite(x0) and np.isnan(x1):
      n += 1
      sig11 += (x0 - mus0) ** 2
    # Case 3: Only x1 is finite
    elif np.isnan(i[0]) and np.isfinite(i[1]): 
      l += 1
      sig22 += (x1 - mus1) ** 2

  # Calculate the variance and covariance
  del_case = max(del_case/(m-1),0) # Ensures non-negative covariance

  sig11 /= (m+n) # Variance of x0
  sig22 /= (m+l) # Variance of x1
  sig12 = solving(-m, s12,(m*sig11*sig22-s22*sig11-s11*sig22), s12*sig11*sig22, del_case)

  return sig11,sig22,sig12

def DPER(X):
  """
    Estimates the covariance matrix for the dataset X, accounting for missing data.

    Parameters:
    X (np.ndarray): A 2D numpy array where rows are observations and columns are variables.

    Returns:
    np.ndarray: The estimated covariance matrix.
    """
  
  # Initialize an empty covariance matrix
  sig = np.zeros((X.shape[1],X.shape[1]))     
  # Estimate the mean for each column, ignoring NaNs
  mu = np.nanmean(X, axis=0)

  # Estimate the covariance matrix
  for a in range(X.shape[1]):
    for b in range(a):
      # Estimate variances and covariance between columns b and a
      temp = sig_estimate(np.array([X[:,b],X[:,a]]), mu[b], mu[a])
      # Assign the estimated variances and covariance to the covariance matrix
      sig[b][b] = temp[0]
      sig[a][a] = temp[1]
      sig[b][a] = sig[a][b]=temp[2]

  return sig