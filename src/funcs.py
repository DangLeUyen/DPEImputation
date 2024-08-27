import numpy as np
from sklearn.preprocessing import StandardScaler

def normalization(X: np.ndarray):
  """
    Rescale the input array X using StandardScaler module of Scikitlearn library

    Args:
        X (numpy.ndarray): The input array to be rescaled.
    Returns:
        numpy.ndarray: The rescaled version of the input array X.
    """
  scaler = StandardScaler()
  scaler.fit(X)
  return scaler.transform(X)

    
def generate_randomly_missing(X , missing_rate):
    """
    Creates a randomly missing mask for the input data.

    Args:
        data (np.ndarray): The input data.
        missing_rate (float): The ratio of missing values to create.

    Returns:
        np.ndarray: An array with the same shape as `data` where missing values are marked as NaN.
    """
    
    non_missing = [0]
    X_copy=np.copy(X)
    
    X_non_missing_col = X_copy[:, non_missing]
    X1_missing = X_copy[:, [i for i in range(X.shape[1]) if i not in non_missing]]

    X_non_missing_row = X1_missing[non_missing]
    X_missing = X1_missing[len(non_missing):(X.shape[0]+1)]
    XmShape = X_missing.shape
    na_id = np.random.randint(0, X_missing.size, round(missing_rate * X_missing.size))
    X_nan = X_missing.flatten()
    X_nan[na_id] = np.nan
    X_nan = X_nan.reshape(XmShape)

    X1_nan = np.vstack((X_non_missing_row, X_nan))
    X_nan = np.hstack((X_non_missing_col, X1_nan))
    
    return X_nan


def solving(a,b,c,d,del_case):
  roots = np.roots([a,b,c,d])
  real_roots = np.real(roots[np.isreal(roots)])
  if len(real_roots)==1:
    return real_roots[0]
  else:
    f = lambda x: abs(x-del_case)
    F=[f(x) for x in real_roots]
    return real_roots[np.argmin(F)]
