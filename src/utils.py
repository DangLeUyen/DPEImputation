import numpy as np
import pandas as pd
import time
import math

from numpy.linalg import norm, inv, det
from fancyimpute import KNN, SoftImpute
from missforest.missforest import MissForest
from sklearn.impute import IterativeImputer, SimpleImputer

from sklearn import datasets
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer

from sklearn.model_selection import train_test_split
from scipy import stats

import math
from typing import *

import numpy as np


def rescale(X: np.ndarray):
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

    
def generate_randomly_missing(data: np.ndarray, missing_rate: float):
    """
    Creates a randomly missing mask for the input data.

    Args:
        data (np.ndarray): The input data.
        missing_rate (float): The ratio of missing values to create.

    Returns:
        np.ndarray: An array with the same shape as `data` where missing values are marked as NaN.
    """
    non_missing = [0]
    data_copy=np.copy(data)

    data_non_missing_col = data_copy[:, non_missing]
    data1_missing = data_copy[:, [i for i in range(data.shape[1]) if i not in non_missing]]

    data_non_missing_row = data1_missing[non_missing]
    data_missing = data1_missing[len(non_missing):(data.shape[0]+1)]

    datamShape = data_missing.shape
    na_id = np.random.randint(0, data_missing.size, round(missing_rate * data_missing.size))
    data_nan = data_missing.flatten()
    data_nan[na_id] = np.nan
    data_nan = data_nan.reshape(datamShape)

    data1_nan = np.vstack((data_non_missing_row, data_nan))
    data_nan = np.hstack((data_non_missing_col, data1_nan))
    return data_nan

def solving(a,b,c,d,del_case):
  roots = np.roots([a,b,c,d])
  real_roots = np.real(roots[np.isreal(roots)])
  if len(real_roots)==1:
    return real_roots[0]
  else:
    f = lambda x: abs(x-del_case)
    F=[f(x) for x in real_roots]
    return real_roots[np.argmin(F)]

def threshold_regularize(S):
    p = S.shape[1]
    alpha = 0
    threshold = 0.01 - 0.0099*np.exp(-0.02*(p-3))
    while det(S + alpha*np.eye(p)) < threshold:
        alpha += 0.01
    return S + alpha*np.eye(p)
   
