from dpers import DPERS
import numpy as np

def _cov(X, n_jobs=None):
    """Estimate covariance using DPER algorithm.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    Returns
    -------
    S : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.
    """
    S = DPERS().fit(X, n_jobs=n_jobs)
    return S

def _shinkage_cov_estimator(S):
    """
    Returns
    -------
    s : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.
    """
    p = S.shape[1]
    alpha = 0
    threshold = 0.01 - 0.0099*np.exp(-0.02*(p-3))
    while np.linalg.det(S + alpha * np.eye(p)) < threshold:
        alpha += 0.01
    s = S + alpha * np.eye(p)
    return s

class DPEI():
    def __init__(self):
        self.X = None
        self.y = None

    def fit(self, X, y = None):
        """Fit the my model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.n_samples, _ = X.shape
        self.X = X.astype(np.float64)
        if y is not None:
            self.y = y
            self.classes_ = np.unique(y)
            n_classes = self.classes_.shape[0]
            if self.n_samples == n_classes:
                raise ValueError(
                    "The number of samples must be more than the number of classes."
                )

        return self

    def transform(self):
        """Project data to maximize class separation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components) or \
            (n_samples, min(rank, n_components))
            Transformed data. In the case of the 'svd' solver, the shape
            is (n_samples, min(rank, n_components)).
        """
        
        if np.isnan(self.X).any():
            if self.y is not None:
                return self._transform_with_label()
            else:
                return self._transform_without_label()
        else:
            print("No missing values found in the input array. Returning the input array.")
            return self.X
        
        
    def _transform_without_label(self):
        Ximp = self.X.copy()
        S = _cov(self.X, n_jobs=None)
        if np.linalg.det(S) < 0:
            S = _shinkage_cov_estimator(S)
        for idx in range(self.n_samples):
            missing_idxes = np.isnan(self.X[idx])
            if np.any(missing_idxes):
                obs_indices = ~missing_idxes
                X_o = self.X[idx, obs_indices]
                mu_o = self.mus[obs_indices]
                mu_m = self.mus[missing_idxes]
                S_oo = S[np.ix_(obs_indices, obs_indices)]
                S_oo = _shinkage_cov_estimator(S_oo)
                S_mo = self.cov[np.ix_(missing_idxes, obs_indices)]
                Ximp[idx, missing_idxes] = mu_m + S_mo @ np.inv(S_oo) @ (X_o - mu_o)

        return Ximp
    
    def _transform_with_label(self):
        Ximp = np.empty_like(self.X)
        for label in self.classes_:
            class_idxes = np.where(self.y == label)[0]
            class_data = self.X[class_idxes]
            imputed_class_data = self._transform_without_label(class_data)
            Ximp[class_idxes] = imputed_class_data
        return Ximp