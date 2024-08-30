from dpers import DPER
import numpy as np
from sklearn.utils.validation import check_array
import warnings

def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask == "NaN" or np.isnan(value_to_mask):
        return np.isnan(X)
    else:
        return X == value_to_mask

def _shinkage_cov_estimator(S):
    """
    Compute the shrinkage covariance matrix.

    Parameters:
    -------
    S : ndarray of shape (n_features, n_features)
        covariance matrix

    Returns:
    -------
    s : array, shape (n_features, n_features)
        The shrinkage covariance matrix.
    """
    p = S.shape[1]
    alpha = 0
    #threshold = 0.01 - 0.0099*np.exp(-0.02*(p-3))
    threshold = 0.001
    while np.linalg.det(S + alpha*np.eye(p)) < threshold or np.linalg.det(S + alpha*np.eye(p)) == 0:
        alpha += 0.01
    s = S + alpha * np.eye(p)
    return s

class DPEImputer:
    def __init__(self, missing_values = np.nan, copy=True):
        self.missing_values = missing_values
        self.copy = copy

    def fit(self, X, y = None, window_size=None):
        """Fit the imputer on `X`.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like, shape (n_samples,), optional (default=None)
            The target data associated with X. This is optional and can be ignored.

        window_size : int
                The size of the window used to extend the feature matrix and perform imputation.
                This determines how many consecutive features are used in the imputation of
                missing values for each feature.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        n_features = X.shape[1]
        if window_size is None:
            if (n_features < 7):
                self.window_size = n_features
            else:
                self.window_size = 7
        else:
            self.window_size = window_size

        # Check the window_size is valid or not
        if (self.window_size > n_features):
            raise ValueError("The window size must be less than the number of features.")
        
        # Check data integrity and calling arguments
        force_all_finite = False if self.missing_values in ["NaN", np.nan] else True

        X = check_array(X, accept_sparse=False, dtype=np.float64,
                        force_all_finite=force_all_finite, copy=self.copy)

        # Check for +/- inf
        if np.any(np.isinf(X)):
            raise ValueError("+/- inf values are not supported.")
        
        # Check if any column has all missing
        mask = _get_mask(X, self.missing_values)
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            raise ValueError("One or more columns have all rows missing.")
        
        # First replace missing values with NaN if it is something else
        if self.missing_values not in ['NaN', np.nan]:
            X[np.where(X == self.missing_values)] = np.nan

        if y is not None:
            n_samples = X.shape[0]
            n_labels = np.unique(y).shape[0]
            if n_samples == n_labels:
                raise ValueError(
                    "The number of samples must be more than the number of classes."
                )

        return self


    def transform(self, X, y = None):
        """Impute all missing values in `X`.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The input data to complete.

        y:  array-like, shape (n_samples,), optional (default=None)
            The target data associated with X. This is optional and can be ignored.

        Returns
        -------
        X : {array-like}, shape (n_samples, n_features)
            The imputed dataset.
        """

        # Check data integrity
        force_all_finite = False if self.missing_values in ["NaN", np.nan] else True
        X = check_array(X, accept_sparse=False, dtype=np.float64,
                        force_all_finite=force_all_finite, copy=self.copy)
        
        # Check for +/- inf
        if np.any(np.isinf(X)):
            raise ValueError("+/- inf values are not supported.")
        
        # Check if any column has all missing
        mask = _get_mask(X, self.missing_values)
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            raise ValueError("One or more columns have all rows missing.")
        
        if not mask.sum() > 0:
            warnings.warn("No missing values found in the input array. Returning the input array.")
            return X
        
        X = X.astype(np.float64)
        # Call function to impute missing
        if y is not None:
            X = self._impute_windowed_with_label(X, y)
            
        else:
            X = self._impute_windowed_without_label(X)
        
        # Return imputed dataset
        return X

        
    def _impute_without_label(self, X):
        """
            Imputes missing values in the dataset X using a covariance-based method.

            Parameters:
            ----------
            X : {array-like}, shape (n_samples, n_features)
                The data may contain missing values that need to be imputed.

            Returns:
            -------
            X_imputed : {array-like}, shape (n_samples, n_features)
                The imputed data matrix, where missing values have been filled in.
        """
        # Create a copy of X to store the imputed data
        X_imputed = X.copy()
        n_samples = X.shape[0]

        # Estimate the covariance matrix of X
        S = DPER(X)
        # Apply shrinkage to the covariance matrix if its determinant is negative
        if np.linalg.det(S) < 0:
            S = _shinkage_cov_estimator(S)

        # Calculate the mean of each feature (ignoring NaNs)
        mus = np.nanmean(X, axis = 0)

        for idx in range(n_samples):
            # Identify missing values in the current sample
            missing_idxes = np.isnan(X[idx])
            if np.any(missing_idxes):
                # Identify observed values
                obs_idxes = ~missing_idxes
                X_o = X[idx, obs_idxes] # Observed values
                mu_o = mus[obs_idxes] # Mean of observed features
                mu_m = mus[missing_idxes] # Mean of missing features
                S_oo = S[np.ix_(obs_idxes, obs_idxes)] # Covariance of observed features
                S_oo = _shinkage_cov_estimator(S_oo) # Apply shrinkage to S_oo
                S_mo = S[np.ix_(missing_idxes, obs_idxes)] # Covariance between missing and observed features
                # Impute missing values based on conditional expectation
                X_imputed[idx, missing_idxes] = mu_m + S_mo @ np.linalg.inv(S_oo) @ (X_o - mu_o)

        # Return imputed dataset
        return X_imputed
    
    def _impute_with_label(self, X, y):
        """
            Transforms the dataset X by imputing missing values within each class/label in y.

            Parameters:
            ----------
            X : {array-like}, shape (n_samples, n_features)
                The data may contain missing values that need to be imputed.
                
            y : array-like, shape (n_samples,)
                The array of labels associated with the input data X. Each element in y corresponds to the label
                for the respective sample in X.

            Returns:
            -------
            X_imputed : {array-like}, shape (n_samples, n_features)
                The imputed data matrix, where missing values have been filled in.
        """

        # Initialize an empty array with the same shape as X for the imputed data
        X_imputed = np.empty_like(X)
        
        # Get the unique classes/labels in y
        unique_labels = np.unique(y)
        
        for label in unique_labels:
            # Get the indices of samples belonging to the current label
            label_idxes = np.where(y == label)[0]
            # Extract the data for the current class/label
            data_for_label = X[label_idxes]
            # Apply transformation without using the label
            imputed_label_data = self._impute_without_label(data_for_label)
            # Assign the imputed data back to the appropriate indices in X_imputed
            X_imputed[label_idxes] = imputed_label_data

        # Return imputed dataset
        return X_imputed
    
    def _impute_windowed_with_label(self, X, y):
        """
            Applies a windowed imputation process to the input feature matrix `X` using
            the given labels `y`. The function extends the feature matrix, imputes missing
            values within a specified window size, and then averages the imputed values.

            Parameters
            ----------
            X : {array-like}, shape (n_samples, n_features)
                The input feature matrix of shape (n_samples, n_features) where each row
                represents a sample and each column represents a feature.
                
            y : array-like, shape (n_samples,)
                The array of labels associated with the input data X. Each element in y corresponds to the label
                for the respective sample in X.

            Returns
            -------
            X_imputed : {array-like}, shape (n_samples, n_features)
                The imputed data matrix, where missing values have been filled in.
        """
        n_features = X.shape[1]
        extra_features = self.window_size - 1
        # Extend the feature matrix by adding the first 'extra_features' columns to the end
        X_extended = np.hstack([X, X[:, :extra_features]])
        # Initialize an array to store the imputed values
        X_imputed_total = np.zeros_like(X_extended)

        # Perform windowed imputation for each feature
        for i in range(n_features):
            X_pad = X_extended[:, i:i+self.window_size]
            X_pad_imputed = self._impute_with_label(X_pad, y)  #Label
            X_imputed_total[:, i:i+self.window_size] += X_pad_imputed

        # Extract the final imputed features
        X_final_imputed = X_imputed_total[:, :n_features]
        # Adjust for the overlap in the extended feature matrix
        for i in range(extra_features):
            X_final_imputed[:, i] += X_imputed_total[:, n_features + i]
        
        # Average the imputed values over the window size
        X_final_imputed /= self.window_size

        return X_final_imputed
    
    def _impute_windowed_without_label(self, X):
        """
            Applies a windowed imputation process to the input feature matrix `X` using
            the given labels `y`. The function extends the feature matrix, imputes missing
            values within a specified window size, and then averages the imputed values.

            Parameters
            ----------
            X : {array-like}, shape (n_samples, n_features)
                The input feature matrix of shape (n_samples, n_features) where each row
                represents a sample and each column represents a feature.
                
            y : array-like, shape (n_samples,)
                The array of labels associated with the input data X. Each element in y corresponds to the label
                for the respective sample in X.

            Returns
            -------
            X_imputed : {array-like}, shape (n_samples, n_features)
                The imputed data matrix, where missing values have been filled in.
        """

        n_features = X.shape[1]
        extra_features = self.window_size - 1
        # Extend the feature matrix by adding the first 'extra_features' columns to the end
        X_extended = np.hstack([X, X[:, :extra_features]])
        # Initialize an array to store the imputed values
        X_imputed_total = np.zeros_like(X_extended)

        # Perform windowed imputation for each feature
        for i in range(n_features):
            X_pad = X_extended[:, i:i+self.window_size]
            X_pad_imputed = self._impute_without_label(X_pad)
            X_imputed_total[:, i:i+self.window_size] += X_pad_imputed

        # Extract the final imputed features
        X_final_imputed = X_imputed_total[:, :n_features]

        # Adjust for the overlap in the extended feature matrix
        for i in range(extra_features):
            X_final_imputed[:, i] += X_imputed_total[:, n_features + i]

        # Average the imputed values over the window size
        X_final_imputed /= self.window_size

        return X_final_imputed

    def fit_transform(self, X, y=None, window_size=None):
        """Fit and impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        y:  array-like, shape (n_samples,), optional (default=None)
            The target data associated with X. This is optional and can be ignored.

        Returns
        -------
        X : {array-like}, shape (n_samples, n_features)
            Returns imputed dataset.
        """
        return self.fit(X, y, window_size).transform(X, y)
    