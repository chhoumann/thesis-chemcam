import numpy as np
import pandas as pd


def identify_outliers_with_mad_iterative_multidim(X, k=3.0, max_iterations=10):
    """
    Identifies outliers in a multidimensional dataset (pandas DataFrame) based on the Median Absolute Deviation (MAD),
    applied feature-wise.

    Parameters:
    - X: A pandas DataFrame of data points (features).
    - k: The number of MADs away from the median to consider as an outlier.
    - max_iterations: Maximum number of iterations to prevent infinite loops.

    Returns:
    - indices of the data points that are not considered outliers across all features.
    - The number of iterations performed.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")

    n_samples, n_features = X.shape
    keep_mask = np.ones(n_samples, dtype=bool)

    for feature in range(n_features):
        data = X.iloc[:, feature].to_numpy()  # Convert the pandas Series to a NumPy array for calculations
        for iteration in range(max_iterations):
            if not np.any(keep_mask):  # If no samples left, stop
                break

            median = np.median(data[keep_mask])
            absolute_deviation = np.abs(data[keep_mask] - median)
            mad = np.median(absolute_deviation)

            if mad == 0:
                break

            modified_z_scores = 0.6745 * absolute_deviation / mad
            outliers = modified_z_scores > k

            # Update keep_mask to remove outliers for this feature
            keep_mask[keep_mask] = ~outliers  # Apply mask where keep_mask is already True

            if not np.any(outliers):
                break

    return np.where(keep_mask)[0], iteration + 1
