import numpy as np
import scipy.linalg as la


def prewhiten(X, num_components=None):
    """
    Prewhiten the data matrix X and reduce dimensionality to num_components if specified.
    """
    # Subtract the mean of each row
    X_mean = X.mean(axis=1, keepdims=True)
    X_centered = X - X_mean

    # Compute the covariance matrix
    cov_matrix = np.cov(X_centered)

    # Perform eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = la.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select a subset of the eigenvectors if num_components is specified
    if num_components is not None and num_components < eigenvectors.shape[1]:
        eigenvectors = eigenvectors[:, :num_components]
        eigenvalues = eigenvalues[:num_components]

    # Compute the whitening matrix using the inverse square root of the eigenvalues
    whitening_matrix = (eigenvectors / np.sqrt(eigenvalues)).T

    # Apply the whitening matrix to X_centered
    X_whitened = whitening_matrix @ X_centered

    return X_whitened, whitening_matrix


def compute_cumulants(X):
    """
    Compute the fourth-order cumulants of the data matrix X.
    """
    m, n = X.shape
    cumulants = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                # Fourth cumulant for i == j is the kurtosis minus 3 (since data is standardized)
                cumulants[i, j] = np.mean(X[i] ** 4) - 3
            else:
                # For i != j, the cumulant is the joint cumulant
                cumulants[i, j] = np.mean(X[i] ** 2 * X[j] ** 2) - (
                    np.mean(X[i] ** 2) * np.mean(X[j] ** 2)
                )
    return cumulants


# Correcting the implementation of the optimization function


def optimize_contrast_function(X):
    """
    Optimize a contrast function to find the rotation matrix.
    Here we use a simple form of contrast function based on the fourth cumulant (kurtosis).
    The goal is to maximize non-Gaussianity which is measured by the absolute value of kurtosis.
    """
    m, n = X.shape

    # Initialize the rotation matrix to identity
    rotation_matrix = np.eye(m)

    # Define the learning rate
    learning_rate = 0.1

    # Perform gradient ascent
    for _ in range(100):  # Number of iterations
        # Compute the cumulants for the rotated data
        cumulants = compute_cumulants(rotation_matrix.T @ X)
        # Update rule: rotation_matrix += learning_rate * gradient
        # The gradient is approximated by the cumulants since we aim to maximize them
        rotation_matrix += learning_rate * cumulants @ rotation_matrix
        # Re-orthogonalize the rotation matrix
        u, _, vh = la.svd(rotation_matrix, full_matrices=False)
        rotation_matrix = u @ vh

    return rotation_matrix
