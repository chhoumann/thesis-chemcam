from sys import stdout
import numpy as np
from numpy import *
from numpy.linalg import eig, pinv

def check_input(X_input, num_sources=None, verbose=True):
    # Check if X is a NumPy ndarray
    assert isinstance(X_input, np.ndarray), \
        "X (input data matrix) is of the wrong type (%s)" % type(X_input)
    
    # Remember the original data type of X
    input_data_type = X_input.dtype

    # Convert X to a NumPy matrix of type float64
    X_input = np.matrix(X_input.astype(np.float64))

    # Check if X is a 2-dimensional matrix
    assert X_input.ndim == 2, "X_input has %d dimensions, should be 2" % X_input.ndim
    
    # Check if the verbose parameter is either True or False
    assert isinstance(verbose, bool), \
        "verbose parameter should be either True or False"

    # Get the number of input signals (num_signals (n)) and number of samples (num_samples (T))
    num_signals, num_samples = X_input.shape

    # Set the number of sources to the number of sensors if not specified
    if num_sources is None:
        num_sources = num_signals
    # Check if the number of sources does not exceed the number of sensors
    assert num_sources <= num_signals, \
        "jade -> Do not ask more sources (%d) than sensors (%d) here!!!" % (num_sources, num_signals)

    # Verbose output
    if verbose:
        print("jade -> Looking for " + str(num_sources) + " sources")
        print("jade -> Removing the mean value")
    
    # Remove the mean value from X
    X_input -= X_input.mean(1)

    return X_input, input_data_type, num_sources, num_samples

def perform_PCA_and_whitening(preprocessed_data, num_components, num_signals, num_samples, verbose=False):
    """
    Perform Principal Component Analysis (PCA) and whitening on the given preprocessed data.

    Parameters:
    preprocessed_data (numpy.matrix): The data matrix after preprocessing.
    num_components (int): The number of principal components to extract.
    num_signals (int): The total number of signals in the data.
    num_samples (int): The number of samples in each signal.
    verbose (bool): If True, additional information is printed.

    Returns:
    numpy.matrix: The matrix of principal components. Each column is a principal component.
    numpy.array: The array of sorted eigenvalues corresponding to the principal components.
    """

    if verbose:
        print("jade -> Whitening the data")

    # Compute the covariance matrix of the whitened data
    covariance_matrix = (preprocessed_data * preprocessed_data.T) / float(num_samples)

    # Perform eigenvalue decomposition to find the principal components
    eigenvalues, eigenvectors = eig(covariance_matrix)

    # Sort eigenvalues in descending order
    sorted_indices = eigenvalues.argsort()[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]

    # Extract the principal components corresponding to the most significant eigenvalues
    principal_components = eigenvectors[:, sorted_indices[:num_components]]

    # Transpose the matrix so each row represents a principal component
    principal_components = principal_components.T

    return principal_components, sorted_eigenvalues

def initialize_cumulant_matrices_storage(num_components):
    """
    Initialize the storage for cumulant matrices.

    Parameters:
    num_components (int): Number of principal components.

    Returns:
    numpy.matrix: Initialized matrix for storing cumulant matrices.
    int: Number of cumulant matrices.
    """
    dim_symmetric_matrices = (num_components * (num_components + 1)) / 2
    num_cumulant_matrices = int(dim_symmetric_matrices)
    cumulant_matrices_storage = np.matrix(np.zeros([num_components, num_components * num_cumulant_matrices], dtype=np.float64))

    return cumulant_matrices_storage, num_cumulant_matrices

def compute_cumulant_matrix(preprocessed_data, num_samples, component_index, num_cumulant_matrices):
    """
    Compute an individual cumulant matrix for a given component.

    Parameters:
    preprocessed_data (numpy.matrix): Transposed preprocessed data matrix.
    num_samples (int): Number of samples in each signal.
    component_index (int): Index of the current component.
    num_cumulant_matrices (int): Total number of cumulant matrices.

    Returns:
    numpy.matrix: The computed cumulant matrix for the given component.
    """
    component_signal = preprocessed_data[:, component_index]
    component_signal_squared = np.multiply(component_signal, component_signal)
    cumulant_matrix = np.multiply(component_signal_squared, preprocessed_data).T * preprocessed_data / float(num_samples)

    return cumulant_matrix

def initialize_diagonalization(num_components, num_cumulant_matrices):
    """
    Initialize matrices and variables for the diagonalization process.

    Parameters:
    num_components (int): Number of principal components.
    num_cumulant_matrices (int): Total number of cumulant matrices.

    Returns:
    Tuple containing:
        - rotation_matrix (numpy.matrix): Matrix for joint diagonalization.
        - diagonal_values (numpy.array): Diagonal values array.
        - on_diagonal (float): Sum of squared diagonal elements.
        - off_diagonal (float): Sum of squared off-diagonal elements.
    """
    rotation_matrix = np.matrix(np.eye(num_components, dtype=np.float64))
    on_diagonal = 0.0
    off_diagonal = 0.0

    return rotation_matrix, on_diagonal, off_diagonal

def joint_diagonalization(cumulant_matrices_storage, num_components, num_cumulant_matrices, num_samples):
    """
    Perform joint diagonalization on the cumulant matrices.

    Parameters:
    cumulant_matrices_storage (numpy.matrix): Storage matrix containing cumulant matrices.
    num_components (int): Number of principal components.
    num_cumulant_matrices (int): Total number of cumulant matrices.
    num_samples (int): Number of samples in each signal.

    Returns:
    numpy.matrix: The diagonalized matrix.
    """
    rotation_matrix, on_diagonal, off_diagonal = initialize_diagonalization(num_components, num_cumulant_matrices)

    threshold = 1.0e-6 / np.sqrt(num_samples)
    encore = True
    sweep = 0
    updates = 0

    while encore:
        encore = False
        sweep += 1
        upds = 0

        for p in range(num_components - 1):
            for q in range(p + 1, num_components):

                Ip = np.arange(p, num_components * num_cumulant_matrices, num_components)
                Iq = np.arange(q, num_components * num_cumulant_matrices, num_components)

                # Compute Givens angles
                g = np.concatenate([cumulant_matrices_storage[p, Ip] - cumulant_matrices_storage[q, Iq], 
                                    cumulant_matrices_storage[p, Iq] + cumulant_matrices_storage[q, Ip]])
                gg = np.dot(g, g.T)
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton * ton + toff * toff))
                Gain = (np.sqrt(ton * ton + toff * toff) - ton) / 4.0

                # Givens update
                if abs(theta) > threshold:
                    encore = True
                    upds += 1
                    c = np.cos(theta)
                    s = np.sin(theta)
                    G = np.matrix([[c, -s], [s, c]])
                    pair = np.array([p, q])

                    rotation_matrix[:, pair] *= G
                    cumulant_matrices_storage[pair, :] = G.T * cumulant_matrices_storage[pair, :]
                    cumulant_matrices_storage[:, np.concatenate([Ip, Iq])] = \
                        np.append(c * cumulant_matrices_storage[:, Ip] + s * cumulant_matrices_storage[:, Iq], 
                                  -s * cumulant_matrices_storage[:, Ip] + c * cumulant_matrices_storage[:, Iq], axis=1)
                    on_diagonal += Gain
                    off_diagonal -= Gain

        updates += upds

    return rotation_matrix

def sort_separating_matrix(separating_matrix):
    """
    Sort the rows of the separating matrix based on the energy of the components.

    Parameters:
    separating_matrix (numpy.matrix): The separating matrix.

    Returns:
    numpy.matrix: Sorted separating matrix.
    """
    mixing_matrix = np.linalg.pinv(separating_matrix)
    energy_order = np.argsort(np.sum(np.multiply(mixing_matrix, mixing_matrix), axis=0))[::-1]
    sorted_matrix = separating_matrix[energy_order, :]

    return sorted_matrix[::-1, :]  # Reverse to have the most energetic components first

def fix_matrix_signs(separating_matrix):
    """
    Adjust the signs of the rows of the separating matrix.

    Parameters:
    separating_matrix (numpy.matrix): The separating matrix.

    Returns:
    numpy.matrix: The separating matrix with adjusted signs.
    """
    for i in range(separating_matrix.shape[0]):
        if np.sign(separating_matrix[i, 0]) == -1 or np.sign(separating_matrix[i, 0]) == 0:
            separating_matrix[i, :] *= -1
            
    return separating_matrix