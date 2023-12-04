from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

class JADE:
    def __init__(self, num_components: int = 4) -> None:
        self.num_components: int = num_components
        self.unmixing_matrix: np.ndarray = None
        self.whitening_matrix: np.ndarray = None
        self.ica_jade_loadings: np.ndarray = None
        self.ica_jade_corr: pd.DataFrame = None
        self.ica_jade_ids: List[str] = None

    def fit(self, mixed_signal_matrix: np.ndarray) -> np.ndarray:
        """
        Fit the JADE model to the data.

        Parameters:
        mixed_signal_matrix (numpy.ndarray): The mixed signal data matrix.

        Returns:
        numpy.ndarray: The unmixing matrix after applying JADE.
        """
        mixed_signal_matrix = np.array(mixed_signal_matrix)
        unmixing_matrix, self.whitening_matrix = jadeR(mixed_signal_matrix, num_components=self.num_components)

        # Adjust the sign of each row for better interpretability
        for i in range(unmixing_matrix.shape[0]):
            if np.abs(np.max(unmixing_matrix[i, :])) < np.abs(np.min(unmixing_matrix[i, :])):
                unmixing_matrix[i, :] *= -1

        self.unmixing_matrix = unmixing_matrix

        return unmixing_matrix

    def transform(self, mixed_signal_matrix: np.ndarray) -> np.ndarray:
        if self.unmixing_matrix is None or self.whitening_matrix is None:
            raise ValueError("Model has not been fit yet. Call 'fit' with training data.")

        # Transpose the mixed_signal_matrix to align it for matrix multiplication
        mixed_signal_matrix = mixed_signal_matrix.T

        # First, apply the whitening matrix to the transposed input data
        whitened_data = np.dot(self.whitening_matrix, mixed_signal_matrix)

        # Apply the unmixing matrix to the whitened data
        separated_signals = np.dot(self.unmixing_matrix, whitened_data).T
        return separated_signals

    def correlate_loadings(self, df: pd.DataFrame, corrcols: List[str], ic_labels: List[str]) -> None:
        """
        Find the correlation between loadings and a set of columns.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing data.
        corrcols (list): List of columns to correlate.
        ic_labels (list): List of ICA column labels.

        Updates:
        self.ica_jade_corr: DataFrame of correlations.
        self.ica_jade_ids: Identifiers for the correlated loadings.
        """
        if self.unmixing_matrix is None:
            raise ValueError("Model has not been fit yet. Call 'fit' with training data.")

        # Compute the correlation matrix and filter it for relevant columns and rows
        corrdf = df.corr().drop(ic_labels, axis=1).drop(corrcols, axis=0)
        ica_jade_ids = []

        # Iterate over each independent component label
        for ic_label in ic_labels:
            tmp = corrdf.loc[ic_label]
            max_corr = np.max(tmp)
            match = tmp.values == max_corr
            matched_col = corrcols[np.where(match)[0][0]]
            ica_jade_ids.append(f"{matched_col} (r={np.round(max_corr, 1)})")

        self.ica_jade_corr = corrdf
        self.ica_jade_ids = ica_jade_ids



def validate_input(X_input, num_components=None, verbose=True):
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
    if num_components is None:
        num_components = num_signals
    # Check if the number of sources does not exceed the number of sensors
    assert num_components <= num_signals, \
        "jade -> Do not ask more sources (%d) than sensors (%d) here!!!" % (num_components, num_signals)

    # Verbose output
    if verbose:
        print("jade -> Looking for " + str(num_components) + " sources")
        print("jade -> Removing the mean value")

    # Remove the mean value from X
    X_input -= X_input.mean(1)

    return X_input, input_data_type, num_components, num_samples


def perform_whitening(preprocessed_data, num_components, verbose=True):
    """
    Perform whitening on the given preprocessed data.

    Parameters:
    preprocessed_data (numpy.ndarray): The data matrix after preprocessing.
    num_samples (int): The number of samples in each signal.
    num_components (int): The number of independent components to extract.
    verbose (bool): If True, additional information is printed.

    Returns:
    numpy.ndarray: The whitened data matrix.
    numpy.ndarray: The whitening matrix.
    """
    print("function: perform whitening")
    if verbose:
        print("jade -> Performing whitening on the data")

    # Compute the covariance matrix of the data
    covariance_matrix = np.cov(preprocessed_data.T)

    # Perform eigenvalue decomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues in descending order and get the sorted indices
    sorted_indices = np.argsort(eigenvalues)[::-1]

    # Select the top 'num_components' eigenvectors
    eigenvectors = eigenvectors[:, sorted_indices[:num_components]]
    eigenvalues = eigenvalues[sorted_indices[:num_components]]

    # Whitening: Create the whitening matrix
    scaling_factors = np.sqrt(eigenvalues)
    whitening_matrix = (eigenvectors / scaling_factors).T

    # Apply the whitening matrix to the data
    whitened_data = np.dot(whitening_matrix, preprocessed_data.T)

    if verbose:
        print("Shape of whitening_matrix:", whitening_matrix.shape)
        print("Shape of whitened_data:", whitened_data.shape)
        print("jade -> Whitening completed")

    return whitened_data.T, whitening_matrix


def initialize_cumulant_matrices_storage(num_samples, num_components):
    """
    Initialize the storage for cumulant matrices.

    Parameters:
    num_samples (int): Number of samples in the dataset.
    num_components (int): Number of principal components.

    Returns:
    numpy.matrix: Initialized matrix for storing cumulant matrices.
    int: Number of cumulant matrices.
    """
    print("function: initialize cumulant matrices storage")
    # Validate input
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")
    if not isinstance(num_components, int) or num_components <= 0:
        raise ValueError("num_components must be a positive integer.")

    # The number of cumulant matrices is equal to the number of components.
    num_cumulant_matrices = num_components

    # Initialize a zero matrix for storing cumulant matrices.
    # The size is num_samples x (num_samples * num_cumulant_matrices)
    # This structure allows storing each cumulant matrix as a column block.
    cumulant_matrices_storage = np.zeros([num_components, num_components * num_components], dtype=np.float64)
    print("Shape of cumulant matrices storage {}", cumulant_matrices_storage.shape)

    # Ensure that the matrix has the correct dimensions
    expected_shape = (num_components, num_components * num_cumulant_matrices)
    print("Expected shape of cumulant matrices storage {}", expected_shape)
    assert cumulant_matrices_storage.shape == expected_shape, \
        f"Cumulant matrices storage has incorrect dimensions. Expected: {expected_shape}, Got: {cumulant_matrices_storage.shape}"

    return cumulant_matrices_storage, num_cumulant_matrices


def compute_cumulant_matrix(preprocessed_data, num_components, component_index, num_cumulant_matrices):
    """
    Compute an individual cumulant matrix for a given component.

    Parameters:
    preprocessed_data (numpy.matrix): Transposed preprocessed data matrix.
    num_components (int): Number of components.
    component_index (int): Index of the current component.
    num_cumulant_matrices (int): Total number of cumulant matrices.

    Returns:
    numpy.matrix: The computed cumulant matrix for the given component.
    """
    print("function: compute cumulant matrix")
    # Validate input
    if not isinstance(preprocessed_data, np.matrix) or preprocessed_data.ndim != 2:
        raise TypeError("preprocessed_data must be a 2-dimensional numpy matrix.")

    if not isinstance(num_components, int) or num_components <= 0:
        raise ValueError("num_components must be a positive integer.")

    if not isinstance(component_index, int) or component_index < 0 or component_index >= preprocessed_data.shape[1]:
        raise ValueError("component_index must be a non-negative integer less than the number of columns in preprocessed_data.")

    if not isinstance(num_cumulant_matrices, int) or num_cumulant_matrices <= 0:
        raise ValueError("num_cumulant_matrices must be a positive integer.")

    # Extract the signal corresponding to the specified component index
    component_signal = preprocessed_data[:, component_index]

    # Initialize the cumulant matrix
    cumulant_matrix = np.zeros((num_components, num_components))

    # Vectorize computation for the cumulant matrix
    for i in range(num_components):
        Xii = component_signal[i] * component_signal[i]  # Square of the ith component
        for j in range(i, num_components):
            Xjj = component_signal[j] * component_signal[j]  # Square of the jth component
            Xij = component_signal[i] * component_signal[j]  # Product of ith and jth components

            # Compute the cumulant
            cumulant = np.mean(Xii * Xjj) - 3 * np.mean(Xij) ** 2

            # Assign cumulant value (exploiting symmetry)
            cumulant_matrix[i, j] = cumulant
            if i != j:
                cumulant_matrix[j, i] = cumulant

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
        - on_diagonal (float): Sum of squared diagonal elements.
        - off_diagonal (float): Sum of squared off-diagonal elements.
    """
    print("function: initialize diagonalization")
    # Validate input
    if not isinstance(num_components, int) or num_components <= 0:
        raise ValueError("num_components must be a positive integer.")

    if not isinstance(num_cumulant_matrices, int) or num_cumulant_matrices <= 0:
        raise ValueError("num_cumulant_matrices must be a positive integer.")

    # Initialize the rotation matrix as an identity matrix of size num_components
    # This matrix will be updated during the joint diagonalization process.
    rotation_matrix = np.matrix(np.eye(num_components, dtype=np.float64))

    # Initialize on_diagonal and off_diagonal values
    # These values will be used to track the progress of the diagonalization process
    # and assess its convergence.
    on_diagonal = 0.0
    off_diagonal = 0.0

    # The initial values of on_diagonal and off_diagonal are set to zero since the
    # rotation_matrix is initialized as an identity matrix, and thus, all off-diagonal
    # elements are initially zero.

    return rotation_matrix, on_diagonal, off_diagonal


def joint_diagonalization(cumulant_matrices, num_components):
    print("function: joint diagonalization")

    # Input validation
    if not isinstance(cumulant_matrices, np.ndarray) or cumulant_matrices.ndim != 2:
        raise TypeError("cumulant_matrices must be a 2-dimensional numpy array.")

    if cumulant_matrices.shape[0] != num_components or cumulant_matrices.shape[1] != num_components * num_components:
        raise ValueError("cumulant_matrices must have shape (num_components, num_components * num_components)")

    if not isinstance(num_components, int) or num_components <= 0:
        raise ValueError("num_components must be a positive integer.")

    # Initialize rotation matrix
    rotation_matrix = np.eye(num_components)

    # Convergence threshold and initialization
    convergence_threshold = 1.0e-6 / np.sqrt(num_components)
    continue_diagonalization = True
    max_iterations = 1000  # Set a maximum number of iterations to prevent potential infinite loop
    iteration_count = 0

    while continue_diagonalization and iteration_count < max_iterations:
        continue_diagonalization = False
        iteration_count += 1

        for component_p in range(num_components - 1):
            index_p = np.arange(component_p, num_components * num_components, num_components)
            Cpp = np.sum(cumulant_matrices[component_p, index_p] ** 2)

            for component_q in range(component_p + 1, num_components):
                index_q = np.arange(component_q, num_components * num_components, num_components)
                Cqq = np.sum(cumulant_matrices[component_q, index_q] ** 2)
                Cpq = np.sum(cumulant_matrices[component_p, index_p] * cumulant_matrices[component_q, index_q])

                tonality = Cpp - Cqq
                off_diagonal_sum_new = 2 * Cpq
                rotation_angle = 0.5 * np.arctan2(off_diagonal_sum_new, tonality + np.sqrt(tonality * tonality + off_diagonal_sum_new * off_diagonal_sum_new))

                # Update based on Givens rotation
                if abs(rotation_angle) > convergence_threshold:
                    continue_diagonalization = True
                    cosine_theta = np.cos(rotation_angle)
                    sine_theta = np.sin(rotation_angle)
                    givens_matrix = np.array([[cosine_theta, -sine_theta], [sine_theta, cosine_theta]])
                    component_pair = np.array([component_p, component_q])

                    rotation_matrix[:, component_pair] = np.dot(rotation_matrix[:, component_pair], givens_matrix)
                    cumulant_matrices[component_pair, :] = np.dot(givens_matrix.T, cumulant_matrices[component_pair, :])
                    cumulant_matrices[:, np.concatenate([index_p, index_q])] = \
                        np.append(cosine_theta * cumulant_matrices[:, index_p] + sine_theta * cumulant_matrices[:, index_q],
                                  -sine_theta * cumulant_matrices[:, index_p] + cosine_theta * cumulant_matrices[:, index_q], axis=1)

    if iteration_count >= max_iterations:
        print("Warning: Maximum iterations reached in joint diagonalization")

    return rotation_matrix


def sort_separating_matrix(separating_matrix: np.ndarray) -> np.ndarray:
    """
    Sort the rows of the separating matrix based on the energy of the components.

    Parameters:
    separating_matrix (numpy.ndarray): The separating matrix.

    Returns:
    numpy.ndarray: Sorted separating matrix.
    """
    # Validate input
    if not isinstance(separating_matrix, np.ndarray) or separating_matrix.ndim != 2:
        raise TypeError("separating_matrix must be a 2-dimensional numpy array.")

    # Compute the pseudo-inverse (mixing matrix) of the separating matrix
    mixing_matrix = np.linalg.pinv(separating_matrix)

    # Calculate the energy of each component in the mixing matrix
    energy_per_component = np.sum(np.square(mixing_matrix), axis=0)
    energy_per_component = np.asarray(energy_per_component).ravel()  # Convert to 1D array

    # Determine the order of components based on their energy (descending order)
    energy_order = np.argsort(energy_per_component)[::-1]

    # Sort the separating matrix rows according to the energy order
    sorted_matrix = separating_matrix[energy_order, :]

    # No need to convert back to matrix
    return sorted_matrix[::-1, :]


def fix_matrix_signs(separating_matrix: np.ndarray) -> np.ndarray:
    """
    Adjust the signs of the rows of the separating matrix.

    Parameters:
    separating_matrix (numpy.ndarray): The separating matrix.

    Returns:
    numpy.ndarray: The separating matrix with adjusted signs.
    """
    # Validate input
    if not isinstance(separating_matrix, np.ndarray) or separating_matrix.ndim != 2:
        raise TypeError("separating_matrix must be a 2-dimensional numpy array.")

    # Iterate over each row of the separating matrix
    for i in range(separating_matrix.shape[0]):
        # Check the sign of the first element in each row
        # If the sign is negative or zero, multiply the entire row by -1
        # This standardizes the sign of the components for consistency
        if np.sign(separating_matrix[i, 0]) <= 0:
            separating_matrix[i, :] *= -1

    return separating_matrix


def jadeR(mixed_signal_matrix: np.ndarray, num_components: Optional[int] = None, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:

        mixed_signal_matrix -- an nxT data matrix (n sensors, T samples). May be a numpy array or
             matrix.

        num_components -- output matrix B has size mxn so that only m sources are
             extracted.  This is done by restricting the operation of jadeR
             to the m first principal components. Defaults to None, in which
             case m=n.

        verbose -- print info on progress. Default is True.

    Returns:

        An m*n matrix B (NumPy matrix type), such that Y=B*X are separated
        sources extracted from the n*T data matrix X. If m is omitted, B is a
        square n*n matrix (as many sources as sensors). The rows of B are
        ordered such that the columns of pinv(B) are in order of decreasing
        norm; this has the effect that the `most energetically significant`
        components appear first in the rows of Y=B*X.
    """

    # GB: we do some checking of the input arguments and copy data to new
    # variables to avoid messing with the original input. We also require double
    # precision (float64) and a numpy matrix type for preprocessed_data.

    # Original code had: X, origtype, m, n, T

    # Validating input and performing whitening & PCA
    preprocessed_data, input_data_type, num_components, num_samples = validate_input(mixed_signal_matrix, num_components, verbose)
    whitened_data, whitened_matrix = perform_whitening(preprocessed_data, num_components, verbose)

    if verbose:
        print("jade -> Estimating cumulant matrices")

    # Initialize the storage for cumulant matrices
    cumulant_matrices_storage, num_cumulant_matrices = initialize_cumulant_matrices_storage(num_samples, num_components)

    # Compute and store cumulant matrices
    for component_index in range(num_components):
        cumulant_matrix = compute_cumulant_matrix(whitened_data.T, num_components, component_index, num_cumulant_matrices)

        # Store the computed cumulant matrix
        storage_start_index = component_index * num_components
        storage_end_index = storage_start_index + num_components
        cumulant_matrices_storage[:, storage_start_index:storage_end_index] = cumulant_matrix

    # Perform joint diagonalization
    rotation_matrix = joint_diagonalization(cumulant_matrices_storage, num_components)

    # Extract separating matrix
    separating_matrix = rotation_matrix.T

    # Sorting the components
    if verbose:
        print("jade -> Sorting the components")

    separating_matrix = sort_separating_matrix(separating_matrix)

    # Fix the signs of the separating matrix
    if verbose:
        print("jade -> Fixing the signs")

    separating_matrix = fix_matrix_signs(separating_matrix)

    return separating_matrix.astype(input_data_type), whitened_matrix
