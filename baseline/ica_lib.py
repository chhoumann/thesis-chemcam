from sys import stdout
import numpy as np
from numpy import *
from numpy.linalg import eig, pinv
from pathlib import Path
import pandas as pd
from reproduction import masks


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

def perform_PCA_and_whitening(preprocessed_data, num_components, num_samples, verbose=False):
    """
    Perform Principal Component Analysis (PCA) and whitening on the given preprocessed data.

    Parameters:
    preprocessed_data (numpy.matrix): The data matrix after preprocessing.
    num_components (int): The number of principal components to extract.
    num_samples (int): The number of samples in each signal.
    verbose (bool): If True, additional information is printed.

    Returns:
    numpy.matrix: The matrix of principal components. Each column is a principal component.
    numpy.array: The array of sorted eigenvalues corresponding to the principal components.
    numpy.matrix: The whitening matrix.
    """

    if verbose:
        print("jade -> Performing PCA and whitening the data")

    # Validate input data
    if not isinstance(preprocessed_data, np.matrix):
        raise TypeError("preprocessed_data must be a numpy matrix.")
    
    if preprocessed_data.ndim != 2:
        raise ValueError("preprocessed_data must be a 2-dimensional matrix.")
    
    num_signals, _ = preprocessed_data.shape
    if num_components is None or not isinstance(num_components, int):
        raise TypeError("num_components must be an integer.")
    
    if num_components < 1 or num_components > num_signals:
        raise ValueError("num_components must be between 1 and the number of signals (rows) in preprocessed_data.")

    if num_samples is None or not isinstance(num_samples, int):
        raise TypeError("num_samples must be an integer.")
    
    if num_samples < 1:
        raise ValueError("num_samples must be a positive integer.")

    # Compute the covariance matrix of the data
    covariance_matrix = (preprocessed_data * preprocessed_data.T) / float(num_samples)

    # Perform eigenvalue decomposition to find the principal components
    eigenvalues, eigenvectors = eig(covariance_matrix)

    # Sort eigenvalues in descending order and get the sorted indices
    sorted_indices = eigenvalues.argsort()[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]

    # Extract the principal components corresponding to the most significant eigenvalues
    principal_components = eigenvectors[:, sorted_indices[:num_components]]

    # Transpose the matrix so each row represents a principal component
    principal_components = principal_components.T

    # Whitening: Scale the principal components to have unit variance
    # The scaling factor for each principal component is the inverse of the square root of its corresponding eigenvalue
    scaling_factors = np.sqrt(sorted_eigenvalues[:num_components])
    whitening_matrix = np.diag(1. / scaling_factors) * principal_components.T

    if verbose:
        print("jade -> PCA and whitening completed")

    return principal_components, sorted_eigenvalues, whitening_matrix

def initialize_cumulant_matrices_storage(num_components):
    """
    Initialize the storage for cumulant matrices.

    Parameters:
    num_components (int): Number of principal components.

    Returns:
    numpy.matrix: Initialized matrix for storing cumulant matrices.
    int: Number of cumulant matrices.
    """

    # Validate input
    if not isinstance(num_components, int) or num_components <= 0:
        raise ValueError("num_components must be a positive integer.")

    # Calculate the number of elements in a symmetric matrix of size num_components
    # A symmetric matrix has (n * (n + 1)) / 2 unique elements.
    dim_symmetric_matrices = (num_components * (num_components + 1)) / 2

    # The number of cumulant matrices is equal to the number of unique elements
    # in a symmetric matrix of size num_components.
    num_cumulant_matrices = int(dim_symmetric_matrices)

    # Initialize a zero matrix for storing cumulant matrices.
    # The size is num_components x (num_components * num_cumulant_matrices)
    # This structure allows storing each cumulant matrix as a column block.
    cumulant_matrices_storage = np.matrix(np.zeros([num_components, num_components * num_cumulant_matrices], dtype=np.float64))

    # Ensure that the matrix has the correct dimensions
    assert cumulant_matrices_storage.shape == (num_components, num_components * num_cumulant_matrices), \
        "Cumulant matrices storage has incorrect dimensions."

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

    # Validate input
    if not isinstance(preprocessed_data, np.matrix) or preprocessed_data.ndim != 2:
        raise TypeError("preprocessed_data must be a 2-dimensional numpy matrix.")

    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")

    if not isinstance(component_index, int) or component_index < 0 or component_index >= preprocessed_data.shape[1]:
        raise ValueError("component_index must be a non-negative integer less than the number of columns in preprocessed_data.")

    if not isinstance(num_cumulant_matrices, int) or num_cumulant_matrices <= 0:
        raise ValueError("num_cumulant_matrices must be a positive integer.")

    # Extract the signal corresponding to the specified component index
    component_signal = preprocessed_data[:, component_index]

    # Compute the square of the component signal
    component_signal_squared = np.multiply(component_signal, component_signal)

    # Calculate the cumulant matrix
    cumulant_matrix = np.multiply(component_signal_squared, preprocessed_data).T * preprocessed_data / float(num_samples)

    # Validate the shape of the computed cumulant matrix
    num_signals = preprocessed_data.shape[0]
    if cumulant_matrix.shape != (num_signals, num_signals):
        raise ValueError("Computed cumulant matrix has incorrect dimensions.")

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

def joint_diagonalization(cumulant_matrices, num_principal_components, total_cumulant_matrices, sample_count):
    """
    Perform joint diagonalization on the cumulant matrices.

    Parameters:
    cumulant_matrices (numpy.matrix): Storage matrix containing cumulant matrices.
    num_principal_components (int): Number of principal components.
    total_cumulant_matrices (int): Total number of cumulant matrices.
    sample_count (int): Number of samples in each signal.

    Returns:
    numpy.matrix: The diagonalized matrix.
    """

    # Input validation
    if not isinstance(cumulant_matrices, np.matrix) or cumulant_matrices.ndim != 2:
        raise TypeError("cumulant_matrices must be a 2-dimensional numpy matrix.")
    
    if not isinstance(num_principal_components, int) or num_principal_components <= 0:
        raise ValueError("num_principal_components must be a positive integer.")

    if not isinstance(total_cumulant_matrices, int) or total_cumulant_matrices <= 0:
        raise ValueError("total_cumulant_matrices must be a positive integer.")

    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer.")

    # Initialize rotation matrix and diagonal/off-diagonal values
    rotation_matrix, on_diagonal_sum, off_diagonal_sum = initialize_diagonalization(num_principal_components, total_cumulant_matrices)

    convergence_threshold = 1.0e-6 / np.sqrt(sample_count)
    continue_diagonalization = True
    total_sweeps = 0
    total_updates = 0

    while continue_diagonalization:
        continue_diagonalization = False
        total_sweeps += 1
        current_sweep_updates = 0

        for component_p in range(num_principal_components - 1):
            for component_q in range(component_p + 1, num_principal_components):

                index_p = np.arange(component_p, num_principal_components * total_cumulant_matrices, num_principal_components)
                index_q = np.arange(component_q, num_principal_components * total_cumulant_matrices, num_principal_components)

                # Compute Givens rotation angles
                givens_vector = np.concatenate([cumulant_matrices[component_p, index_p] - cumulant_matrices[component_q, index_q], 
                                                cumulant_matrices[component_p, index_q] + cumulant_matrices[component_q, index_p]])
                givens_dot_product = np.dot(givens_vector, givens_vector.T)
                tonality = givens_dot_product[0, 0] - givens_dot_product[1, 1]
                off_diagonal_sum_new = givens_dot_product[0, 1] + givens_dot_product[1, 0]
                rotation_angle = 0.5 * np.arctan2(off_diagonal_sum_new, tonality + np.sqrt(tonality * tonality + off_diagonal_sum_new * off_diagonal_sum_new))
                rotation_gain = (np.sqrt(tonality * tonality + off_diagonal_sum_new * off_diagonal_sum_new) - tonality) / 4.0

                # Update based on Givens rotation
                if abs(rotation_angle) > convergence_threshold:
                    continue_diagonalization = True
                    current_sweep_updates += 1
                    cosine_theta = np.cos(rotation_angle)
                    sine_theta = np.sin(rotation_angle)
                    givens_matrix = np.matrix([[cosine_theta, -sine_theta], [sine_theta, cosine_theta]])
                    component_pair = np.array([component_p, component_q])

                    rotation_matrix[:, component_pair] *= givens_matrix
                    cumulant_matrices[component_pair, :] = givens_matrix.T * cumulant_matrices[component_pair, :]
                    cumulant_matrices[:, np.concatenate([index_p, index_q])] = \
                        np.append(cosine_theta * cumulant_matrices[:, index_p] + sine_theta * cumulant_matrices[:, index_q], 
                                  -sine_theta * cumulant_matrices[:, index_p] + cosine_theta * cumulant_matrices[:, index_q], axis=1)
                    on_diagonal_sum += rotation_gain
                    off_diagonal_sum -= rotation_gain

        total_updates += current_sweep_updates

    return rotation_matrix

def sort_separating_matrix(separating_matrix):
    """
    Sort the rows of the separating matrix based on the energy of the components.

    Parameters:
    separating_matrix (numpy.matrix): The separating matrix.

    Returns:
    numpy.matrix: Sorted separating matrix.
    """

    # Validate input
    if not isinstance(separating_matrix, np.matrix) or separating_matrix.ndim != 2:
        raise TypeError("separating_matrix must be a 2-dimensional numpy matrix.")

    # Compute the pseudo-inverse (mixing matrix) of the separating matrix
    # The mixing matrix represents the inverse transformation of the separating matrix
    mixing_matrix = np.linalg.pinv(separating_matrix)

    # Calculate the energy of each component in the mixing matrix
    # Energy is computed as the sum of squares of the elements in each column
    energy_per_component = np.sum(np.square(mixing_matrix), axis=0)

    # Determine the order of components based on their energy (descending order)
    energy_order = np.argsort(energy_per_component)[::-1]

    # Sort the separating matrix rows according to the energy order
    # This places the most energetic components at the top
    sorted_matrix = separating_matrix[energy_order, :]

    # Return the sorted matrix with the most energetic components first
    # The reversal is done to align with the standard convention in ICA
    return sorted_matrix[::-1, :]

def fix_matrix_signs(separating_matrix):
    """
    Adjust the signs of the rows of the separating matrix.

    Parameters:
    separating_matrix (numpy.matrix): The separating matrix.

    Returns:
    numpy.matrix: The separating matrix with adjusted signs.
    """

    # Validate input
    if not isinstance(separating_matrix, np.matrix) or separating_matrix.ndim != 2:
        raise TypeError("separating_matrix must be a 2-dimensional numpy matrix.")

    # Iterate over each row of the separating matrix
    for i in range(separating_matrix.shape[0]):
        # Check the sign of the first element in each row
        # If the sign is negative or zero, multiply the entire row by -1
        # This standardizes the sign of the components for consistency
        if np.sign(separating_matrix[i, 0]) <= 0:
            separating_matrix[i, :] *= -1

    return separating_matrix

def get_dataset_frame(dataset_path):
    with open(dataset_path) as f:
        # Find index of last line starting with "#" and skip rows until then
        for i, line in enumerate(f):
            if not line.startswith("#"):
                break
        # Read CSV from that line - columns also start with "#"
        return pd.read_csv(dataset_path, skiprows=i-1)

def preprocess_LIBS_data(file_path, debug=False):
    """
    Preprocess a single LIBS dataset file for ICA JADE algorithm.

    Parameters:
    file_path (str): Path to the dataset file.

    Returns:
    pd.DataFrame: Preprocessed data suitable for ICA JADE algorithm.
    """
    # Load the dataset using the get_dataset_frame function
    df = get_dataset_frame(file_path)

    # Clean up column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("# ", "")

    if debug:
        return df, df.columns
    else:
        # Drop columns not needed
        exclude = ["mean", "median"]
        first_five_shots = [f"shot{i}" for i in range(1, 6)]
        df.drop(exclude + first_five_shots, axis=1, inplace=True)

        # Apply masks to remove noisy wavelength ranges
        for mask in masks:
            df = df.loc[~((df["wave"] >= mask[0]) & (df["wave"] <= mask[1]))]

        # Transform the DataFrame
        transformed_df = transform_dataframe(df)
        
        return transformed_df

def transform_dataframe(df):
    """
    Transforms the DataFrame into a format where each row represents a shot
    and each column represents a wavelength.
    """
    # Extract wavelengths and use them as column headers
    wavelengths = df.iloc[:, 0]  # assuming the first column is 'wave'
    df = df.drop(df.columns[0], axis=1)  # drop the wavelength column

    # The remaining columns are shots, transpose them
    transformed_df = df.transpose()
    transformed_df.columns = wavelengths

    return transformed_df