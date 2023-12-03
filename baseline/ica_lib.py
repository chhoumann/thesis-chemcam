from sys import stdout
import os
import numpy as np
from numpy import *
from numpy.linalg import eig, pinv
from pathlib import Path
import pandas as pd
from reproduction import masks
import matplotlib.pyplot as plt


def check_input(X_input, num_components=None, verbose=True):
    print("function: check input")

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


def perform_whitening(preprocessed_data, num_samples, num_components, verbose=True):
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
    covariance_matrix = np.cov(preprocessed_data)

    # Perform eigenvalue decomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues in descending order and get the sorted indices
    sorted_indices = np.argsort(eigenvalues)[::-1]

    # Select the top 'num_components' eigenvectors
    eigenvectors = eigenvectors[:, sorted_indices[:num_components]]
    eigenvalues = eigenvalues[sorted_indices[:num_components]]

    # Whitening: Create the whitening matrix
    scaling_factors = np.sqrt(eigenvalues)
    whitening_matrix = np.dot(eigenvectors, np.diag(1. / scaling_factors))

    # Apply the whitening matrix to the data
    whitened_data = np.dot(whitening_matrix.T, preprocessed_data)

    if verbose:
        print("Shape of whitening_matrix:", whitening_matrix.shape)
        print("Shape of whitened_data:", whitened_data.shape)
        print("jade -> Whitening completed")

    return whitened_data, whitening_matrix


    

    return whitened_data, whitening_matrix

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
    cumulant_matrices_storage = np.matrix(np.zeros([num_samples, num_samples * num_cumulant_matrices], dtype=np.float64))
    print("Shape of cumulant matrices storage {}", cumulant_matrices_storage.shape)

    # Ensure that the matrix has the correct dimensions
    expected_shape = (num_samples, num_samples * num_cumulant_matrices)
    print("Expected shape of cumulant matrices storage {}", expected_shape)
    assert cumulant_matrices_storage.shape == expected_shape, \
        f"Cumulant matrices storage has incorrect dimensions. Expected: {expected_shape}, Got: {cumulant_matrices_storage.shape}"

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
    print("function: compute cumulant matrix")
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

    # Initialize the cumulant matrix
    cumulant_matrix = np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[0]))

    # Vectorize computation for the cumulant matrix
    for i in range(preprocessed_data.shape[0]):
        Xii = component_signal[i] * component_signal[i]  # Square of the ith component
        for j in range(i, preprocessed_data.shape[0]):
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


def joint_diagonalization(cumulant_matrices, num_independent_components, num_samples):
    """
    Perform joint diagonalization on the cumulant matrices for ICA.

    Parameters:
    cumulant_matrices (numpy.ndarray): Storage matrix containing cumulant matrices.
    num_independent_components (int): Number of independent components to extract.
    num_samples (int): Number of samples in the dataset.

    Returns:
    numpy.ndarray: The rotation matrix for ICA.
    """
    print("function: joint diagonalization")

    # Input validation
    if not isinstance(cumulant_matrices, np.ndarray) or cumulant_matrices.ndim != 2:
        raise TypeError("cumulant_matrices must be a 2-dimensional numpy array.")

    if cumulant_matrices.shape[0] != num_samples or cumulant_matrices.shape[1] != num_samples * num_independent_components:
        raise ValueError("cumulant_matrices must have shape (num_samples, num_samples * num_independent_components)")

    if not isinstance(num_independent_components, int) or num_independent_components <= 0:
        raise ValueError("num_independent_components must be a positive integer.")

    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")

    # Initialize rotation matrix and diagonal/off-diagonal values
    rotation_matrix = np.eye(num_independent_components)
    on_diagonal_sum = off_diagonal_sum = 0.0

    convergence_threshold = 1.0e-6 / np.sqrt(num_samples)
    continue_diagonalization = True
    total_sweeps = 0
    total_updates = 0

    while continue_diagonalization:
        continue_diagonalization = False
        total_sweeps += 1
        current_sweep_updates = 0

        for component_p in range(num_independent_components - 1):
            for component_q in range(component_p + 1, num_independent_components):

                index_p = np.arange(component_p, num_samples * num_independent_components, num_independent_components)
                index_q = np.arange(component_q, num_samples * num_independent_components, num_independent_components)

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
                    givens_matrix = np.array([[cosine_theta, -sine_theta], [sine_theta, cosine_theta]])
                    component_pair = np.array([component_p, component_q])

                    rotation_matrix[:, component_pair] = np.dot(rotation_matrix[:, component_pair], givens_matrix)
                    cumulant_matrices[component_pair, :] = np.dot(givens_matrix.T, cumulant_matrices[component_pair, :])
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
    separating_matrix (numpy.ndarray): The separating matrix.

    Returns:
    numpy.ndarray: Sorted separating matrix.
    """
    print("function: sort separating matrix", type(separating_matrix))
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


def fix_matrix_signs(separating_matrix):
    """
    Adjust the signs of the rows of the separating matrix.

    Parameters:
    separating_matrix (numpy.ndarray): The separating matrix.

    Returns:
    numpy.ndarray: The separating matrix with adjusted signs.
    """

    print("function: fix matrix signs")
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


def jadeR(mixed_signal_matrix, num_components=None, verbose=True):
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

    preprocessed_data, input_data_type, num_components, num_samples = check_input(mixed_signal_matrix, num_components, verbose)

    # whitening & PCA
    whitened_data, whitened_matrix = perform_whitening(preprocessed_data, num_samples, num_components, verbose)

    if verbose:
        print("jade -> Estimating cumulant matrices")

    # Initialize the storage for cumulant matrices
    cumulant_matrices_storage, num_cumulant_matrices = initialize_cumulant_matrices_storage(num_samples, num_components)

    # Compute and store cumulant matrices
    for component_index in range(num_components):
        cumulant_matrix = compute_cumulant_matrix(whitened_data.T, num_samples, component_index, num_cumulant_matrices)
        print("Shape of cumulant matrix {}", cumulant_matrix.shape)

        # Store the computed cumulant matrix in the appropriate location
        storage_start_index = component_index * num_samples
        storage_end_index = storage_start_index + num_samples
        cumulant_matrices_storage[:, storage_start_index:storage_end_index] = cumulant_matrix


    rotation_matrix = joint_diagonalization(cumulant_matrices_storage, num_components, num_samples)
    print("Rotation matrix {}", rotation_matrix.shape)

    separating_matrix = rotation_matrix.T
    print("Separating matrix {}", separating_matrix.shape)

    # Apply the sorting and sign fixing
    if verbose:
        print("jade -> Sorting the components")
    separating_matrix = sort_separating_matrix(separating_matrix)
    print("Separating matrix after sort separating matrix function {}", separating_matrix.shape)
    print("Separating matrix type after sort separating matrix function {}", type(separating_matrix))

    if verbose:
        print("jade -> Fixing the signs")
    separating_matrix = fix_matrix_signs(separating_matrix)
    print("Separating matrix after fix matrix signs function {}", separating_matrix)

    return separating_matrix.astype(input_data_type), whitened_matrix

class JADE:
    def __init__(self, num_components=4):
        self.num_components = num_components
        self.unmixing_matrix = None
        self.whitening_matrix = None
        self.ica_jade_loadings = None
        self.ica_jade_corr = None
        self.ica_jade_ids = None

    def fit(self, mixed_signal_matrix):
        """
        Fit the JADE model to the data.

        Parameters:
        mixed_signal_matrix (numpy.ndarray): The mixed signal data matrix.

        Returns:
        numpy.ndarray: The unmixing matrix after applying JADE.
        """
        mixed_signal_matrix = np.array(mixed_signal_matrix)
        unmixing_matrix, self.whitening_matrix = jadeR(mixed_signal_matrix, num_components=self.num_components)
        print("shape of unmixing matrix ", unmixing_matrix.shape)

        # Adjust the sign of each row for better interpretability
        for i in range(unmixing_matrix.shape[0]):
            if np.abs(np.max(unmixing_matrix[i, :])) < np.abs(np.min(unmixing_matrix[i, :])):
                unmixing_matrix[i, :] *= -1

        self.unmixing_matrix = unmixing_matrix
        return unmixing_matrix

    def transform(self, mixed_signal_matrix):
        print("shape of mixed signal matrix", mixed_signal_matrix.shape)

        if self.unmixing_matrix is None or self.whitening_matrix is None:
            raise ValueError("Model has not been fit yet. Call 'fit' with training data.")

        # First, apply the whitening matrix to the input data
        whitened_data = np.dot(self.whitening_matrix.T, mixed_signal_matrix)

        # Check if transposition is needed
        if whitened_data.shape[0] != self.unmixing_matrix.shape[0]:
            whitened_data = whitened_data.T

        # Apply the unmixing matrix to the whitened data
        separated_signals = np.dot(self.unmixing_matrix, whitened_data).T
        return separated_signals

    def correlate_loadings(self, df, corrcols, icacols):
        """
        Find the correlation between loadings and a set of columns.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing data.
        corrcols (list): List of columns to correlate.
        icacols (list): List of ICA columns.

        Updates:
        self.ica_jade_corr: DataFrame of correlations.
        self.ica_jade_ids: Identifiers for the correlated loadings.
        """
        if self.unmixing_matrix is None:
            raise ValueError("Model has not been fit yet. Call 'fit' with training data.")

        corrdf = df.corr().drop(icacols, axis=1).drop(corrcols, axis=0)
        ica_jade_ids = []
        for i in corrdf.loc['ICA-JADE'].index:
            tmp = corrdf.loc[('ICA-JADE', i)]
            max_corr = np.max(tmp)
            match = tmp.values == max_corr
            matched_col = corrcols[np.where(match)[0][0]]
            ica_jade_ids.append(f"{matched_col} (r={np.round(max_corr, 1)})")

        self.ica_jade_corr = corrdf
        self.ica_jade_ids = ica_jade_ids

def get_dataset_frame(dataset_path):
    with open(dataset_path) as f:
        # Find index of last line starting with "#" and skip rows until then
        for i, line in enumerate(f):
            if not line.startswith("#"):
                break
        # Read CSV from that line - columns also start with "#"
        return pd.read_csv(dataset_path, skiprows=i-1)

def initial_preprocess(file_path):
    df = get_dataset_frame(file_path)

    # Clean up column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("# ", "")

    # Drop specific columns
    exclude = ["mean", "median"]
    first_five_shots = [f"shot{i}" for i in range(1, 6)]
    df.drop(exclude + first_five_shots, axis=1, inplace=True)

    # Apply any masking required
    for mask in masks:
        df = df.loc[~((df["wave"] >= mask[0]) & (df["wave"] <= mask[1]))]

    return df

def variance_based_selection(df):
    # Calculate variances without transposing
    variances = df.var(axis=1)
    threshold = variances.mean()

    # Select wavelengths based on the threshold
    selected_wavelengths = variances[variances > threshold].index
    df_selected = df.loc[selected_wavelengths]

    # Transpose the DataFrame for ICA
    df_transposed = df_selected.transpose()

    return df_transposed

def average_datasets(parent_directory):
    aggregated_data = []

    for subdirpath, subdirnames, _ in os.walk(parent_directory):
        for subdirname in subdirnames:
            dir_path = os.path.join(subdirpath, subdirname)
            csv_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.csv')]

            for file_path in csv_files:
                df = initial_preprocess(file_path)
                aggregated_data.append(df)

    if aggregated_data:
        # Concatenate all DataFrames along the rows
        combined_df = pd.concat(aggregated_data, axis=0)

        # Calculate the mean across rows for each wavelength
        # This assumes that the index of each DataFrame is the wavelength
        averaged_df = combined_df.groupby('wave').mean()
        
        return averaged_df
    else:
        return None


def main():
    parent_directory = "/home/iho/projects/thesis-chemcam/baseline/data/data/calib/calib_2015/1600mm/pls"

    # Averaging datasets
    combined_data = average_datasets(parent_directory)

    if combined_data is None:
        print("No data to process.")
        return

    processed_data = variance_based_selection(combined_data)
    #processed_data.to_csv("processed_data.csv")
    print("Processed data shape:", processed_data.shape)

    # Debug mode
    debug = False
    if debug:
        subset_data = processed_data.iloc[:4, :50]
        print("Subset data shape:", subset_data.shape)

        jade_model = JADE(num_components=4)
        jade_model.fit(subset_data.values)
        separated_signals = jade_model.transform(subset_data.values)
        print(separated_signals)

    else:
        # Normal processing with the full dataset
        num_features = processed_data.shape[1]
        jade_model = JADE(num_components=min(8, num_features))
        print("processed data:", processed_data.values)
        #return
        jade_model.fit(processed_data.values)
        separated_signals = jade_model.transform(processed_data.values)
        print("Separated signals shape:", separated_signals.shape)
        print("Separated signals:", separated_signals)

if __name__ == "__main__":
    main()
