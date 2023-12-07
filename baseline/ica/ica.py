import os

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA

from ica.ica_2 import optimize_contrast_function, prewhiten
from ica.jade import JADE
from ica.postprocess import postprocess_data
from ica.preprocess import preprocess_data


def main():
    root_dir = "./data/data/calib/calib_2015/1600mm/pls"
    max_runs = 1
    runs = 0

    df = pd.DataFrame()

    for target_dir_name in os.listdir(root_dir):
        target_dir_path = os.path.join(root_dir, target_dir_name)

        data = preprocess_data(target_dir_path)
        separated_signals = run_ica(data, model="custom_jade")
        data = postprocess_data(target_dir_name, separated_signals)

        # df = df.append(data)
        df = pd.concat([df, data])

        runs += 1

        if runs >= max_runs:
            break

    df.to_csv("./ica_results.csv")


def custom_ica(df: pd.DataFrame, num_components: int = 8):
    # Transpose data so that wavelengths are rows and intensity values are columns
    data = df.to_numpy().T
    # Whitening
    X_whitened, whitening_matrix = prewhiten(data, num_components=num_components)

    # Optimization
    rotation_matrix = optimize_contrast_function(X_whitened)

    # Separation
    separated_signals = rotation_matrix @ X_whitened

    # Ensure separated_signals has the same number of columns as the original data
    if separated_signals.shape[1] != data.shape[1]:
        raise ValueError(
            "Separated signals and original data must have the same number of columns"
        )

    # Concatenate separated_signals and data for correlation computation
    concatenated_signals = np.vstack((separated_signals, data))

    print(separated_signals)

    # Now calculate the correlation matrix with the concatenated array
    correlation_matrix = np.corrcoef(concatenated_signals, rowvar=False)

    # Extract the relevant part of the correlation matrix
    # The correlation matrix will be (num_components + num_features) x (num_components + num_features)
    m = separated_signals.shape[0]
    n = data.shape[0]
    print("mn", m, n)
    correlation_Z_preprocessed_data = correlation_matrix[:m, m:]

    print("///////////////////")
    print(data.shape)
    print("///////////////////")
    print(correlation_Z_preprocessed_data)
    pd.DataFrame(correlation_Z_preprocessed_data).to_csv("./correlation.csv")
    print("///////////////////")
    print(separated_signals.shape)
    return separated_signals
    # # Whitening
    # data = df.to_numpy()
    # whitened_data, whitening_matrix = prewhiten(data)

    # # Optimization
    # rotation_matrix = optimize_contrast_function(whitened_data)

    # # Separation
    # separated_signals = rotation_matrix @ whitened_data
    # print(separated_signals.shape)

    # # Compute the correlation matrix between separated signals and original data
    # # Ensure that the separated signals are correctly shaped to match the data
    # separated_signals_reshaped = separated_signals[:num_components, :]

    # # Compute correlation only if the number of components is less than or equal to the number of features in data
    # if separated_signals_reshaped.shape[0] <= data.shape[0]:
    #     correlation_matrix_with_preprocessed_data = np.corrcoef(
    #         separated_signals_reshaped, data, rowvar=False
    #     )
    #     m = 45
    #     correlation_Z_preprocessed_data = correlation_matrix_with_preprocessed_data[
    #         :m, m:
    #     ]
    #     print(correlation_Z_preprocessed_data)


def run_ica(processed_data, model=""):
    num_components = 8
    separated_signals = None

    if model == "jade":
        jade_model = JADE(num_components)
        scores = jade_model.fit(processed_data)
        separated_signals = jade_model.transform(processed_data)
    elif model == "fastica":
        fastica_model = FastICA(
            n_components=num_components, random_state=0, max_iter=5000
        )
        separated_signals = fastica_model.fit_transform(processed_data)
    elif model == "custom_jade":
        separated_signals = custom_ica(processed_data, num_components=num_components)
    else:
        raise ValueError("Invalid model specified. Must be 'jade' or 'fastica'.")

    # Convert separated signals to NumPy array if it's not already
    # correlation_matrix = np.corrcoef(separated_signals, rowvar=False)

    # # Independence check
    # independence = np.allclose(correlation_matrix, np.eye(correlation_matrix.shape[0]), atol=0.1)

    # # Sum of squares check
    # sum_of_squares = np.sum(correlation_matrix**2, axis=1)
    # sum_of_squares_close_to_one = np.allclose(sum_of_squares, np.ones(sum_of_squares.shape[0]))

    # print("Independence:", independence)
    # print("Sum of squares:\n", sum_of_squares)
    # print("Sum of squares close to one:", sum_of_squares_close_to_one)

    return separated_signals


if __name__ == "__main__":
    main()
