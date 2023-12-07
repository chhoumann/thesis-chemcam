import os

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA

from ica.ica_2 import optimize_contrast_function, prewhiten
from ica.jade import JADE
from ica.postprocess import postprocess_data
from ica.preprocess import preprocess_data


def main():
    root_dir = "./data/data/calib/calib_2015/1600mm/ica"
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


def compute_correlations(df, separated_signals):
    # Convert separated signals to DataFrame for easier processing
    separated_df = pd.DataFrame(separated_signals.T)

    # Initialize a DataFrame to store correlation coefficients
    correlations = pd.DataFrame(
        index=df.columns,
        columns=[f"Component_{i}" for i in range(separated_signals.shape[0])],
    )

    # Calculate correlation for each component with each feature in the original dataset
    for i in range(separated_signals.shape[0]):
        for col in df.columns:
            correlation = np.corrcoef(df[col], separated_df[i])[0, 1]
            correlations.at[col, f"Component_{i}"] = correlation

    return correlations


def custom_ica(df: pd.DataFrame, num_components: int = 8):
    # print(df.transpose())
    # transformed = wmt.fit_transform(df.transpose())
    # Transpose data so that wavelengths are rows and intensity values are columns
    data = df.to_numpy().T

    # calculate mean and stdev for each feature
    # for i in range(data.shape[0]):
    #     mean = np.mean(data[i])
    #     stdev = np.std(data[i])
    #     print(f"mean: {mean:.2f}, stdev: {stdev:.2f}")

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

    correlations = compute_correlations(df, separated_signals)
    
    return separated_signals


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
    # correlation_matrix = np.corrcoef(separated_signals, processed_data.to_numpy().T, rowvar=False)

    # # Independence check
    # independence = np.allclose(corr, np.eye(corr.shape[0]), atol=0.1)

    # # # Sum of squares check
    # sum_of_squares = np.sum(corr**2, axis=1)
    # sum_of_squares_close_to_one = np.allclose(
    #     sum_of_squares, np.ones(sum_of_squares.shape[0])
    # )

    # print("Independence:", independence)
    # print("Sum of squares:\n", sum_of_squares)
    # print("Sum of squares close to one:", sum_of_squares_close_to_one)

    return separated_signals


if __name__ == "__main__":
    main()
