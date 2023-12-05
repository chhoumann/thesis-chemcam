import os
import pandas as pd
import numpy as np
from ica.preprocess import preprocess_data
from ica.postprocess import postprocess_data
from ica.jade import JADE
from sklearn.decomposition import FastICA


def main():
    root_dir = "./data/calib_2015/1600mm/pls"
    max_runs = 5
    runs = 0

    df = pd.DataFrame()

    for target_dir_name in os.listdir(root_dir):
        target_dir_path = os.path.join(root_dir, target_dir_name)

        data = preprocess_data(target_dir_path)
        separated_signals = run_ica(data, model="fastica")
        data = postprocess_data(target_dir_name, separated_signals)

        #df = df.append(data)
        df = pd.concat([df, data])

        runs += 1

        if runs >= max_runs:
            break

    df.to_csv("./ica_results.csv")


def run_ica(processed_data, model=""):
    num_components = 8
    separated_signals = None

    if model == "jade":
        jade_model = JADE(num_components=num_components)
        scores = jade_model.fit(processed_data)
        separated_signals = jade_model.transform(processed_data)
    elif model == "fastica":
        fastica_model = FastICA(n_components=num_components, random_state=0, max_iter=5000)
        separated_signals = fastica_model.fit_transform(processed_data)
    else:
        raise ValueError("Invalid model specified. Must be 'jade' or 'fastica'.")

    # Convert separated signals to NumPy array if it's not already
    correlation_matrix = np.corrcoef(separated_signals, rowvar=False)

    # Independence check
    independence = np.allclose(correlation_matrix, np.eye(correlation_matrix.shape[0]), atol=0.1)

    # Sum of squares check
    sum_of_squares = np.sum(correlation_matrix**2, axis=1)
    sum_of_squares_close_to_one = np.allclose(sum_of_squares, np.ones(sum_of_squares.shape[0]))

    print("Independence:", independence)
    print("Sum of squares:\n", sum_of_squares)
    print("Sum of squares close to one:", sum_of_squares_close_to_one)

    return separated_signals


def run_jade(processed_data):
    num_components = 8
    num_signals = processed_data.shape[1]
    jade_model = JADE(num_components=num_components)

    jade_model.fit(processed_data)
    separated_signals = jade_model.transform(processed_data) # Note: separated signals are "scores", oftenf denoted S
    # separated_signals = (separated_signals - np.mean(separated_signals, axis=0)) / np.std(separated_signals, axis=0)

    separated_signals_array = np.array(separated_signals)
    print("separated signals:\n", separated_signals_array)
    processed_data_array = np.array(processed_data)

    num_components = separated_signals_array.shape[1]  # Number of components in S (8)
    num_signals = processed_data_array.shape[1]  # Number of signals in X (462)

    # Initialize an empty matrix to store correlations
    correlation_matrix = np.zeros((num_components, num_signals))

    # Compute pairwise correlations
    for i in range(num_components):
        for j in range(num_signals):
            corr = np.corrcoef(separated_signals_array[:, i], processed_data_array[:, j])[0, 1]
            correlation_matrix[i, j] = corr


    squared_correlations = correlation_matrix ** 2
    sum_of_squares = np.sum(squared_correlations, axis=1)
    print("sum of squares:\n", sum_of_squares)


    # Assuming you want to correlate all original features with the independent components
    # corrcols = processed_data.columns.tolist()  # All columns in processed_data
    # icacols = ['IC' + str(i) for i in range(1, jade_model.num_components + 1)]  # List of independent components


    # Add the separated signals to the processed data for correlation
    # for i, col in enumerate(icacols):
        # processed_data[col] = separated_signals[:, i]

    # Perform correlation
    # jade_model.correlate_loadings(processed_data, corrcols, icacols)
    # corr = jade_model.ica_jade_corr
    # print("ICA-JADE Correlations:\n", corr)

    # sums = []
    # means = corr.mean()

    # for i, col in enumerate(corr.columns):
    #     sum = 0

    #     for row in corr[col]:
    #         sum += (row - means[i]) ** 2

    #     sums.append(sum)

    # sum_of_squares = 0

    # for sum in sums:
    #     sum_of_squares += sum

    # # sum_of_squares = (df**2).sum(axis=1)(corr**2).sum().sum()
    # print("the final sum:", sum_of_squares)

    # Print or inspect the correlation results
    # print("ICA-JADE Correlations:", jade_model.ica_jade_corr)
    # print("ICA-JADE IDs:", jade_model.ica_jade_ids)

    return separated_signals


if __name__ == "__main__":
    main()
