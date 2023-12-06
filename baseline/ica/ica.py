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
        separated_signals = run_ica(data, model="jade")
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
        jade_model = JADE(num_components)
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


if __name__ == "__main__":
    main()
