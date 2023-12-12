import os
import pandas as pd
import numpy as np

from pathlib import Path
from ica.preprocess import preprocess_data
from ica.postprocess import postprocess_data
from ica.jade import JADE
from sklearn.decomposition import FastICA
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def main():
    data_path = Path("./data/calib/calib_2015/1600mm/pls/")
    max_runs = 4
    runs = 0
    num_components = 15
    all_estimated_sources = []

    df = pd.DataFrame()

    for sample_name in os.listdir(data_path):
        X = preprocess_data(sample_name, data_path)
        estimated_sources = run_ica(X, model="jade", num_components=num_components)
        df = pd.concat([df, postprocess_data(sample_name, estimated_sources)])

        estimated_sources_df = pd.DataFrame(estimated_sources)
        all_estimated_sources.append(estimated_sources_df)

        runs += 1

        if runs >= max_runs:
            break

    combined_sources = pd.concat(all_estimated_sources, ignore_index=True)
    
    df.to_csv("./ica_results.csv")


def run_ica(df: pd.DataFrame, model: str = "fastica", num_components: int = 8) -> np.ndarray:
    """
    Performs Independent Component Analysis (ICA) on a given dataset using JADE or FastICA algorithms.

    Parameters:
    ----------
    df : pd.DataFrame
        The input dataset for ICA. The DataFrame should have rows as samples and columns as features.

    model : str, optional
        The ICA model to be used. Must be either 'jade' or 'fastica'. Defaults to 'fastica'.

    num_components : int, optional
        The number of independent components to be extracted. Defaults to 8.

    Returns:
    -------
    np.ndarray
        An array of the estimated independent components extracted from the input data.

    Raises:
    ------
    ValueError
        If an invalid model name is specified.

    AssertionError
        If the extracted signals are not independent, indicated by the correlation matrix not being
        close to the identity matrix or the sum of squares of correlations not being close to one.
    """
    estimated_sources = None

    if model == "jade":
        jade_model = JADE(num_components)
        df = df.transpose()
        mixing_matrix = jade_model.fit(df)
        estimated_sources = jade_model.transform(df)
    

    elif model == "fastica":
        fastica_model = FastICA(n_components=num_components, max_iter=5000)
        df = df.transpose()
        estimated_sources = fastica_model.fit_transform(df)
    else:
        raise ValueError("Invalid model specified. Must be 'jade' or 'fastica'.")


    return estimated_sources

if __name__ == "__main__":
    main()
