import os
import pandas as pd
import numpy as np

from pathlib import Path
from ica.data_processing import ICASampleProcessor
from ica.jade import JADE
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lib.reproduction import major_oxides
from scipy.stats import median_abs_deviation


def remove_outliers(df, threshold=25):
    # Calculate median and MAD for each column
    medians = df.median()
    mad = df.apply(median_abs_deviation)

    # Identify outliers: a row is an outlier if any of its values are outliers in their respective column
    outlier_mask = ((df - medians).abs() > threshold * mad).any(axis=1)

    print(f"Removing {outlier_mask.sum()} outliers from {len(df)} samples.")

    # Remove outliers and return the cleaned DataFrame
    return df[~outlier_mask]


def fit_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, predictions))


def main():
    ica_df, compositions_df = get_train_data(num_components=8)
    ica_train, ica_test, comp_train, comp_test = train_test_split(ica_df, compositions_df, test_size=0.2, random_state=42)
    max_iterations = 10

    oxide_rmses = {}
    oxide_models = {
        "SiO2": "Log-square",
        "TiO2": "Geometric",
        "Al2O3": "Geometric",
        "FeOT": "Geometric",
        "MgO": "Exponential",
        "CaO": "Parabolic",
        "Na2O": "Parabolic",
        "K2O": "Geometric"
    }

    for oxide in major_oxides:
        model_name = oxide_models[oxide]

        if model_name is None:
            print(f"No model specified for {oxide}. Skipping...")
            continue

        print(f"Training model {model_name} for {oxide}...")

        X_train = ica_train
        y_train = comp_train[oxide]

        X_test = ica_test
        y_test = comp_test[oxide]

        if model_name == "Log-square":
            y_train = np.log(y_train**2)
            y_test = np.log(y_test**2)
        elif model_name == "Exponential":
            y_train = np.log(y_train)
            y_test = np.log(y_test)
        elif model_name == "Geometric":
            # Presence of negative values prevents log transformation of data from working, so we add a constant for now
            X_train = np.log(X_train + 1)
            X_test = np.log(X_test + 1)
        elif model_name == "Parabolic":
            X_train = np.column_stack((X_train, X_train**2))
            X_test = np.column_stack((X_test, X_test**2))

        model = LinearRegression()
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        oxide_rmses[oxide] = rmse

        # test_rmse = np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))
        # min_rmse = test_rmse
        # no_improvement_count = 0
        # tolerance = 3  # Number of iterations to continue without improvement

        # for i in range(max_iterations):
        #     X_train_filtered = remove_outliers(X_train)
        #     y_train_filtered = comp_train.loc[X_train_filtered.index][oxide]

        #     if len(X_train_filtered) < len(X_train):
        #         new_model = LinearRegression()
        #         new_model.fit(X_train_filtered, y_train_filtered)

        #         new_rmse = np.sqrt(mean_squared_error(y_test, new_model.predict(X_test)))

        #         if new_rmse < min_rmse:
        #             print(f"RMSE improved from {min_rmse} to {new_rmse} for {oxide}.")
        #             min_rmse = new_rmse
        #             best_model = new_model
        #             X_train = X_train_filtered
        #             no_improvement_count = 0
        #         else:
        #             no_improvement_count += 1
        #             if no_improvement_count >= tolerance:
        #                 print(f"No improvement for {oxide} after {i+1} iterations. Stopping early.")
        #                 break


        # oxide_rmses[oxide] = min_rmse

    for oxide, rmse in oxide_rmses.items():
        print(f"RMSE for {oxide} with {oxide_models[oxide]} model: {rmse}")



def get_train_data(num_components: int) -> (pd.DataFrame, pd.DataFrame):
    calib_data_path = Path("./data/calib_2015/1600mm/pls")

    ica_df_csv_loc = Path("./data/ica_data.csv")
    compositions_csv_loc = Path("./data/composition_data.csv")

    if ica_df_csv_loc.exists() and compositions_csv_loc.exists():
        ica_df = pd.read_csv(ica_df_csv_loc, index_col=0)
        compositions_df = pd.read_csv(compositions_csv_loc, index_col=0)
    else:
        print("No preprocessed data found. Creating and saving preprocessed data...")
        ica_df, compositions_df = create_train_data(calib_data_path, num_components=num_components)
        ica_df.to_csv(ica_df_csv_loc)
        compositions_df.to_csv(compositions_csv_loc)
        print(f"Preprocessed data saved to {ica_df_csv_loc} and {compositions_csv_loc}.\n")

    return ica_df, compositions_df


def create_train_data(calib_data_path: Path, ica_model: str = "jade", num_components: int = 8) -> (pd.DataFrame, pd.DataFrame):
    composition_data_loc = "./data/data/ccam_calibration_compositions.csv"

    ica_df = pd.DataFrame()
    compositions_df = pd.DataFrame()

    for sample_name in os.listdir(calib_data_path):
        processor = ICASampleProcessor(sample_name, num_components)

        if not processor.try_load_composition_df(composition_data_loc):
            continue

        print(f"Processing {sample_name}...")

        processor.preprocess(calib_data_path)

        # Run ICA and get the estimated sources
        ica_estimated_sources = run_ica(processor.df, model=ica_model, num_components=num_components)

        # Postprocess the data
        processor.postprocess(ica_estimated_sources)

        # Aggregate the ICA results and composition data to their respective DataFrames
        compositions_df = pd.concat([compositions_df, processor.composition_df])
        ica_df = pd.concat([ica_df, processor.ic_wavelengths])

    # Set the index and column names for the DataFrames
    ica_df.index.name = "target"
    ica_df.columns.name = "wavelegths"

    compositions_df.index.name = "target"
    compositions_df.columns.name = "oxide"

    return ica_df, compositions_df


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
        # cols = df.columns
        jade_model = JADE(num_components)
        mixing_matrix = jade_model.fit(df)
        estimated_sources = jade_model.transform(df)
    elif model == "fastica":
        fastica_model = FastICA(n_components=num_components, max_iter=5000)
        estimated_sources = fastica_model.fit_transform(df)
    else:
        raise ValueError("Invalid model specified. Must be 'jade' or 'fastica'.")

    return estimated_sources


if __name__ == "__main__":
    main()
