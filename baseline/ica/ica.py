import os
from pathlib import Path

import dotenv
import mlflow
import numpy as np
import pandas as pd
import tqdm
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from ica.data_processing import ICASampleProcessor
from ica.jade import JADE
from lib.norms import Norm

mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = f"ICA_{pd.Timestamp.now().strftime('%m-%d-%y_%H%M%S')}"
mlflow.set_experiment(experiment_name)
mlflow.autolog()

env = dotenv.dotenv_values(dotenv.find_dotenv())
CALIB_DATA_PATH = env.get("DATA_PATH", "")
CALIB_COMP_PATH = env.get("COMPOSITION_DATA_PATH", "")
MLFLOW_TRACKING_URI = env.get("MLFLOW_TRACKING_URI", "")

if CALIB_DATA_PATH is None or CALIB_COMP_PATH is None or MLFLOW_TRACKING_URI is None:
    exit()


def main():
    ica_df_n1, compositions_df_n1 = get_train_data(num_components=8, norm=Norm.NORM_1)
    ica_df_n1 = ica_df_n1.abs()

    ica_df_n3, compositions_df_n3 = get_train_data(num_components=8, norm=Norm.NORM_3)
    ica_df_n3 = ica_df_n3.abs()

    assert len(ica_df_n1) == len(
        ica_df_n3
    ), "The number of rows in the two DataFrames must be equal."

    n_rows = len(ica_df_n1)

    # Split indices
    train_indices, test_indices = train_test_split(
        range(n_rows), test_size=0.2, random_state=42
    )

    # Split ica_df_n1
    ica_train_n1 = ica_df_n1.iloc[train_indices]
    ica_test_n1 = ica_df_n1.iloc[test_indices]

    # Split ica_df_n3
    ica_train_n3 = ica_df_n3.iloc[train_indices]
    ica_test_n3 = ica_df_n3.iloc[test_indices]

    # Assuming compositions_df_n1 and compositions_df_n3 are aligned with ica_df_n1 and ica_df_n3
    comp_train_n1 = compositions_df_n1.iloc[train_indices]
    comp_test_n1 = compositions_df_n1.iloc[test_indices]

    comp_train_n3 = compositions_df_n3.iloc[train_indices]
    comp_test_n3 = compositions_df_n3.iloc[test_indices]

    oxide_rmses = {}
    oxide_models = {
        "SiO2": {"law": "Log-square", "norm": Norm.NORM_1},
        "TiO2": {"law": "Geometric", "norm": Norm.NORM_3},
        "Al2O3": {"law": "Geometric", "norm": Norm.NORM_3},
        "FeOT": {"law": "Geometric", "norm": Norm.NORM_1},
        "MgO": {"law": "Exponential", "norm": Norm.NORM_1},
        "CaO": {"law": "Parabolic", "norm": Norm.NORM_1},
        "Na2O": {"law": "Parabolic", "norm": Norm.NORM_3},
        "K2O": {"law": "Geometric", "norm": Norm.NORM_3},
    }

    for oxide, info in tqdm.tqdm(oxide_models.items()):
        model_name = info["law"]
        norm = info["norm"]

        if model_name is None:
            print(f"No model specified for {oxide}. Skipping...")
            continue

        with mlflow.start_run(run_name=f"ICA_{oxide}"):
            print(f"Training model {model_name} for {oxide}...")

            X_train = ica_train_n1 if norm == Norm.NORM_1 else ica_train_n3
            y_train = (
                comp_train_n1[oxide] if norm == Norm.NORM_1 else comp_train_n3[oxide]
            )

            X_test = ica_test_n1 if norm == Norm.NORM_1 else ica_test_n3
            y_test = comp_test_n1[oxide] if norm == Norm.NORM_1 else comp_test_n3[oxide]

            if model_name == "Log-square":
                X_train = np.log(X_train**2)
                X_test = np.log(X_test**2)
            elif model_name == "Exponential":
                X_train = np.log(X_train)
                X_test = np.log(X_test)
            elif model_name == "Geometric":
                X_train = np.sqrt(X_train)
                X_test = np.sqrt(X_test)
            elif model_name == "Parabolic":
                X_train = X_train**2
                X_test = X_test**2

            model = LinearRegression()
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))

            oxide_rmses[oxide] = rmse
            mlflow.log_metric("RMSE", float(rmse))
            mlflow.log_params({"model": model_name, "oxide": oxide, "norm": norm.value})

            mlflow.sklearn.log_model(model, f"ICA_{oxide}")

    for oxide, rmse in oxide_rmses.items():
        print(f"RMSE for {oxide} with {oxide_models[oxide]['law']} model: {rmse}")


def get_train_data(num_components: int, norm: Norm) -> (pd.DataFrame, pd.DataFrame):
    calib_data_path = Path(CALIB_DATA_PATH)
    output_dir = Path(f"./data/data/jade/ica/norm{norm.value}")

    ica_df_csv_loc = Path(f"{output_dir}/ica_data.csv")
    compositions_csv_loc = Path(f"{output_dir}/composition_data.csv")

    if ica_df_csv_loc.exists() and compositions_csv_loc.exists():
        print("Preprocessed data found. Loading data...")
        ica_df = pd.read_csv(ica_df_csv_loc, index_col=0)
        compositions_df = pd.read_csv(compositions_csv_loc, index_col=0)
    else:
        print("No preprocessed data found. Creating and saving preprocessed data...")
        output_dir.mkdir(parents=True, exist_ok=True)
        ica_df, compositions_df = create_train_data(
            calib_data_path, num_components=num_components
        )
        ica_df.to_csv(ica_df_csv_loc)
        compositions_df.to_csv(compositions_csv_loc)
        print(
            f"Preprocessed data saved to {ica_df_csv_loc} and {compositions_csv_loc}.\n"
        )

    return ica_df, compositions_df


def create_train_data(
    calib_data_path: Path,
    ica_model: str = "jade",
    num_components: int = 8,
    norm: Norm = Norm.NORM_3,
) -> (pd.DataFrame, pd.DataFrame):
    composition_data_loc = CALIB_COMP_PATH

    ica_df = pd.DataFrame()
    compositions_df = pd.DataFrame()

    for sample_name in tqdm.tqdm(os.listdir(calib_data_path)):
        processor = ICASampleProcessor(sample_name, num_components)

        if not processor.try_load_composition_df(composition_data_loc):
            continue

        print(f"Processing {sample_name}...")

        processor.preprocess(calib_data_path, norm)

        # Run ICA and get the estimated sources
        ica_estimated_sources = run_ica(
            processor.df, model=ica_model, num_components=num_components
        )

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


def run_ica(
    df: pd.DataFrame, model: str = "jade", num_components: int = 8
) -> np.ndarray:
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
        fastica_model = FastICA(
            n_components=num_components, whiten="unit-variance", max_iter=5000
        )
        estimated_sources = fastica_model.fit_transform(df)
    else:
        raise ValueError("Invalid model specified. Must be 'jade' or 'fastica'.")

    return estimated_sources


if __name__ == "__main__":
    main()
