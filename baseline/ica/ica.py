import os
from pathlib import Path
from typing import Tuple

import dotenv
import mlflow
import numpy as np
import pandas as pd
import tqdm
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from ica.data_processing import ICASampleProcessor
from ica.jade import JADE
from lib.norms import Norm

TEST = True
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = f"""ICA_{
    'TEST' if TEST else 'TRAIN'
    }_{pd.Timestamp.now().strftime('%m-%d-%y_%H%M%S')}"""
# experiment_name = "ICA_Train_Test_Split"
mlflow.set_experiment(experiment_name)
mlflow.autolog()

env = dotenv.dotenv_values(dotenv.find_dotenv())
CALIB_DATA_PATH = env.get("DATA_PATH", "")
CALIB_COMP_PATH = env.get("COMPOSITION_DATA_PATH", "")
MLFLOW_TRACKING_URI = env.get("MLFLOW_TRACKING_URI", "")

if CALIB_DATA_PATH is None or CALIB_COMP_PATH is None or MLFLOW_TRACKING_URI is None:
    exit()

model_configs = {
    "SiO2": {"law": "Log-square", "norm": Norm.NORM_1},
    "TiO2": {"law": "Geometric", "norm": Norm.NORM_3},
    "Al2O3": {"law": "Geometric", "norm": Norm.NORM_3},
    "FeOT": {"law": "Geometric", "norm": Norm.NORM_1},
    "MgO": {"law": "Exponential", "norm": Norm.NORM_1},
    "CaO": {"law": "Parabolic", "norm": Norm.NORM_1},
    "Na2O": {"law": "Parabolic", "norm": Norm.NORM_3},
    "K2O": {"law": "Geometric", "norm": Norm.NORM_3},
}


def train(ica_df_n1, ica_df_n3, compositions_df_n1, compositions_df_n3):
    # This is not suited for more than a single location per sample.
    id_col = ica_df_n1["id"]
    ica_df_n1.drop(columns=["id"], inplace=True)
    ica_df_n3.drop(columns=["id"], inplace=True)

    oxide_rmses = {}

    target_predictions = pd.DataFrame(compositions_df_n1.index)
    target_predictions["id"] = id_col

    for oxide, info in tqdm.tqdm(model_configs.items()):
        model_name = info["law"]
        norm = info["norm"]

        if model_name is None:
            print(f"No model specified for {oxide}. Skipping...")
            continue

        with mlflow.start_run(run_name=f"ICA_TRAIN_{oxide}"):
            print(f"Training model {model_name} for {oxide}...")

            X_train = ica_df_n1 if norm == Norm.NORM_1 else ica_df_n3
            y_train = (
                compositions_df_n1[oxide]
                if norm == Norm.NORM_1
                else compositions_df_n3[oxide]
            )

            X_test = ica_df_n1 if norm == Norm.NORM_1 else ica_df_n3
            y_test = (
                compositions_df_n1[oxide]
                if norm == Norm.NORM_1
                else compositions_df_n3[oxide]
            )

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

            oxide_prediction_path = Path("./data/data/jade/ica/predictions_new")
            oxide_prediction_path.mkdir(parents=True, exist_ok=True)

            target_predictions[oxide] = pd.Series(pred)

            pd.Series(pred).to_csv(oxide_prediction_path / f"{oxide}_pred1.csv")
            mlflow.log_metric("RMSE", float(rmse))
            mlflow.log_params({"model": model_name, "oxide": oxide, "norm": norm.value})

            mlflow.sklearn.log_model(model, f"ICA_{oxide}")

    for oxide, rmse in oxide_rmses.items():
        print(f"RMSE for {oxide} with {model_configs[oxide]['law']} model: {rmse}")

    target_predictions.to_csv("./data/data/jade/ica/TRAIN_tar_pred.csv")


def test(ica_df_n1, ica_df_n3, compositions_df_n1, compositions_df_n3):
    models = {}
    experiment_id = "196830081326801404"
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    for _, run in runs.iterrows():
        run_id = run["run_id"]
        oxide_value = run["params.oxide"]  # Assuming 'oxide' is stored as a parameter

        # Fetch the model artifact if it's a scikit-learn model
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        for artifact in artifacts:
            if "model" in artifact.path.lower():
                model_uri = f"runs:/{run_id}/{artifact.path}"
                model = mlflow.sklearn.load_model(model_uri)
                models[oxide_value] = model

    id_col = ica_df_n1["id"]
    ica_df_n1.drop(columns=["id"], inplace=True)
    ica_df_n3.drop(columns=["id"], inplace=True)

    oxide_rmses = {}
    oxide_preds = {}
    for oxide, info in tqdm.tqdm(model_configs.items()):
        model_name = info["law"]
        norm = info["norm"]

        if model_name is None:
            print(f"No model specified for {oxide}. Skipping...")
            continue

        with mlflow.start_run(run_name=f"ICA_TEST_{oxide}"):
            print(f"Testing model {model_name} for {oxide}...")

            X_test = ica_df_n1 if norm == Norm.NORM_1 else ica_df_n3
            y_test = (
                compositions_df_n1[oxide]
                if norm == Norm.NORM_1
                else compositions_df_n3[oxide]
            )

            if model_name == "Log-square":
                X_test = np.log(X_test**2)
            elif model_name == "Exponential":
                X_test = np.log(X_test)
            elif model_name == "Geometric":
                X_test = np.sqrt(X_test)
            elif model_name == "Parabolic":
                X_test = X_test**2

            pred = models[oxide].predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))

            oxide_rmses[oxide] = rmse

            oxide_prediction_path = Path("./data/data/jade/ica/predictions_new")
            oxide_prediction_path.mkdir(parents=True, exist_ok=True)

            oxide_preds[oxide] = pred

            pd.Series(pred).to_csv(oxide_prediction_path / f"{oxide}_pred1.csv")
            mlflow.log_metric("RMSE", float(rmse))
            mlflow.log_params({"model": model_name, "oxide": oxide, "norm": norm.value})

    for oxide, rmse in oxide_rmses.items():
        print(f"RMSE for {oxide} with {model_configs[oxide]['law']} model: {rmse}")

    target_predictions = pd.DataFrame()
    target_predictions["ID"] = id_col
    for oxide, pred in oxide_preds.items():
        target_predictions[oxide] = pd.Series(pred, index=target_predictions.index)

    target_predictions.to_csv("./data/data/jade/ica/tar_pred.csv")


def main():
    exclude_columns_abs = ["id"]

    ica_df_n1, compositions_df_n1 = get_data(num_components=8, norm=Norm.NORM_1)
    temp_df = ica_df_n1.drop(columns=exclude_columns_abs)
    temp_df = temp_df.abs()
    ica_df_n1_abs = pd.concat([ica_df_n1[exclude_columns_abs], temp_df], axis=1)

    ica_df_n3, compositions_df_n3 = get_data(num_components=8, norm=Norm.NORM_3)
    temp_df = ica_df_n3.drop(columns=exclude_columns_abs)
    temp_df = temp_df.abs()
    ica_df_n3_abs = pd.concat([ica_df_n3[exclude_columns_abs], temp_df], axis=1)

    assert len(ica_df_n1_abs) == len(
        ica_df_n3_abs
    ), "The number of rows in the two DataFrames must be equal."

    assert (
        ica_df_n1_abs["id"] == ica_df_n1_abs["id"]
    ).all(), "The IDs of the two DataFrames must be aligned."

    if TEST:
        test(ica_df_n1_abs, ica_df_n3_abs, compositions_df_n1, compositions_df_n3)
    else:
        train(ica_df_n1_abs, ica_df_n3_abs, compositions_df_n1, compositions_df_n3)


def get_data(num_components: int, norm: Norm) -> (pd.DataFrame, pd.DataFrame):
    calib_data_path = Path(CALIB_DATA_PATH)
    output_dir = Path(f"./data/data/jade/ica/norm{norm.value}{'-test' if TEST else ''}")

    ica_df_csv_loc = Path(f"{output_dir}/ica_data.csv")
    compositions_csv_loc = Path(f"{output_dir}/composition_data.csv")

    if ica_df_csv_loc.exists() and compositions_csv_loc.exists():
        print("Preprocessed data found. Loading data...")
        ica_df = pd.read_csv(ica_df_csv_loc, index_col=0)
        compositions_df = pd.read_csv(compositions_csv_loc, index_col=0)
    else:
        print("No preprocessed data found. Creating and saving preprocessed data...")
        output_dir.mkdir(parents=True, exist_ok=True)
        ica_df, compositions_df = create_processed_data(
            calib_data_path, num_components=num_components, norm=norm
        )
        ica_df.to_csv(ica_df_csv_loc)
        compositions_df.to_csv(compositions_csv_loc)
        print(
            f"Preprocessed data saved to {ica_df_csv_loc} and {compositions_csv_loc}.\n"
        )

    return ica_df, compositions_df


def create_processed_data(
    calib_data_path: Path,
    ica_model: str = "jade",
    num_components: int = 8,
    norm: Norm = Norm.NORM_3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    composition_data_loc = CALIB_COMP_PATH

    ica_df = pd.DataFrame()
    compositions_df = pd.DataFrame()

    test_train_split_idx = pd.read_csv("./train_test_split.csv")

    not_in_set = []
    missing = []

    desired_dataset = "test" if TEST else "train"
    for sample_name in tqdm.tqdm(list(os.listdir(calib_data_path))):
        split_info_sample_row = test_train_split_idx[
            test_train_split_idx["sample_name"] == sample_name
        ]["train_test"]

        if split_info_sample_row.empty:
            print(
                f"""No split info found for {
                sample_name}. Likely has missing data or is not used in calib2015."""
            )
            missing.append(sample_name)
            continue

        isSampleNotInSet = split_info_sample_row.values[0] != desired_dataset

        if isSampleNotInSet:
            # print(f"Skipping {sample_name}... Not in {desired_dataset} set.")
            not_in_set.append(sample_name)
            continue

        processor = ICASampleProcessor(sample_name, num_components)

        if not processor.try_load_composition_df(
            composition_data_loc=composition_data_loc
        ):
            print(f"No composition data found for {sample_name}. Skipping.")
            missing.append(sample_name)
            continue

        processor.preprocess(calib_data_path, norm)

        # Run ICA and get the estimated sources
        ica_estimated_sources = run_ica(
            processor.df, model=ica_model, num_components=num_components
        )

        # Postprocess the data
        processor.postprocess(ica_estimated_sources)

        # Aggregate the ICA results and composition data to their respective DataFrames
        compositions_df = pd.concat([compositions_df, processor.composition_df])
        # Add the sample ID to the ICA DataFrame
        processor.ic_wavelengths["id"] = processor.sample_id
        ica_df = pd.concat([ica_df, processor.ic_wavelengths])

    # Set the index and column names for the DataFrames
    ica_df.index.name = "target"
    ica_df.columns.name = "wavelegths"

    compositions_df.index.name = "target"
    compositions_df.columns.name = "oxide"

    print(f"#{len(not_in_set)} | Samples not in {desired_dataset} set: {not_in_set}")
    print(f"#{len(missing)} | Samples missing composition data: {missing}")
    print(f"#{len(ica_df)} | Samples with ICA data: {ica_df.index.unique()}")
    print(
        f"#{len(compositions_df)} | Samples with composition data: {compositions_df.index.unique()}"
    )

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
        mixing_matrix = jade_model.fit(X=df)
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
