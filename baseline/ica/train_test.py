import numpy as np
import pandas as pd
import mlflow
import tqdm

from lib.full_flow_dataloader import load_train_test_data
from lib.config import AppConfig
from lib.norms import Norm
from sklearn.linear_model import LinearRegression
from lib.data_handling import CompositionData
from sklearn.metrics import mean_squared_error
from typing import Dict, Tuple
from mlflow.entities import Experiment
from train_test_split import get_all_samples

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

config = AppConfig()


def train(
    ica_df_n1: pd.DataFrame,
    ica_df_n3: pd.DataFrame,
    compositions_df_n1: pd.DataFrame,
    compositions_df_n3: pd.DataFrame,
) -> Experiment:
    experiment = _setup_mlflow_experiment("TRAIN")

    id_col, sample_name_col = _setup(ica_df_n1, ica_df_n3)

    target_predictions = pd.DataFrame(compositions_df_n1.index)
    target_predictions["ID"] = id_col
    target_predictions["Sample Name"] = sample_name_col

    oxide_rmses = {}

    for oxide, info in tqdm.tqdm(model_configs.items()):
        model_name = info["law"]
        norm = info["norm"]

        if model_name is None:
            print(f"No model specified for {oxide}. Skipping...")
            continue

        ica_df = ica_df_n1 if norm == Norm.NORM_1 else ica_df_n3
        compositions_df = (
            compositions_df_n1 if norm == Norm.NORM_1 else compositions_df_n3
        )

        with mlflow.start_run(run_name=f"ICA_TRAIN_{oxide}"):
            print(f"Training model {model_name} for {oxide}...")

            X_train, X_test, y_train, y_test = (
                ica_df,
                ica_df,
                compositions_df[oxide],
                compositions_df[oxide],
            )

            # Transform the data
            X_train = _transform(X_train, model_name)
            X_test = _transform(X_test, model_name)

            X_train.fillna(0, inplace=True)
            X_test.fillna(0, inplace=True)

            model = LinearRegression()
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))

            oxide_rmses[oxide] = rmse

            mlflow.log_metric("RMSE", float(rmse))
            mlflow.log_params({"model": model_name, "oxide": oxide, "norm": norm.value})
            mlflow.sklearn.log_model(model, f"ICA_{oxide}")

    _print_rmses(oxide_rmses)

    return experiment


def test(
    ica_df_n1: pd.DataFrame,
    ica_df_n3: pd.DataFrame,
    compositions_df_n1: pd.DataFrame,
    compositions_df_n3: pd.DataFrame,
    experiment_id: str,
) -> pd.DataFrame:
    models = _get_ica_models(experiment_id)
    _setup_mlflow_experiment("TEST")

    id_col, sample_name_col = _setup(ica_df_n1, ica_df_n3)

    target_predictions = pd.DataFrame()
    target_predictions["ID"] = id_col
    target_predictions["Sample Name"] = sample_name_col

    oxide_rmses = {}
    oxide_preds = {}

    for oxide, info in tqdm.tqdm(model_configs.items()):
        model_name = info["law"]
        norm = info["norm"]

        if model_name is None:
            print(f"No model specified for {oxide}. Skipping...")
            continue

        ica_df = ica_df_n1 if norm == Norm.NORM_1 else ica_df_n3
        compositions_df = (
            compositions_df_n1 if norm == Norm.NORM_1 else compositions_df_n3
        )

        with mlflow.start_run(run_name=f"ICA_TEST_{oxide}"):
            print(f"Testing model {model_name} for {oxide}...")

            X_test, y_test = ica_df, compositions_df[oxide]

            # Transform the data
            X_test = _transform(X_test, model_name)

            pred = models[oxide].predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))

            oxide_rmses[oxide] = rmse
            oxide_preds[oxide] = pred

            mlflow.log_metric("RMSE", float(rmse))
            mlflow.log_params({"model": model_name, "oxide": oxide, "norm": norm.value})

    _print_rmses(oxide_rmses)

    for oxide, pred in oxide_preds.items():
        target_predictions[oxide] = pd.Series(pred, index=target_predictions.index)

    return target_predictions


def _setup_mlflow_experiment(mode: str) -> Experiment:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    _experiment_id = mlflow.create_experiment(
        f"ICA_{mode}_{pd.Timestamp.now().strftime('%m-%d-%y_%H%M%S')}"
    )
    experiment = mlflow.set_experiment(experiment_id=_experiment_id)
    mlflow.autolog()

    return experiment


def _setup(
    ica_df_n1: pd.DataFrame, ica_df_n3: pd.DataFrame
) -> Tuple[pd.Series, pd.Series]:
    id_col = ica_df_n1["ID"]
    sample_name_col = ica_df_n1["Sample Name"]

    ica_df_n1.drop(columns=["ID", "Sample Name"], inplace=True)
    ica_df_n3.drop(columns=["ID", "Sample Name"], inplace=True)

    return id_col, sample_name_col


def _transform(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if model_name == "Exponential" or model_name == "Log-square":
        if model_name == "Log-square":
            df = df**2

        # turn -inf to 0
        df = np.log(df)
        df[df == -np.inf] = 0
    elif model_name == "Geometric":
        df = np.sqrt(df)
    elif model_name == "Parabolic":
        df = df**2

    return df


def _print_rmses(oxide_rmses: Dict[str, float]):
    for oxide, rmse in oxide_rmses.items():
        print(f"RMSE for {oxide} with {model_configs[oxide]['law']} model: {rmse}")


def _get_ica_models(experiment_id: str) -> Dict[str, LinearRegression]:
    models = {}
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    for _, run in runs.iterrows():  # type: ignore
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

    return models
