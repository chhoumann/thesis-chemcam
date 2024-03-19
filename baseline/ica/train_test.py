import numpy as np
import pandas as pd
import mlflow
import tqdm

from lib.config import AppConfig
from lib.norms import Norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from typing import Dict

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


def train(ica_df_n1, ica_df_n3, compositions_df_n1, compositions_df_n3):
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    _experiment_id = mlflow.create_experiment(
        f"ICA_TRAIN_{pd.Timestamp.now().strftime('%m-%d-%y_%H%M%S')}"
    )
    experiment = mlflow.set_experiment(experiment_id=_experiment_id)
    mlflow.autolog()

    id_col = ica_df_n1["ID"]
    sample_name_col = ica_df_n1["Sample Name"]

    ica_df_n1.drop(columns=["ID", "Sample Name"], inplace=True)
    ica_df_n3.drop(columns=["ID", "Sample Name"], inplace=True)

    oxide_rmses = {}

    target_predictions = pd.DataFrame(compositions_df_n1.index)
    target_predictions["ID"] = id_col
    target_predictions["Sample Name"] = sample_name_col

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

            if model_name == "Exponential" or model_name == "Log-square":
                if model_name == "Log-square":
                    X_train = X_train**2
                    X_test = X_test**2

                X_train = np.log(X_train)
                X_test = np.log(X_test)

                # turn -inf to 0
                X_train[X_train == -np.inf] = 0
                X_test[X_test == -np.inf] = 0
            elif model_name == "Geometric":
                X_train = np.sqrt(X_train)
                X_test = np.sqrt(X_test)
            elif model_name == "Parabolic":
                X_train = X_train**2
                X_test = X_test**2

            X_train.fillna(0, inplace=True)
            X_test.fillna(0, inplace=True)

            model = LinearRegression()
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))

            oxide_rmses[oxide] = rmse
            target_predictions[oxide] = pd.Series(pred)

            mlflow.log_metric("RMSE", float(rmse))
            mlflow.log_params({"model": model_name, "oxide": oxide, "norm": norm.value})

            mlflow.sklearn.log_model(model, f"ICA_{oxide}")

    for oxide, rmse in oxide_rmses.items():
        print(f"RMSE for {oxide} with {model_configs[oxide]['law']} model: {rmse}")

    return experiment


def test(
    ica_df_n1, ica_df_n3, compositions_df_n1, compositions_df_n3, experiment_id: str
):
    models = get_ica_models(experiment_id)

    id_col = ica_df_n1["ID"]
    sample_name_col = ica_df_n1["Sample Name"]

    ica_df_n1.drop(columns=["ID", "Sample Name"], inplace=True)
    ica_df_n3.drop(columns=["ID", "Sample Name"], inplace=True)

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    _experiment_id = mlflow.create_experiment(
        f"ICA_TEST_{pd.Timestamp.now().strftime('%m-%d-%y_%H%M%S')}"
    )
    mlflow.set_experiment(experiment_id=_experiment_id)
    mlflow.autolog()

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

            if model_name == "Exponential" or model_name == "Log-square":
                if model_name == "Log-square":
                    X_test = X_test**2

                X_test = np.log(X_test)

                # turn -inf to 0
                X_test[X_test == -np.inf] = 0
            elif model_name == "Geometric":
                X_test = np.sqrt(X_test)
            elif model_name == "Parabolic":
                X_test = X_test**2

            pred = models[oxide].predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))

            oxide_rmses[oxide] = rmse
            oxide_preds[oxide] = pred

            mlflow.log_metric("RMSE", float(rmse))
            mlflow.log_params({"model": model_name, "oxide": oxide, "norm": norm.value})

    for oxide, rmse in oxide_rmses.items():
        print(f"RMSE for {oxide} with {model_configs[oxide]['law']} model: {rmse}")

    target_predictions = pd.DataFrame()
    target_predictions["ID"] = id_col

    for oxide, pred in oxide_preds.items():
        target_predictions[oxide] = pd.Series(pred, index=target_predictions.index)

    return target_predictions


def get_ica_models(experiment_id: str) -> Dict[str, LinearRegression]:
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
