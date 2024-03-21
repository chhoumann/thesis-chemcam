from typing import Dict, Optional

import mlflow
import numpy as np
import pandas as pd
import tqdm
from mlflow.entities import Experiment
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from lib.config import AppConfig
from lib.norms import Norm

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

    _run_experiment(
        ica_df_n1, ica_df_n3, compositions_df_n1, compositions_df_n3, "train"
    )

    return experiment


def test(
    ica_df_n1: pd.DataFrame,
    ica_df_n3: pd.DataFrame,
    compositions_df_n1: pd.DataFrame,
    compositions_df_n3: pd.DataFrame,
    train_experiment_id: str,  # Used to fetch the trained models
) -> pd.DataFrame:
    _setup_mlflow_experiment("TEST")
    models = _get_ica_models(train_experiment_id)

    return _run_experiment(
        ica_df_n1, ica_df_n3, compositions_df_n1, compositions_df_n3, "test", models
    )


def _run_experiment(
    ica_df_n1: pd.DataFrame,
    ica_df_n3: pd.DataFrame,
    compositions_df_n1: pd.DataFrame,
    compositions_df_n3: pd.DataFrame,
    phase: str,  # "train" or "test"
    models: Optional[Dict[str, LinearRegression]] = None,
) -> pd.DataFrame:
    if phase == "test" and models is None:
        raise ValueError("Models must be provided when testing.")

    target_predictions = _create_target_predictions_df(ica_df_n1, ica_df_n3)
    oxide_rmses = {}

    for oxide, info in tqdm.tqdm(model_configs.items()):
        law_name = info["law"]
        norm = info["norm"]

        if law_name is None:
            print(f"No model specified for {oxide}. Skipping...")
            continue

        ica_df = ica_df_n1 if norm == Norm.NORM_1 else ica_df_n3
        compositions_df = (
            compositions_df_n1 if norm == Norm.NORM_1 else compositions_df_n3
        )

        run_name_prefix = "ICA_TRAIN" if phase == "train" else "ICA_TEST"

        with mlflow.start_run(run_name=f"{run_name_prefix}_{oxide}"):
            print(f"{phase.capitalize()}ing model {law_name} for {oxide}...")

            X = _transform_to_regression_law(ica_df, law_name)
            y = compositions_df[oxide]

            X.fillna(0, inplace=True)

            if phase == "train":
                # Training phase: fit a new model and log it
                model = LinearRegression()
                model.fit(X, y)
                pred = model.predict(X)
                mlflow.sklearn.log_model(model, f"ICA_{oxide}")
            else:
                # Testing phase: use the provided models for predictions
                pred = models[oxide].predict(X)  # type: ignore

            # Store predictions
            target_predictions[oxide] = pd.Series(pred, index=target_predictions.index)

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y, pred))
            oxide_rmses[oxide] = rmse

            # Log metrics and params
            mlflow.log_metric("RMSE", rmse.item())
            mlflow.log_params({"model": law_name, "oxide": oxide, "norm": norm.value})

    _print_rmses(oxide_rmses)

    return target_predictions


def _setup_mlflow_experiment(mode: str) -> Experiment:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    _experiment_id = mlflow.create_experiment(
        f"ICA_{mode}_{pd.Timestamp.now().strftime('%m-%d-%y_%H%M%S')}"
    )
    experiment = mlflow.set_experiment(experiment_id=_experiment_id)
    mlflow.autolog()

    return experiment


def _create_target_predictions_df(
    ica_df_n1: pd.DataFrame,
    ica_df_n3: pd.DataFrame,
    index: Optional[pd.Index] = None,
) -> pd.DataFrame:
    id_col = ica_df_n1["ID"]
    sample_name_col = ica_df_n1["Sample Name"]

    ica_df_n1.drop(columns=["ID", "Sample Name"], inplace=True)
    ica_df_n3.drop(columns=["ID", "Sample Name"], inplace=True)

    target_predictions = pd.DataFrame(index)
    target_predictions["ID"] = id_col
    target_predictions["Sample Name"] = sample_name_col

    return target_predictions


def _transform_to_regression_law(df: pd.DataFrame, law_name: str) -> pd.DataFrame:
    if law_name == "Exponential" or law_name == "Log-square":
        if law_name == "Log-square":
            df = df**2

        # turn -inf to 0
        df = np.log(df)  # type: ignore
        df[df == -np.inf] = 0
    elif law_name == "Geometric":
        df = np.sqrt(df)  # type: ignore
    elif law_name == "Parabolic":
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
