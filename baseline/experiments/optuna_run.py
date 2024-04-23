import math

import mlflow
import optuna
import requests
import typer
from optuna_models import (
    instantiate_extra_trees,
    instantiate_gbr,
    instantiate_pls,
    instantiate_svr,
    instantiate_xgboost,
)
from optuna_preprocessors import (
    instantiate_kernel_pca,
    instantiate_max_abs_scaler,
    instantiate_min_max_scaler,
    instantiate_pca,
    instantiate_power_transformer,
    instantiate_quantile_transformer,
    instantiate_robust_scaler,
    instantiate_standard_scaler,
)
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from lib import full_flow_dataloader
from lib.config import AppConfig
from lib.reproduction import major_oxides

mlflow.set_tracking_uri(AppConfig().mlflow_tracking_uri)
optuna.logging.set_verbosity(optuna.logging.ERROR)

drop_cols = ["ID", "Sample Name"]
CURRENT_OXIDE = ""

train_processed, test_processed = full_flow_dataloader.load_full_flow_data(load_cache_if_exits=True, average_shots=True)
target = major_oxides[0]

drop_cols.extend([oxide for oxide in major_oxides if oxide != target])
train_processed = train_processed.drop(columns=drop_cols)
test_processed = test_processed.drop(columns=drop_cols)

X_train = train_processed.drop(columns=[target])
y_train = train_processed[target]

X_test = test_processed.drop(columns=[target])
y_test = test_processed[target]


# https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs.html
def get_or_create_experiment(experiment_name: str) -> str:
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


# define a logging callback that will report on only new challenger parameter configurations if a
# trial has usurped the state of 'best conditions'


# https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs.html
def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values, including the model type and other details that achieved the new best value.
    """
    winner = study.user_attrs.get("winner", None)
    model_type = frozen_trial.params.get("model_type", "Unknown model type")
    scaler = frozen_trial.params.get("scaler_type", "Unknown scaler")
    transformer = frozen_trial.params.get("transformer_type", "None")
    pca = frozen_trial.params.get("pca_type", "None")

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            message = (
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent:.4f}% improvement using {model_type}. "
                f"Scaler: {scaler}, Transformer: {transformer}, PCA: {pca}"
            )
        else:
            message = (
                f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value} using {model_type}. "
                f"Scaler: {scaler}, Transformer: {transformer}, PCA: {pca}"
            )
        print(message)
        notify_discord(message)


def notify_discord(message):
    """
    Send a notification message to a Discord webhook.

    Parameters:
    - message (str): The message to send.
    """
    webhook_url = AppConfig().discord_webhook_url
    if webhook_url:
        data = {"content": f"{CURRENT_OXIDE} | {message}"}
        response = requests.post(webhook_url, json=data)
        if response.status_code != 204:
            print(f"Failed to send message to Discord: {response.status_code}, {response.text}")
    else:
        print("Discord webhook URL is not set. Skipping notification.")


def instantiate_model(trial, model_selector, logger):
    if model_selector == "gbr":
        return instantiate_gbr(trial, logger)
    elif model_selector == "svr":
        return instantiate_svr(trial, logger)
    elif model_selector == "xgboost":
        return instantiate_xgboost(trial, logger)
    elif model_selector == "extra_trees":
        return instantiate_extra_trees(trial, logger)
    elif model_selector == "pls":
        return instantiate_pls(trial, logger)
    else:
        raise ValueError(f"Unsupported model type: {model_selector}")


def instantiate_scaler(trial, scaler_selector, logger):
    if scaler_selector == "robust_scaler":
        return instantiate_robust_scaler(trial, logger)
    elif scaler_selector == "standard_scaler":
        return instantiate_standard_scaler(trial, logger)
    elif scaler_selector == "min_max_scaler":
        return instantiate_min_max_scaler(trial, logger)
    elif scaler_selector == "max_abs_scaler":
        return instantiate_max_abs_scaler(trial, logger)
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_selector}")


def combined_objective(trial):
    with mlflow.start_run(nested=True):
        # Model selection
        model_selector = trial.suggest_categorical("model_type", ["gbr", "svr", "xgboost", "extra_trees", "pls"])
        model = instantiate_model(trial, model_selector, lambda params: mlflow.log_params(params))

        # Preprocessor components
        scaler_selector = trial.suggest_categorical(
            "scaler_type", ["robust_scaler", "standard_scaler", "min_max_scaler", "max_abs_scaler"]
        )
        scaler = instantiate_scaler(trial, scaler_selector, lambda params: mlflow.log_params(params))

        transformer_selector = trial.suggest_categorical(
            "transformer_type", ["power_transformer", "quantile_transformer", "none"]
        )
        if transformer_selector == "power_transformer":
            transformer = instantiate_power_transformer(trial, lambda params: mlflow.log_params(params))
        elif transformer_selector == "quantile_transformer":
            transformer = instantiate_quantile_transformer(trial, lambda params: mlflow.log_params(params))
        else:
            transformer = None

        pca_selector = trial.suggest_categorical("pca_type", ["pca", "kernel_pca", "none"])
        if pca_selector == "pca":
            pca = instantiate_pca(trial, lambda params: mlflow.log_params(params))
        elif pca_selector == "kernel_pca":
            pca = instantiate_kernel_pca(trial, lambda params: mlflow.log_params(params))

        # Constructing the pipeline
        steps = [("scaler", scaler)]
        if transformer_selector != "none":
            steps.append((transformer_selector, transformer))
        if pca_selector != "none":
            steps.append((pca_selector, pca))

        preprocessor = Pipeline(steps)

        # Data transformation
        X_train_transformed = preprocessor.fit_transform(X_train.copy())
        X_test_transformed = preprocessor.transform(X_test.copy())

        # Model training and evaluation
        model.fit(X_train_transformed, y_train.copy())
        preds = model.predict(X_test_transformed)
        mse = mean_squared_error(y_test.copy(), preds)
        rmse = math.sqrt(mse)

        # Log metrics
        mlflow.log_metric("mse", float(mse))
        mlflow.log_metric("rmse", rmse)

    return rmse


def main(
    experiment_name: str = typer.Option(..., "--experiment-name", "-e", help="Name of the MLflow experiment"),
    n_trials: int = typer.Option(500, "--n-trials", "-n", help="Number of trials for hyperparameter optimization"),
):
    global CURRENT_OXIDE
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    for oxide in major_oxides:
        run_name = oxide
        CURRENT_OXIDE = oxide
        print(f"Optimizing for {oxide}")
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
            # Initialize the Optuna study
            study = optuna.create_study(direction="minimize")

            # Execute the hyperparameter optimization trials.
            # Note the addition of the `champion_callback` inclusion to control our logging
            study.optimize(combined_objective, n_trials=n_trials, callbacks=[champion_callback])

            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_mse", study.best_value)
            mlflow.log_metric("best_rmse", math.sqrt(study.best_value))

            # Log tags
            mlflow.set_tags(
                tags={
                    "project": "AutoML Experiments",
                    "optimizer_engine": "optuna",
                    "feature_set_version": 1,
                }
            )


if __name__ == "__main__":
    typer.run(main)
