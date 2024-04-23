import math

import mlflow
import optuna
import typer
from optuna_models import (
    instantiate_extra_trees,
    instantiate_gbr,
    instantiate_pls,
    instantiate_svr,
    instantiate_xgboost,
)
from optuna_preprocessors import (
    instantiate_min_max_scaler,
    instantiate_power_transformer,
    instantiate_robust_scaler,
    instantiate_standard_scaler,
)
from sklearn.metrics import mean_squared_error

from lib import full_flow_dataloader
from lib.config import AppConfig
from lib.reproduction import major_oxides

mlflow.set_tracking_uri(AppConfig().mlflow_tracking_uri)
optuna.logging.set_verbosity(optuna.logging.ERROR)

drop_cols = ["ID", "Sample Name"]


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
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


def combined_objective(trial):
    with mlflow.start_run(nested=True):
        # Select and instantiate a model
        model_selector = trial.suggest_categorical("model_type", ["gbr", "svr", "xgboost", "extra_trees", "pls"])
        if model_selector == "gbr":
            model = instantiate_gbr(trial, lambda params: mlflow.log_params(params))
        elif model_selector == "svr":
            model = instantiate_svr(trial, lambda params: mlflow.log_params(params))
        elif model_selector == "xgboost":
            model = instantiate_xgboost(trial, lambda params: mlflow.log_params(params))
        elif model_selector == "extra_trees":
            model = instantiate_extra_trees(trial, lambda params: mlflow.log_params(params))
        elif model_selector == "pls":
            model = instantiate_pls(trial, lambda params: mlflow.log_params(params))

        # Select and instantiate a preprocessor
        preprocessor_selector = trial.suggest_categorical(
            "preprocessor_type", ["robust_scaler", "standard_scaler", "min_max_scaler", "power_transformer"]
        )
        if preprocessor_selector == "robust_scaler":
            preprocessor = instantiate_robust_scaler(trial, lambda params: mlflow.log_params(params))
        elif preprocessor_selector == "standard_scaler":
            preprocessor = instantiate_standard_scaler(trial, lambda params: mlflow.log_params(params))
        elif preprocessor_selector == "min_max_scaler":
            preprocessor = instantiate_min_max_scaler(trial, lambda params: mlflow.log_params(params))
        elif preprocessor_selector == "power_transformer":
            preprocessor = instantiate_power_transformer(trial, lambda params: mlflow.log_params(params))
        mlflow.log_params({"model_type": model_selector, "preprocessor_type": preprocessor_selector})

        # Preprocess the data
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Train the model
        model.fit(X_train_transformed, y_train)
        preds = model.predict(X_test_transformed)
        mse = mean_squared_error(y_test, preds)
        rmse = math.sqrt(mse)

        # Log metrics
        mlflow.log_metric("mse", float(mse))
        mlflow.log_metric("rmse", rmse)

    return rmse


def main(
    experiment_name: str = typer.Option(..., "--experiment-name", "-e", help="Name of the MLflow experiment"),
    n_trials: int = typer.Option(500, "--n-trials", "-n", help="Number of trials for hyperparameter optimization"),
):
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    for oxide in major_oxides:
        run_name = oxide
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
