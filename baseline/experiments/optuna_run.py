import math
from typing import List

import mlflow
import numpy as np
import optuna
import requests
import typer
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
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
    instantiate_norm3_scaler,
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
from lib.utils import CustomKFoldCrossValidator, perform_cross_validation

mlflow.set_tracking_uri(AppConfig().mlflow_tracking_uri)
optuna.logging.set_verbosity(optuna.logging.ERROR)


def get_preprocess_fn(preprocessor, target_col: str, drop_cols: List[str]):
    def preprocess_fn(train, test):
        X_train = train.drop(columns=drop_cols)
        y_train = train[target_col]

        X_test = test.drop(columns=drop_cols)
        y_test = test[target_col]

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        return X_train_transformed, y_train, X_test_transformed, y_test

    return preprocess_fn


def rmse_metric(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def std_dev_metric(y_true, y_pred):
    return float(np.std(y_true - y_pred))


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
def champion_callback(study, frozen_trial, oxide, model):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values, including the model type and other details that achieved the new best value.
    """
    winner = study.user_attrs.get("winner", None)
    model_type = frozen_trial.params.get("model_type", model)
    scaler = frozen_trial.params.get("scaler_type", "Unknown scaler")
    transformer = frozen_trial.params.get("transformer_type", "None")
    pca = frozen_trial.params.get("pca_type", "None")
    std_dev = frozen_trial.user_attrs.get("std_dev", float("inf"))
    rmse_cv = frozen_trial.user_attrs.get("rmse_cv", float("inf"))
    std_dev_cv = frozen_trial.user_attrs.get("std_dev_cv", float("inf"))
    rmse_cv_folds = frozen_trial.user_attrs.get("rmse_cv_folds", {})
    std_dev_cv_folds = frozen_trial.user_attrs.get("std_dev_cv_folds", {})

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        study.set_user_attr("best_std_dev", std_dev)  # Store the best std_dev
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            message = (
                f"{oxide} | Trial {frozen_trial.number} achieved value: `{frozen_trial.value:.4f}` with "
                f"{improvement_percent:.4f}% improvement using {model_type}.\n"
                f"Scaler: {scaler}, Transformer: {transformer}, PCA: {pca}, Std Dev: `{std_dev:.4f}`, \n"
                f"Cross-Validation Metrics: RMSE CV: `{rmse_cv:.4f}`, Std Dev CV: `{std_dev_cv:.4f}`, \n"
                f"RMSE CV Folds: {rmse_cv_folds}\nStd Dev CV Folds: {std_dev_cv_folds}"
            )
        else:
            message = (
                f"{oxide} | Initial trial {frozen_trial.number} achieved value: `{frozen_trial.value:.4f}` using {model_type}.\n"
                f"Scaler: {scaler}, Transformer: {transformer}, PCA: {pca}, Std Dev: `{std_dev:.4f}`, \n"
                f"Cross-Validation Metrics: RMSE CV: `{rmse_cv:.4f}`, Std Dev CV: `{std_dev_cv:.4f}`, \n"
                f"RMSE CV Folds: {rmse_cv_folds}\nStd Dev CV Folds: {std_dev_cv_folds}"
            )
        print(message)
        notify_discord(message)


def notify_discord(message: str):
    """
    Send a notification message to a Discord webhook.

    Parameters:
    - message (str): The message to send.
    """
    webhook_url = AppConfig().discord_webhook_url
    if webhook_url:
        try:
            response = requests.post(webhook_url, json={"content": message})
            if response.status_code != 204:
                print(f"Failed to send message to Discord: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"Failed to send message to Discord: {e}")
    else:
        print("Discord webhook URL is not set. Skipping notification.")


def instantiate_model(trial, model_selector, logger):
    def _logger(params):
        logger({f"{model_selector}_{k}": v for k, v in params.items()})

    if model_selector == "gbr":
        return instantiate_gbr(trial, lambda params: _logger(params))
    elif model_selector == "svr":
        return instantiate_svr(trial, lambda params: _logger(params))
    elif model_selector == "xgboost":
        return instantiate_xgboost(trial, lambda params: _logger(params))
    elif model_selector == "extra_trees":
        return instantiate_extra_trees(trial, lambda params: _logger(params))
    elif model_selector == "pls":
        return instantiate_pls(trial, lambda params: _logger(params))
    else:
        raise ValueError(f"Unsupported model type: {model_selector}")


def instantiate_scaler(trial, scaler_selector, logger):
    def _logger(params):
        logger({f"{scaler_selector}_{k}": v for k, v in params.items()})

    if scaler_selector == "robust_scaler":
        return instantiate_robust_scaler(trial, _logger)
    elif scaler_selector == "standard_scaler":
        return instantiate_standard_scaler(trial, _logger)
    elif scaler_selector == "min_max_scaler":
        return instantiate_min_max_scaler(trial, _logger)
    elif scaler_selector == "max_abs_scaler":
        return instantiate_max_abs_scaler(trial, _logger)
    elif scaler_selector == "norm3_scaler":
        return instantiate_norm3_scaler(trial, _logger)
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_selector}")


def combined_objective(trial, oxide, model):
    try:
        with mlflow.start_run(nested=True):
            mlflow.log_param("trial_number", trial.number)

            # Model selection
            # model_selector = trial.suggest_categorical("model_type", ["gbr", "svr", "xgboost", "extra_trees", "pls"])
            model = instantiate_model(trial, model, lambda params: mlflow.log_params(params))
            mlflow.log_param("model_type", model)

            # Preprocessor components
            scaler_selector = trial.suggest_categorical(
                "scaler_type", ["robust_scaler", "standard_scaler", "min_max_scaler", "max_abs_scaler", "norm3_scaler"]
            )
            scaler = instantiate_scaler(trial, scaler_selector, lambda params: mlflow.log_params(params))
            mlflow.log_param("scaler_type", scaler_selector)

            transformer_selector = trial.suggest_categorical(
                "transformer_type", ["power_transformer", "quantile_transformer", "none"]
            )
            if transformer_selector == "power_transformer":
                transformer = instantiate_power_transformer(trial, lambda params: mlflow.log_params(params))
            elif transformer_selector == "quantile_transformer":
                transformer = instantiate_quantile_transformer(trial, lambda params: mlflow.log_params(params))
            else:
                transformer = None
            mlflow.log_param("transformer_type", transformer_selector)

            pca_selector = trial.suggest_categorical("pca_type", ["pca", "kernel_pca", "none"])
            if pca_selector == "pca":
                pca = instantiate_pca(trial, lambda params: mlflow.log_params(params))
            elif pca_selector == "kernel_pca":
                pca = instantiate_kernel_pca(trial, lambda params: mlflow.log_params(params))
            mlflow.log_param("pca_type", pca_selector)

            # Constructing the pipeline
            steps = [("scaler", scaler)]
            if transformer_selector != "none":
                steps.append((transformer_selector, transformer))  # type: ignore
            if pca_selector != "none":
                steps.append((pca_selector, pca))  # type: ignore

            preprocessor = Pipeline(steps)

            # Drop all oxides except for the current oxide
            drop_cols = ["ID", "Sample Name"]
            drop_cols.extend([oxide for oxide in major_oxides if oxide != oxide])
            train_full, test_full = full_flow_dataloader.load_full_flow_data(
                load_cache_if_exits=True, average_shots=True
            )

            kf = CustomKFoldCrossValidator(k=5, group_by="Sample Name", random_state=42)
            preprocess_fn = get_preprocess_fn(preprocessor, oxide, drop_cols=drop_cols)

            cv_fold_metrics = perform_cross_validation(
                data=train_full,
                model=model,  # type: ignore
                preprocess_fn=preprocess_fn,
                metric_fns=[rmse_metric, std_dev_metric],
                kf=kf,
            )

            rmse_cv = np.mean(cv_fold_metrics[0])
            std_dev_cv = np.mean(cv_fold_metrics[1])
            rmse_cv_folds = {f"rmse_cv_{i}": rmse for i, (rmse, _) in enumerate(cv_fold_metrics)}
            std_dev_cv_folds = {f"std_dev_cv_{i}": std_dev for i, (_, std_dev) in enumerate(cv_fold_metrics)}

            for i, (rmse, std_dev) in enumerate(cv_fold_metrics):
                mlflow.log_metric("rmse_cv", rmse, step=i)
                mlflow.log_metric("std_dev_cv", std_dev, step=i)

            mlflow.log_metric("rmse_cv_avg", float(rmse_cv))
            mlflow.log_metric("std_dev_cv_avg", float(std_dev_cv))
            mlflow.log_metrics(rmse_cv_folds)
            mlflow.log_metrics(std_dev_cv_folds)

            X_train, y_train, X_test, y_test = preprocess_fn(train_full, test_full)

            # Model training and evaluation
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            rmse = math.sqrt(mse)
            std_dev = np.std(y_test - preds)

            trial.set_user_attr("std_dev", float(std_dev))
            trial.set_user_attr("mse", float(mse))
            trial.set_user_attr("rmse", rmse)
            trial.set_user_attr("rmse_cv", float(rmse_cv))
            trial.set_user_attr("std_dev_cv", float(std_dev_cv))
            trial.set_user_attr("rmse_cv_folds", rmse_cv_folds)
            trial.set_user_attr("std_dev_cv_folds", std_dev_cv_folds)

            # Log metrics
            mlflow.log_metrics({"mse": float(mse), "rmse": rmse, "std_dev": float(std_dev)})

        return rmse
    except Exception as e:
        import traceback

        print(f"An error occurred: {e}")
        traceback.print_exc()
        return float("inf")  # Return a large number to indicate failure


models = ["gbr", "svr", "xgboost", "extra_trees", "pls"]


def main(
    n_trials: int = typer.Option(500, "--n-trials", "-n", help="Number of trials for hyperparameter optimization"),
):

    sampler = TPESampler(n_startup_trials=50, n_ei_candidates=20, seed=42)
    pruner = HyperbandPruner(min_resource=1, max_resource=10, reduction_factor=3)

    for oxide in major_oxides:
        print(f"Optimizing for {oxide}")
        notify_discord(f"# Optimizing for {oxide}")
        experiment_id = get_or_create_experiment(f"Optuna {oxide}")
        mlflow.set_experiment(experiment_id=experiment_id)

        for model in models:
            print(f"Optimizing {model}")
            notify_discord(f"## Optimizing {model}")

            with mlflow.start_run(experiment_id=experiment_id, run_name=model):
                # Initialize the Optuna study
                study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

                # Execute the hyperparameter optimization trials.
                # Note the addition of the `champion_callback` inclusion to control our logging
                study.optimize(
                    lambda trial: combined_objective(trial, oxide, model),
                    n_trials=n_trials,
                    callbacks=[
                        lambda study, frozen_trial: champion_callback(study, frozen_trial, oxide, model),
                    ],
                )

                mlflow.log_params(study.best_params)
                mlflow.log_metric("best_mse", study.best_value)
                mlflow.log_metric("best_rmse", math.sqrt(study.best_value))
                mlflow.log_metric("best_std_dev", study.user_attrs.get("std_dev", 0))

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
