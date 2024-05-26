from datetime import datetime
from typing import List

import mlflow
import optuna
import pandas as pd
import requests
import typer
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna_models import (
    instantiate_extra_trees,
    instantiate_gbr,
    instantiate_ngboost,
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
from lib.cross_validation import (
    custom_kfold_cross_validation_new,
    get_cross_validation_metrics,
    perform_cross_validation,
)
from lib.get_preprocess_fn import get_preprocess_fn
from lib.metrics import rmse_metric, std_dev_metric
from lib.reproduction import major_oxides

mlflow.set_tracking_uri(AppConfig().mlflow_tracking_uri)
optuna.logging.set_verbosity(optuna.logging.ERROR)


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
    if study.best_trials:
        # Find the best trial with the least average of values
        best_trial = min(study.best_trials, key=lambda t: sum(t.values) / len(t.values))
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
        rmsep = frozen_trial.user_attrs.get("rmse", float("inf"))
        run_id = frozen_trial.user_attrs.get("run_id", None)

        if best_trial.values and winner != best_trial.values:
            study.set_user_attr("winner", best_trial.values)
            cmn = (
                f"Scaler: `{scaler}`, Transformer: `{transformer}`, PCA: `{pca}`, Std Dev (Test): `{std_dev:.4f}`, RMSEP: `{rmsep:.4f}`, \n"
                f"Cross-Validation Metrics: RMSE CV: `{rmse_cv:.4f}`, Std Dev CV: `{std_dev_cv:.4f}`, \n"
                f"RMSE CV Folds: {rmse_cv_folds}\nStd Dev CV Folds: {std_dev_cv_folds}\n"
                f"MLflow Run ID: `{run_id}`"
            )
            if winner:
                improvement_percent = (
                    abs(sum(winner) / len(winner) - sum(best_trial.values) / len(best_trial.values))
                    / (sum(best_trial.values) / len(best_trial.values))
                ) * 100
                message = (
                    f"{oxide} | Trial {frozen_trial.number} achieved avg of RMSE_CV and STD_DEV_CV: `{sum(frozen_trial.values) / len(frozen_trial.values):.4f}` with "
                    f"{improvement_percent:.4f}% improvement using {model_type}.\n"
                    f"{cmn}"
                )
            else:
                message = (
                    f"{oxide} | Initial trial {frozen_trial.number} achieved avg of RMSE_CV and STD_DEV_CV: `{sum(frozen_trial.values) / len(frozen_trial.values):.4f}` using {model_type}.\n"
                    f"{cmn}"
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
    elif model_selector == "ngboost":
        return instantiate_ngboost(trial, lambda params: _logger(params))
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


def get_data(target: str):
    train_full, test_full = full_flow_dataloader.load_full_flow_data(load_cache_if_exits=True, average_shots=True)
    full_data = pd.concat([train_full, test_full], axis=0)

    folds, train, test = custom_kfold_cross_validation_new(
        data=full_data, k=5, group_by="Sample Name", target=target, random_state=42
    )

    return folds, train, test


def combined_objective(trial, oxide, model_selector):
    try:
        with mlflow.start_run(nested=True) as run:
            mlflow.log_param("trial_number", trial.number)

            # Model selection
            # model_selector = trial.suggest_categorical("model_type", ["gbr", "svr", "xgboost", "extra_trees", "pls"])
            model = instantiate_model(trial, model_selector, lambda params: mlflow.log_params(params))
            mlflow.log_param("model_type", model_selector)

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
            else:
                pca = None
            mlflow.log_param("pca_type", pca_selector)

            # Constructing the pipeline
            steps = []
            steps.append(("scaler", scaler))

            if transformer_selector != "none" and transformer is not None:
                steps.append((transformer_selector, transformer))
            if pca_selector != "none" and pca is not None:
                steps.append((pca_selector, pca))

            preprocessor = Pipeline(steps)

            folds, train, test = get_data(oxide)

            # Log the size of the train and test set to mlflow
            mlflow.log_param("train_size", len(train))
            mlflow.log_param("test_size", len(test))
            # Log the size of each fold to mlflow
            for i, (train_fold, val_fold) in enumerate(folds):
                mlflow.log_param(f"fold_{i+1}_train_size", len(train_fold))
                mlflow.log_param(f"fold_{i+1}_val_size", len(val_fold))

            drop_cols = ["ID", "Sample Name"] + major_oxides
            preprocess_fn = get_preprocess_fn(preprocessor, oxide, drop_cols=drop_cols)

            cv_fold_metrics = perform_cross_validation(
                folds=folds,
                model=model,  # type: ignore
                preprocess_fn=preprocess_fn,
                metric_fns=[rmse_metric, std_dev_metric],
            )

            cv_metrics = get_cross_validation_metrics(cv_fold_metrics)
            mlflow.log_metrics(cv_metrics.as_dict())

            X_train, y_train, X_test, y_test = preprocess_fn(train, test)

            # Model training and evaluation
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            rmse = rmse_metric(y_test, preds)
            std_dev = std_dev_metric(y_test, preds)

            trial.set_user_attr("std_dev", float(std_dev))
            trial.set_user_attr("mse", float(mse))
            trial.set_user_attr("rmse", rmse)
            trial.set_user_attr("rmse_cv", float(cv_metrics.rmse_cv))
            trial.set_user_attr("std_dev_cv", float(cv_metrics.std_dev_cv))
            trial.set_user_attr("rmse_cv_folds", cv_metrics.rmse_cv_folds)
            trial.set_user_attr("std_dev_cv_folds", cv_metrics.std_dev_cv_folds)
            trial.set_user_attr("run_id", run.info.run_id)

            # Log metrics
            mlflow.log_metrics({"mse": float(mse), "rmse": rmse, "std_dev": float(std_dev)})

        return float(cv_metrics.rmse_cv), float(cv_metrics.std_dev_cv)
    except Exception as e:
        import traceback

        print(f"An error occurred: {e}")
        traceback.print_exc()
        return float("inf"), float("inf")  # Return a large number to indicate failure


models = ["ngboost"]


def validate_oxides(ctx: typer.Context, param: typer.CallbackParam, value: List[str]) -> List[str]:
    for oxide in value:
        if oxide not in major_oxides:
            raise typer.BadParameter(f"{oxide} is not a valid oxide. Choose from {major_oxides}")
    return value


def main(
    n_trials: int = typer.Option(200, "--n-trials", "-n", help="Number of trials for hyperparameter optimization"),
    selected_oxides: List[str] = typer.Option(
        major_oxides, "--oxides", "-o", help="List of oxides to optimize", callback=validate_oxides
    ),
):
    """
    Executes hyperparameter optimization for specified oxides using Optuna with MLflow tracking.

    This function sets up and runs an Optuna optimization study for different machine learning models
    on specified oxide datasets. Each oxide and model combination is optimized separately with a given
    number of trials. The function logs the optimization process and results to MLflow and optionally
    sends notifications via Discord.

    Parameters:
    - n_trials (int): The number of trials to run for the hyperparameter optimization. Each trial
                      tests a set of parameters on the specified model and oxide.
    - selected_oxides (List[str]): A list of oxide names for which the optimization will be performed.
                                   The oxides should be among the predefined valid oxides, and the
                                   function will validate this list.

    The function uses Typer for CLI argument parsing, allowing the number of trials and the list of
    oxides to be specified at runtime. It supports dynamic adjustment of the experimental setup and
    integrates with MLflow for experiment tracking and management.

    Example CLI Usage:
    - Run optimization for 100 trials on SiO2 and Al2O3:
      `python optuna_run.py --n-trials 100 --oxides SiO2 --oxides Al2O3`
    - Run optimization with default settings (200 trials on all predefined oxides):
      `python optuna_run.py`
    """
    n_startup_trials = int(0.25 * n_trials)  # Reserve 25% of trials for exploration
    sampler = TPESampler(n_startup_trials=n_startup_trials, n_ei_candidates=20, seed=42)
    pruner = HyperbandPruner(min_resource=1, max_resource=10, reduction_factor=3)
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    for oxide in selected_oxides:
        print(f"Optimizing for {oxide}")
        notify_discord(f"# Optimizing for {oxide}")

        experiment_id = get_or_create_experiment(f"Optuna {oxide} - NGBoost - {current_date}")
        mlflow.set_experiment(experiment_id=experiment_id)

        for model in models:
            print(f"Optimizing {model}")
            notify_discord(f"## Optimizing {model}")

            with mlflow.start_run(experiment_id=experiment_id, run_name=model):
                # Initialize the Optuna study
                study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler, pruner=pruner)

                # Execute the hyperparameter optimization trials.
                # Note the addition of the `champion_callback` inclusion to control our logging
                study.optimize(
                    lambda trial: combined_objective(trial, oxide, model),
                    n_trials=n_trials,
                    callbacks=[
                        lambda study, frozen_trial: champion_callback(study, frozen_trial, oxide, model),
                    ],
                )


if __name__ == "__main__":
    typer.run(main)
