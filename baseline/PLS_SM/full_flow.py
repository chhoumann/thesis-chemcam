import logging
from pathlib import Path
from typing import Dict

# import warnings filter
from warnings import simplefilter

import mlflow
import numpy as np
import pandas as pd
import typer
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from lib import full_flow_dataloader

# from config import logger
from lib.config import AppConfig
from lib.norms import Norm1Scaler, Norm3Scaler
from lib.outlier_removal import (
    calculate_leverage_residuals,
    identify_outliers,
    plot_leverage_residuals,
)
from lib.reproduction import (
    major_oxides,
    masks,
    optimized_blending_ranges,
    oxide_ranges,
    paper_individual_sm_rmses,
    training_info,
)
from lib.utils import custom_kfold_cross_validation, filter_data_by_compositional_range
from PLS_SM.inference import predict_composition_with_blending

# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger("train")
config = AppConfig()
mlflow.set_tracking_uri(config.mlflow_tracking_uri)


app = typer.Typer()


@app.command(name="train", help="Train the PLS-SM models.")
def train(
    outlier_removal: bool = True,
    additional_info: str = "",
    outlier_removal_constraint_iteration: int = -1,
):
    train_processed, test_processed = full_flow_dataloader.load_full_flow_data()
    k_folds = 4
    random_state = 42
    influence_plot_dir = Path("plots/")
    timestamp = pd.Timestamp.now().strftime("%m-%d-%y_%H%M%S")
    no_or = "" if outlier_removal else "NO-OR"
    add_info = "" if additional_info == "" else f"{additional_info}"

    if outlier_removal_constraint_iteration > 0:
        add_info += f"OR{outlier_removal_constraint_iteration}C"

    name_tags = ["PLS_Train", no_or, add_info, timestamp]
    experiment_name = "_".join(name_tags)

    experiment = mlflow.set_experiment(experiment_name)
    mlflow.autolog(log_datasets=False, silent=True)

    for oxide in tqdm(major_oxides, desc="Processing oxides"):
        _oxide_ranges = oxide_ranges.get(oxide, None)
        if _oxide_ranges is None:
            logger.info("Skipping oxide: %s", oxide)
            continue

        for compositional_range in _oxide_ranges.keys():
            logger.debug(
                "Starting MLflow run for compositional range: %s, oxide: %s",
                compositional_range,
                oxide,
            )

            train_cols = train_processed.columns
            test_cols = test_processed.columns

            n_components = training_info[oxide][compositional_range]["n_components"]
            norm = training_info[oxide][compositional_range]["normalization"]

            logger.info(
                f"Filtering {oxide} for {compositional_range} compositional range..."
            )
            train = filter_data_by_compositional_range(
                train_processed, compositional_range, oxide, oxide_ranges
            )

            test = filter_data_by_compositional_range(
                test_processed, compositional_range, oxide, oxide_ranges
            )

            scaler = Norm1Scaler() if norm == 1 else Norm3Scaler()
            logger.debug("Initializing scaler: %s", scaler.__class__.__name__)

            logger.debug("Fitting and transforming training data.")
            train = scaler.fit_transform(train.copy())
            logger.debug("Transforming test data.")
            test = scaler.fit_transform(test.copy())

            drop_cols = major_oxides + ["Sample Name", "ID"]

            # turn back into dataframe
            train = pd.DataFrame(train, columns=train_cols)
            test = pd.DataFrame(test, columns=test_cols)

            with mlflow.start_run(run_name=f"{oxide}_{compositional_range}"):
                mlflow.log_metric(
                    "paper_rmse", paper_individual_sm_rmses[compositional_range][oxide]
                )

                mlflow.log_params(
                    {
                        "masks": masks,
                        "range": oxide_ranges[oxide][compositional_range],
                        "compositional_range": compositional_range,
                        "oxide": oxide,
                        "n_spectra": len(train),
                        "outlier_removal": outlier_removal,
                        "norm": norm,
                    }
                )

                # region OUTLIER REMOVAL
                outlier_removal_iterations = 0
                pls_OR = PLSRegression(n_components=n_components)
                X_train_OR = train.drop(columns=drop_cols).to_numpy()
                y_train_OR = train[oxide].to_numpy()

                pls_OR.fit(X_train_OR, y_train_OR)

                current_performance = mean_squared_error(
                    y_train_OR, pls_OR.predict(X_train_OR), squared=False
                )

                mlflow.log_metric(
                    "RMSEOR",
                    float(current_performance),
                    step=outlier_removal_iterations,
                )

                best_or_model = pls_OR

                train_no_outliers = train.copy()

                leverage, Q = None, None

                while True and outlier_removal:
                    outlier_removal_iterations += 1

                    should_calculate_constraints = (
                        outlier_removal_constraint_iteration >= 0
                        and outlier_removal_iterations
                        <= outlier_removal_constraint_iteration
                    ) or outlier_removal_constraint_iteration < 0

                    if should_calculate_constraints:
                        leverage, Q = calculate_leverage_residuals(pls_OR, X_train_OR)

                    outliers = identify_outliers(leverage, Q)

                    if len(outliers) == 0:
                        break

                    outliers_indices = np.where(outliers)[0]

                    # Plotting the influence plot
                    plot_path = Path(
                        influence_plot_dir
                        / f"{experiment_name}"
                        / f"{oxide}_{compositional_range}_{outlier_removal_iterations}.png"
                    )
                    plot_path.parent.mkdir(parents=True, exist_ok=True)

                    plot_leverage_residuals(leverage, Q, outliers, str(plot_path))
                    mlflow.log_artifact(str(plot_path))

                    X_train_OR = np.delete(X_train_OR, outliers_indices, axis=0)
                    y_train_OR = np.delete(y_train_OR, outliers_indices, axis=0)

                    if should_calculate_constraints:
                        pls_OR = PLSRegression(n_components=n_components)
                        pls_OR.fit(X_train_OR, y_train_OR)

                    new_performance = mean_squared_error(
                        y_train_OR, pls_OR.predict(X_train_OR), squared=False
                    )

                    mlflow.log_metric(
                        "RMSEOR",
                        float(new_performance),
                        step=outlier_removal_iterations,
                    )

                    number_of_outliers = np.sum(
                        outliers
                    )  # Counting the number of True values
                    mlflow.log_metric(
                        "outliers_removed",
                        float(number_of_outliers),
                        step=outlier_removal_iterations,
                    )

                    # Check if error has increased: early stop if so
                    if float(new_performance) >= float(current_performance):
                        break

                    # Update to only have best set
                    train_no_outliers = train.drop(index=train.index[outliers_indices])

                    current_performance = new_performance
                    best_or_model = pls_OR

                mlflow.log_metric(
                    "outlier_removal_iterations", outlier_removal_iterations
                )
                mlflow.sklearn.log_model(
                    best_or_model, f"PLS_OR_{oxide}_{compositional_range}"
                )

                # endregion
                # region Cross-Validation
                best_CV_model = None
                best_CV_rmse = np.inf

                logger.info("Performing custom k-fold cross-validation.")

                kf = custom_kfold_cross_validation(
                    train_no_outliers,
                    k=k_folds,
                    group_by="Sample Name",
                    random_state=random_state,
                )

                fold_rmses = []
                for i, (train_data, test_data) in enumerate(kf):  # type: ignore
                    pls_CV = PLSRegression(n_components=n_components)
                    X_train_CV = train_data.drop(columns=drop_cols).to_numpy()
                    y_train_CV = train_data[oxide].to_numpy()
                    X_test = test_data.drop(columns=drop_cols).to_numpy()
                    y_test = test_data[oxide].to_numpy()

                    pls_CV.fit(X_train_CV, y_train_CV)
                    y_pred = pls_CV.predict(X_test)
                    fold_rmse = mean_squared_error(y_test, y_pred, squared=False)
                    fold_rmses.append(fold_rmse)

                    mlflow.log_metric(f"fold_{i}_RMSE", float(fold_rmse))

                    if float(fold_rmse) < float(best_CV_rmse):
                        best_CV_rmse = fold_rmse
                        best_CV_model = pls_CV

                avg_rmse = np.mean(fold_rmses)
                mlflow.log_metric("RMSECV", float(avg_rmse))
                mlflow.log_metric("RMSECV_MIN", float(np.min(fold_rmses)))
                mlflow.sklearn.log_model(
                    best_CV_model, f"PLS_CV_{oxide}_{compositional_range}"
                )
                # endregion

                # region Train model on all data (outliers removed) & get RMSEP
                X_test = test.drop(columns=drop_cols).to_numpy()
                y_test = test[oxide].to_numpy()

                X_train_NO = train_no_outliers.drop(columns=drop_cols).to_numpy()
                y_train_NO = train_no_outliers[oxide].to_numpy()

                pls_all = PLSRegression(n_components=n_components)
                pls_all.fit(X_train_NO, y_train_NO)
                y_pred = pls_all.predict(X_test)
                # y_pred = best_CV_model.predict(X_test)
                test_rmse = mean_squared_error(y_test, y_pred, squared=False)
                mlflow.log_metric("RMSEP", float(test_rmse))
                mlflow.sklearn.log_model(
                    pls_all, f"PLS_ALL_{oxide}_{compositional_range}"
                )
                # endregion
            mlflow.end_run()

    return experiment


# region Use models to predict on test data - Full PLS-SM
def get_models(experiment_id: str) -> Dict[str, Dict[str, PLSRegression]]:
    # Oxide -> Compositional Range -> Model
    models = {}
    client = mlflow.tracking.MlflowClient()

    for oxide, v in training_info.items():
        sub = {}
        for comp_range_name, _ in v.items():
            model_name = f"PLS_ALL_{oxide}_{comp_range_name}"
            model_found = False

            for run in client.search_runs([experiment_id]):
                for artifact in client.list_artifacts(run.info.run_id):
                    if model_name in artifact.path:
                        model_uri = f"{run.info.artifact_uri}/{artifact.path}"
                        model = mlflow.pyfunc.load_model(model_uri)
                        sub[comp_range_name] = model
                        model_found = True
                        break  # Exit after finding the first match

                if model_found:
                    break  # Exit if model is found

            if not model_found:
                logger.warning("No model found for %s", model_name)
                raise ValueError(f"No model found for {model_name}")

        models[oxide] = sub

    return models


@app.command(name="test", help="Test the PLS-SM models.")
def test(
    experiment_id: str, outlier_removal: bool = True, additional_info: str = ""
) -> pd.DataFrame:
    train_processed, test_processed = full_flow_dataloader.load_full_flow_data()
    timestamp = pd.Timestamp.now().strftime("%m-%d-%y_%H%M%S")
    no_or = "" if outlier_removal else "NO-OR"
    add_info = "" if additional_info == "" else f"{additional_info}"
    name_tags = ["PLS_TEST", no_or, add_info, timestamp]

    experiment_name = "_".join(name_tags)

    mlflow.set_experiment(experiment_name)
    mlflow.autolog(log_models=False, log_datasets=False, silent=True)

    models = get_models(experiment_id)

    count_pre_drop = len(test_processed)
    test_processed.dropna(inplace=True, axis="index")
    count_post_drop = len(test_processed)

    logger.warn(
        "Dropped %d rows with NaNs from test data. %d rows remaining.",
        count_pre_drop - count_post_drop,
        count_post_drop,
    )

    Y = test_processed[major_oxides]
    drop_cols = major_oxides + ["Sample Name", "ID"]

    target_predictions = pd.DataFrame(test_processed[["Sample Name", "ID"]])

    n1_scaler = Norm1Scaler()
    n3_scaler = Norm3Scaler()

    X_test_n1 = n1_scaler.fit_transform(test_processed.drop(drop_cols, axis=1))
    X_test_n3 = n3_scaler.fit_transform(test_processed.drop(drop_cols, axis=1))

    with mlflow.start_run(run_name="PLS-SM Test"):
        for oxide in tqdm(major_oxides, desc="Predicting oxides"):
            _oxide_ranges = oxide_ranges.get(oxide, None)
            if _oxide_ranges is None:
                logger.info("Skipping oxide: %s", oxide)
                continue

            pred = np.array(
                predict_composition_with_blending(
                    oxide, X_test_n1, X_test_n3, models, optimized_blending_ranges
                )
            )

            # save
            pred_df = pd.DataFrame(pred, index=Y.index)

            # Check for NaNs in Y[oxide]
            nan_in_Y = Y[oxide].isna()
            if nan_in_Y.any():
                print("NaNs in Y[oxide]:")
                print(Y[oxide][nan_in_Y])

            # Check for NaNs in pred_df
            nan_in_pred_df = pred_df.isna()
            if nan_in_pred_df.any().any():  # Note the double .any() here
                print("NaNs in pred_df:")
                print(pred_df[nan_in_pred_df])


            # Remove NaNs from pred_df and Y[oxide]
            nan_mask = pred_df.notna().squeeze()  # assuming pred_df is a single column
            pred_df = pred_df[nan_mask]
            Y_oxide = Y[oxide][nan_mask]

            # calculate RMSEP
            rmsep = mean_squared_error(Y_oxide, pred_df, squared=False)
            mlflow.log_metric(f"RMSEP_{oxide}", float(rmsep))

            target_predictions[oxide] = pred

        return target_predictions


@app.command(name="full_run", help="Run the full PLS-SM pipeline.")
def full_run(
    outlier_removal: bool = True,
    additional_info: str = "",
    outlier_removal_constraint_iteration: int = -1,
) -> pd.DataFrame:

    experiment = train(
        outlier_removal=outlier_removal,
        additional_info=additional_info,
        outlier_removal_constraint_iteration=outlier_removal_constraint_iteration,
    )

    target_predictions = test_run(
        experiment.experiment_id, outlier_removal, additional_info
    )

    return target_predictions


@app.command(name="test_run", help="Runs test PLS-SM pipeline.")
def test_run(
    train_experiment_id: str, outlier_removal: bool = True, additional_info: str = ""
):
    target_predictions = test(
        train_experiment_id,
        outlier_removal=outlier_removal,
        additional_info=additional_info,
    )

    return target_predictions


if __name__ == "__main__":
    app()
