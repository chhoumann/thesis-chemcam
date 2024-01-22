import logging
from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
from dotenv import dotenv_values
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# from config import logger
from lib.data_handling import CustomSpectralPipeline, load_split_data  # type: ignore
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
    spectrometer_wavelength_ranges,
    training_info,
)
from lib.utils import custom_kfold_cross_validation, filter_data_by_compositional_range
from PLS_SM.inference import predict_composition_with_blending

env = dotenv_values()
comp_data_loc = env.get("COMPOSITION_DATA_PATH")
dataset_loc = env.get("DATA_PATH")

if not comp_data_loc:
    print("Please set COMPOSITION_DATA_PATH in .env file")
    exit(1)

if not dataset_loc:
    print("Please set DATA_PATH in .env file")
    exit(1)

logger = logging.getLogger("train")

mlflow.set_tracking_uri("http://localhost:5000")

preformatted_data_path = Path("./data/_preformatted_sm/")
train_path = preformatted_data_path / "train.csv"
test_path = preformatted_data_path / "test.csv"

if (
    not preformatted_data_path.exists()
    or not train_path.exists()
    or not test_path.exists()
):
    take_samples = None

    logger.info("Loading data from location: %s", dataset_loc)
    # data = load_data(str(dataset_loc))
    train_data, test_data = load_split_data(
        str(dataset_loc), split_loc="./train_test_split.csv", average_shots=True
    )
    logger.info("Data loaded successfully.")

    logger.info("Initializing CustomSpectralPipeline.")
    pipeline = CustomSpectralPipeline(
        masks=masks,
        composition_data_loc=comp_data_loc,
        major_oxides=major_oxides,
    )
    logger.info("Pipeline initialized. Fitting and transforming data.")
    train_processed = pipeline.fit_transform(train_data)
    test_processed = pipeline.fit_transform(test_data)
    logger.info("Data processing complete.")

    preformatted_data_path.mkdir(parents=True, exist_ok=True)

    train_processed.to_csv(train_path, index=False)
    test_processed.to_csv(test_path, index=False)
else:
    logger.info("Loading preformatted data from location: %s", preformatted_data_path)
    train_processed = pd.read_csv(train_path)
    test_processed = pd.read_csv(test_path)

SHOULD_TRAIN = False
SHOULD_PREDICT = True

if SHOULD_TRAIN:
    k_folds = 4
    random_state = 42
    influence_plot_dir = Path("plots/")
    experiment_name = f"PLS_Models_{pd.Timestamp.now().strftime('%m-%d-%y_%H%M%S')}"
    # experiment_name = "PLS_Models_Train_Test_Split"

    mlflow.set_experiment(experiment_name)
    mlflow.autolog()

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

            logger.info("Filtering data by compositional range.")
            train_data_filtered = filter_data_by_compositional_range(
                train_processed, compositional_range, oxide, oxide_ranges
            )

            test_data_filtered = filter_data_by_compositional_range(
                train_processed, compositional_range, oxide, oxide_ranges
            )

            # We don't do this anymore because we already have the split in train_test_split.csv
            # Separate 20% of the data for testing
            # train, test = custom_train_test_split(
            #     data_filtered,
            #     group_by="Sample Name",
            #     test_size=0.2,
            #     random_state=random_state,
            # )

            train_cols = train_data_filtered.columns
            test_cols = test_data_filtered.columns

            n_components = training_info[oxide][compositional_range]["n_components"]
            norm = training_info[oxide][compositional_range]["normalization"]
            scaler = (
                Norm1Scaler(reshaped=True)
                if norm == 1
                else Norm3Scaler(spectrometer_wavelength_ranges, reshaped=True)
            )
            logger.debug("Initializing scaler: %s", scaler.__class__.__name__)

            logger.debug("Fitting and transforming training data.")
            train = scaler.fit_transform(train_data_filtered)
            logger.debug("Transforming test data.")
            test = scaler.fit_transform(test_data_filtered)

            # turn back into dataframe
            train = pd.DataFrame(train, columns=train_cols)
            test = pd.DataFrame(test, columns=test_cols)

            with mlflow.start_run(run_name=f"{oxide}_{compositional_range}"):
                mlflow.log_metric(
                    "paper_rmse", paper_individual_sm_rmses[compositional_range][oxide]
                )

                # region OUTLIER REMOVAL
                mlflow.log_params(
                    {
                        "masks": masks,
                        "range": oxide_ranges[oxide][compositional_range],
                        "compositional_range": compositional_range,
                        "oxide": oxide,
                        "n_spectra": len(train),
                    }
                )

                outlier_removal_iterations = 0
                pls_OR = PLSRegression(n_components=n_components)
                drop_cols = major_oxides + ["Sample Name", "ID"]
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

                while True:
                    outlier_removal_iterations += 1
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
                    if not plot_path.parent.exists():
                        plot_path.parent.mkdir(parents=True)

                    plot_leverage_residuals(leverage, Q, outliers, str(plot_path))
                    mlflow.log_artifact(str(plot_path))

                    X_train_OR = np.delete(X_train_OR, outliers_indices, axis=0)
                    y_train_OR = np.delete(y_train_OR, outliers_indices, axis=0)

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
                    if new_performance >= current_performance:
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
                for i, (train_data, test_data) in enumerate(kf):
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

                    if fold_rmse < best_CV_rmse:
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
                test_rmse = mean_squared_error(y_test, y_pred, squared=False)
                mlflow.log_metric("RMSEP", float(test_rmse))
                mlflow.sklearn.log_model(
                    pls_all, f"PLS_ALL_{oxide}_{compositional_range}"
                )
                # endregion
            mlflow.end_run()


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


if SHOULD_PREDICT:
    experiment_name = f"PLS_TEST_{pd.Timestamp.now().strftime('%m-%d-%y_%H%M%S')}"

    mlflow.set_experiment(experiment_name)
    mlflow.autolog(log_models=False, log_datasets=False)

    models = get_models(experiment_id="288133286244787831")

    # save na to csv
    test_processed[test_processed.isna().any(axis=1)].to_csv(
        "data/data/PLS_SM/na.csv", index=False
    )
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

    n1_scaler = Norm1Scaler(reshaped=True)
    n3_scaler = Norm3Scaler(spectrometer_wavelength_ranges, reshaped=True)

    X_test_n1 = n1_scaler.fit_transform(test_processed.drop(drop_cols, axis=1))
    X_test_n3 = n3_scaler.fit_transform(test_processed.drop(drop_cols, axis=1))

    save_path = Path("data/data/PLS_SM/")
    predictions_save_path = save_path / "predictions"

    predictions_save_path.mkdir(parents=True, exist_ok=True)

    for oxide in tqdm(major_oxides, desc="Predicting oxides"):
        with mlflow.start_run(run_name=f"PLS_TEST_{oxide}"):
            _oxide_ranges = oxide_ranges.get(oxide, None)
            if _oxide_ranges is None:
                logger.info("Skipping oxide: %s", oxide)
                continue

            pred = predict_composition_with_blending(
                oxide, X_test_n1, X_test_n3, models, optimized_blending_ranges
            )

            # save
            pred_df = pd.DataFrame(pred, index=Y.index)
            pred_df.to_csv(predictions_save_path / f"{oxide}.csv")

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

            # calculate RMSEP
            rmsep = mean_squared_error(Y[oxide], pred_df, squared=False)
            mlflow.log_metric("RMSEP", float(rmsep))
            # save
            with open(save_path / "rmsep.txt", "a") as f:
                f.write(f"{oxide}: {rmsep}\n")

            target_predictions[oxide] = pred

    target_predictions.to_csv(save_path / "tar_pred.csv")


# endregion
