import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# from config import logger
from lib.data_handling import CustomSpectralPipeline, load_data  # type: ignore
from lib.norms import Norm1Scaler, Norm3Scaler
from lib.outlier_removal import (
    calculate_leverage_residuals,
    identify_outliers,
    plot_leverage_residuals,
)
from lib.reproduction import (
    major_oxides,
    masks,
    oxide_ranges,
    paper_individual_sm_rmses,
    spectrometer_wavelength_ranges,
    training_info,
)
from lib.utils import (
    custom_kfold_cross_validation,
    custom_train_test_split,
    filter_data_by_compositional_range,
)

logger = logging.getLogger("train")

mlflow.set_tracking_uri("http://localhost:5000")

preformatted_data_path = Path(
    "./data/data/calib/calib_2015/1600mm/pls/all_processed.csv"
)

if not preformatted_data_path.exists():
    dataset_loc = Path("./data/data/calib/calib_2015/1600mm/pls/")
    calib_loc = Path("./data/data/calib/ccam_calibration_compositions.csv")
    take_samples = None

    logger.info("Loading data from location: %s", dataset_loc)
    data = load_data(str(dataset_loc))
    logger.info("Data loaded successfully.")

    logger.info("Initializing CustomSpectralPipeline.")
    pipeline = CustomSpectralPipeline(
        masks=masks,
        composition_data_loc=calib_loc,
        major_oxides=major_oxides,
    )
    logger.info("Pipeline initialized. Fitting and transforming data.")
    processed_data = pipeline.fit_transform(data)
    logger.info("Data processing complete.")

    processed_data.to_csv(preformatted_data_path, index=False)
else:
    logger.info("Loading preformatted data from location: %s", preformatted_data_path)
    processed_data = pd.read_csv(preformatted_data_path)

k_folds = 4
random_state = 42
influence_plot_dir = Path("plots/")
experiment_name = f"PLS_Models_{pd.Timestamp.now().strftime('%m-%d-%y_%H%M%S')}"

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
        data_filtered = filter_data_by_compositional_range(
            processed_data, compositional_range, oxide, oxide_ranges
        )

        # Separate 20% of the data for testing
        train, test = custom_train_test_split(
            data_filtered,
            group_by="Sample Name",
            test_size=0.2,
            random_state=random_state,
        )

        train_cols = train.columns
        test_cols = test.columns

        n_components = training_info[oxide][compositional_range]["n_components"]
        norm = training_info[oxide][compositional_range]["normalization"]
        scaler = (
            Norm1Scaler(reshaped=True)
            if norm == 1
            else Norm3Scaler(spectrometer_wavelength_ranges, reshaped=True)
        )
        logger.debug("Initializing scaler: %s", scaler.__class__.__name__)

        logger.debug("Fitting and transforming training data.")
        train = scaler.fit_transform(train)
        logger.debug("Transforming test data.")
        test = scaler.fit_transform(test)

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
            drop_cols = major_oxides + ["Sample Name"]
            X_train_OR = train.drop(columns=drop_cols).to_numpy()
            y_train_OR = train[oxide].to_numpy()

            pls_OR.fit(X_train_OR, y_train_OR)

            current_performance = mean_squared_error(
                y_train_OR, pls_OR.predict(X_train_OR)
            )

            mlflow.log_metric(
                "RMSEOR", float(current_performance), step=outlier_removal_iterations
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
                    y_train_OR, pls_OR.predict(X_train_OR)
                )

                mlflow.log_metric(
                    "RMSEOR", float(new_performance), step=outlier_removal_iterations
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

            mlflow.log_metric("outlier_removal_iterations", outlier_removal_iterations)
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
            mlflow.sklearn.log_model(pls_all, f"PLS_ALL_{oxide}_{compositional_range}")
            # endregion
        mlflow.end_run()
