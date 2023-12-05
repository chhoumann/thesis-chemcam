import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import typer
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from ..reproduction import (
    major_oxides,
    masks,
    oxide_ranges,
    paper_individual_sm_rmses,
)
from .config import logger
from ..lib.data_handling import CustomSpectralPipeline, load_data
from ..lib.utils import (
    custom_kfold_cross_validation,
    custom_train_test_split,
    filter_data_by_compositional_range,
)

# Initialize MLflow
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("PLS_Models")

app = typer.Typer()

logger.setLevel(logging.WARNING)


@app.command()
def train_model(
    experiment_name: str, dataset_loc: str, calib_loc: str, take_samples=None
):
    logger.info("Loading data from location: %s", dataset_loc)
    data = load_data(dataset_loc, take_samples)
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

    k_folds = 5
    random_state = 42
    n_components = 25  # paper said 20-30

    for oxide in tqdm(major_oxides, desc="Processing oxides"):
        _oxide_ranges = oxide_ranges.get(oxide, None)
        if _oxide_ranges is None:
            logger.info("Skipping oxide: %s", oxide)
            continue

        for compositional_range in _oxide_ranges.keys():
            logger.info(
                "Starting MLflow run for compositional range: %s, oxide: %s",
                compositional_range,
                oxide,
            )
            with mlflow.start_run(
                run_name=f"{experiment_name}_{compositional_range}_{oxide}"
            ):
                best_model = None
                best_rmse = float("inf")
                mlflow.log_param("n_components", n_components)
                mlflow.log_param("random_state", random_state)
                logger.info("Filtering data by compositional range.")
                data_filtered = filter_data_by_compositional_range(
                    processed_data, compositional_range, oxide, oxide_ranges
                )

                train, test = custom_train_test_split(
                    data_filtered,
                    group_by="Sample Name",
                    test_size=0.2,
                    random_state=random_state,
                )

                logger.info("Performing custom k-fold cross-validation.")
                kf = custom_kfold_cross_validation(
                    train,
                    k=k_folds,
                    group_by="Sample Name",
                    random_state=random_state,
                )

                fold_rmse = []
                for i, (train_data, test_data) in enumerate(kf):
                    logger.info("Defining PLSRegression model.")
                    pls = PLSRegression(
                        n_components=n_components
                    )  # Adjust n_components as needed

                    logger.info("Extracting features and target for training.")
                    X_train = train_data.drop(columns=major_oxides + ["Sample Name"])
                    y_train = train_data[oxide]
                    logger.info("Extracting features and target for testing.")
                    X_test = test_data.drop(columns=major_oxides + ["Sample Name"])
                    y_test = test_data[oxide]

                    logger.info("Training the model.")
                    pls.fit(X_train, y_train)
                    logger.info("Model training complete.")

                    logger.info("Predicting on test data.")
                    y_pred = pls.predict(X_test)
                    rmse = mean_squared_error(y_test, y_pred, squared=False)
                    fold_rmse.append(rmse)
                    logger.info("Fold RMSE: %f", rmse)

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = pls

                avg_rmse = sum(fold_rmse) / k_folds
                logger.info("Logging parameters, metrics, and model to MLflow.")
                mlflow.log_param("compositional_range", compositional_range)
                mlflow.log_param("oxide", oxide)
                mlflow.log_param("k_folds", k_folds)
                mlflow.log_metric("avg_rmse", float(avg_rmse))
                mlflow.log_metric("best_rmse", float(best_rmse))
                mlflow.log_metric(
                    "paper_rmse", paper_individual_sm_rmses[compositional_range][oxide]
                )
                mlflow.sklearn.log_model(
                    best_model,
                    "model",
                    registered_model_name=f"{oxide}_{compositional_range}",
                )

                # ----- Influence Plots for Outlier Removal ----- #

                pls = PLSRegression(n_components=n_components)
                train_data = train.drop(columns=major_oxides + ["Sample Name"])
                X_train = train_data.to_numpy()
                Y_train = train[oxide].to_numpy()
                pls.fit(X_train, Y_train)

                # Calculate leverage
                t = pls.x_scores_
                leverage = np.diag(
                    np.dot(t, np.dot(np.linalg.inv(np.dot(t.T, t)), t.T))
                )

                # Calculate residuals
                X_reconstructed = np.dot(t, pls.x_loadings_.T)
                residuals = X_train - X_reconstructed
                Q = np.sum(residuals**2, axis=1)

                # Plotting the influence plot
                plt.scatter(leverage, Q)
                plt.xlabel("Leverage")
                plt.ylabel("Residuals")
                plt.title("Influence Plot")
                plot_path = Path(
                    f"plots/{experiment_name}/{oxide}_{compositional_range}_ip.png"
                )
                if not plot_path.parent.exists():
                    plot_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(plot_path)
                plt.close()

                mlflow.log_artifact(str(plot_path))

                # # Identify outliers (this step is more qualitative and depends on your specific dataset)
                # outliers = identify_outliers(
                #     leverage, Q
                # )  # Implement this function based on your criteria

                # # Remove outliers and repeat the process
                # X_train = np.delete(X_train, outliers, axis=0)
                # Y_train = np.delete(Y_train, outliers, axis=0)

                logger.info(
                    "Compositional Range: %s, Oxide: %s, Average RMSE: %f",
                    compositional_range,
                    oxide,
                    avg_rmse,
                )


if __name__ == "__main__":
    # if ray.is_initialized():
    #     ray.shutdown()
    # ray.init()
    # app()  # initialize Typer app
    train_model(
        "full_pls_sm_2",
        "data/data/calib/calib_2015/1600mm/pls",
        "data/data/calib/ccam_calibration_compositions.csv",
        # 100,
    )
