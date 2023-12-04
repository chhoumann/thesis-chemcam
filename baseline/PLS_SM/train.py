import mlflow
import pandas as pd
import typer
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from .config import logger
from .data import CustomSpectralPipeline, load_data
from .reproduction import (
    major_oxides,
    masks,
    oxide_ranges,
    paper_rmses_full_model,
)
from .utils import custom_kfold_cross_validation

# Initialize MLflow
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("PLS_Models")

app = typer.Typer()


def filter_data_by_compositional_range(data, compositional_range, oxide):
    """
    Filter the dataset for a given compositional range and oxide.

    Parameters:
    - data (pd.DataFrame): The dataset to filter.
    - compositional_range (str): The compositional range ('Full', 'Low', 'Mid', 'High').
    - oxide (str): The oxide to filter by.

    Returns:
    - pd.DataFrame: The filtered dataset.
    """
    # Access the global oxide_ranges dictionary
    # Get the lower and upper bounds for the specified compositional range and oxide
    lower_bound, upper_bound = oxide_ranges[oxide][compositional_range]

    data[oxide] = pd.to_numeric(data[oxide], errors="coerce")
    data = data.dropna(subset=[oxide])

    # Filter the dataset based on the oxide concentration within the specified range
    filtered_data = data[
        (data[oxide] >= lower_bound) & (data[oxide] <= upper_bound)
    ]

    return filtered_data


def train_step():
    pass


def val_step():
    pass


def train_loop_per_worker():
    pass


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

    compositional_range = "Full"
    k_folds = 5
    random_state = 42
    best_model = None
    best_rmse = float("inf")
    n_components = 8

    for oxide in tqdm(major_oxides, desc="Processing oxides"):
        logger.info(
            "Starting MLflow run for compositional range: %s, oxide: %s",
            compositional_range,
            oxide,
        )
        with mlflow.start_run(run_name=f"{experiment_name}_{oxide}"):
            mlflow.log_param("n_components", n_components)
            mlflow.log_param("random_state", random_state)
            logger.info("Filtering data by compositional range.")
            data_filtered = filter_data_by_compositional_range(
                processed_data, compositional_range, oxide
            )

            logger.info("Performing custom k-fold cross-validation.")
            kf = custom_kfold_cross_validation(
                data_filtered,
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
                X_train = train_data.drop(
                    columns=major_oxides + ["Sample Name"]
                )
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
            mlflow.log_metric("paper_rmse", paper_rmses_full_model[oxide])
            mlflow.sklearn.log_model(
                best_model,
                "model",
                registered_model_name=f"Best_{oxide}_full_model",
            )

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
        "full_models_k_fold",
        "data/data/calib/calib_2015/1600mm/pls",
        "data/data/calib/ccam_calibration_compositions.csv",
        # 100,
    )
