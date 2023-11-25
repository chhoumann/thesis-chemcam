import mlflow
import pandas as pd
import typer
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

from .config import logger
from .data import CustomSpectralPipeline, load_data
from .reproduction import major_oxides, masks, oxide_ranges
from .utils import custom_train_test_split

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
    filtered_data = data[(data[oxide] >= lower_bound) & (data[oxide] <= upper_bound)]

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
    logger.debug("Loading data from location: %s", dataset_loc)
    data = load_data(dataset_loc, take_samples)
    logger.debug("Data loaded successfully.")

    logger.debug("Initializing CustomSpectralPipeline.")
    pipeline = CustomSpectralPipeline(
        masks=masks,
        composition_data_loc=calib_loc,
        major_oxides=major_oxides,
    )
    logger.debug("Pipeline initialized. Fitting and transforming data.")
    processed_data = pipeline.fit_transform(data)
    logger.debug("Data processing complete.")

    logger.debug("Splitting dataset into train and test sets.")
    train_data, test_data = custom_train_test_split(
        processed_data, group_by="Sample Name", test_size=0.2
    )
    logger.debug(
        "Dataset split complete. Train size: %d, Test size: %d",
        len(train_data),
        len(test_data),
    )

    compositional_range = "Full"

    for oxide in major_oxides:
        logger.debug(
            "Starting MLflow run for compositional range: %s, oxide: %s",
            compositional_range,
            oxide,
        )
        with mlflow.start_run(run_name=f"{experiment_name}_{oxide}"):
            logger.debug("Filtering train data.")
            train_data_filtered = filter_data_by_compositional_range(
                train_data, compositional_range, oxide
            )
            logger.debug("Filtering test data.")
            test_data_filtered = filter_data_by_compositional_range(
                test_data, compositional_range, oxide
            )

            logger.debug("Defining PLSRegression model.")
            pls = PLSRegression(n_components=2)  # Adjust n_components as needed

            logger.debug("Extracting features and target for training.")
            X_train = train_data_filtered.drop(columns=major_oxides + ["Sample Name"])
            y_train = train_data_filtered[oxide]
            logger.debug("Extracting features and target for testing.")
            X_test = test_data_filtered.drop(columns=major_oxides + ["Sample Name"])
            y_test = test_data_filtered[oxide]

            logger.debug("Training the model.")
            pls.fit(X_train, y_train)
            logger.debug("Model training complete.")

            logger.debug("Predicting on test data.")
            y_pred = pls.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            logger.debug("Prediction complete. RMSE: %f", rmse)

            logger.debug("Logging parameters, metrics, and model to MLflow.")
            mlflow.log_param("compositional_range", compositional_range)
            mlflow.log_param("oxide", oxide)
            mlflow.log_metric("rmse", float(rmse))
            mlflow.sklearn.log_model(pls, "model")

            logger.info(
                "Compositional Range: %s, Oxide: %s, RMSE: %f",
                compositional_range,
                oxide,
                rmse,
            )


if __name__ == "__main__":
    # if ray.is_initialized():
    #     ray.shutdown()
    # ray.init()
    # app()  # initialize Typer app
    train_model(
        "full_model",
        "data/data/calib/calib_2015/1600mm/pls",
        "data/data/calib/ccam_calibration_compositions.csv",
        100,
    )
