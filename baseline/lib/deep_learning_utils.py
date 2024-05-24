from typing import List, Tuple

import keras
import mlflow
import pandas as pd

from lib import full_flow_dataloader
from lib.cross_validation import custom_kfold_cross_validation_new


def get_preprocess_fn(target_cols: List[str], drop_cols: List[str]):
    assert all(
        item in drop_cols for item in target_cols
    ), f"Target columns {target_cols} not in drop columns {drop_cols}"

    def preprocess_fn(train: pd.DataFrame):
        X = train.drop(columns=drop_cols)
        y = train[target_cols]

        X = X.to_numpy().reshape(-1, 6144, 1)
        y = y.to_numpy()

        return X, y

    return preprocess_fn


def get_data_nn(target: str, group_by: str) -> Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retrieves and processes data for the CNN experiment.

    This function concatenates the training and testing datasets, performs custom k-fold cross-validation,
    and returns the folds along with the training, validation, and test datasets.

    Parameters:
    - target (str): The target column name for the prediction task.

    Returns:
    - Tuple containing:
        - List of tuples: Each tuple contains a pair of DataFrames representing the training and testing data for each fold.
        - pd.DataFrame: The training dataset.
        - pd.DataFrame: The validation dataset.
        - pd.DataFrame: The test dataset.
    """
    train_processed, test_processed = full_flow_dataloader.load_full_flow_data(
        load_cache_if_exits=True, average_shots=True
    )
    data = pd.concat([train_processed.copy(), test_processed.copy()])

    # Do initial split of full dataset, yielding 5 folds, 4 for training, 1 for testing
    folds, train_unmod, test = custom_kfold_cross_validation_new(data, k=5, group_by=group_by, target=target)

    # Split the training data into training and validation sets
    _, train, val = custom_kfold_cross_validation_new(train_unmod, k=5, group_by=group_by, target=target)

    return folds, train, val, test


class MLFlowCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key, value in logs.items():
                mlflow.log_metric(f"{key}", value, step=epoch)

