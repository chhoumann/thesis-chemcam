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


class MLFlowCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key, value in logs.items():
                mlflow.log_metric(f"{key}", value, step=epoch)

