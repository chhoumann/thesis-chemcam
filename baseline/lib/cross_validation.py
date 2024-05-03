import random
from typing import Callable, List, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, KFold


def custom_kfold_cross_validation(data, k: int, group_by: str, random_state=None):
    """
    Perform custom k-fold cross-validation on the given data.

    Parameters:
    - data: The input data to be split into train and test sets.
    - k: The number of folds for cross-validation.
    - group_by: The column name to group the data by.
    - random_state: The random seed for shuffling the data.

    Returns:
    - A generator that yields train and test sets for each fold.

    Usage:
        ```python
        for train_data, test_data in custom_kfold_cross_validation(
            final_df, k=5
        ):
            # Apply training and validation using train_data and test_data
        ```
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    grouped = data.groupby(group_by)
    groups_keys = list(grouped.groups.keys())

    if random_state is not None:
        random.seed(random_state)
    random.shuffle(groups_keys)

    for train_keys_idx, test_keys_idx in kf.split(groups_keys):
        train_keys = [groups_keys[idx] for idx in train_keys_idx]
        test_keys = [groups_keys[idx] for idx in test_keys_idx]

        train_data = pd.concat([grouped.get_group(key) for key in train_keys])
        test_data = pd.concat([grouped.get_group(key) for key in test_keys])

        yield train_data, test_data


class CustomKFoldCrossValidator(BaseCrossValidator):
    def __init__(self, k: int, group_by: str, random_state=None):
        self.k = k
        self.group_by = group_by
        self.random_state = random_state

    def split(self, data, y=None, groups=None):
        return custom_kfold_cross_validation(data, self.k, self.group_by, self.random_state)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.k


@runtime_checkable
class FitPredictProtocol(Protocol):
    def fit(self, X, y) -> None: ...

    def predict(self, X) -> np.ndarray: ...


def perform_cross_validation(
    kf: CustomKFoldCrossValidator,
    data: pd.DataFrame,
    model: FitPredictProtocol,
    preprocess_fn: Callable[[pd.DataFrame, pd.DataFrame], tuple],
    metric_fns: List[Callable[[np.ndarray, np.ndarray], float]],
) -> List[List[float]]:
    """
    Perform cross-validation using a custom preprocessing function and multiple metric functions.

    Parameters:
    - kf: The KFold or similar cross-validator object.
    - data: The dataset to be used.
    - model: The regression model to be trained.
    - preprocess_fn: A function that takes train and test data DataFrames and returns
                     the tuples (X_train, y_train) and (X_test, y_test).
    - metric_fns: Tuple of functions to compute the metrics, each takes two arguments: y_true and y_pred.

    Returns:
    - List[List[float]]: A list of lists, each sublist contains computed metrics for each fold.
    """
    all_fold_metrics: List[List[float]] = []
    for i, (train_data, test_data) in enumerate(kf.split(data)):
        X_train, y_train, X_test, y_test = preprocess_fn(train_data, test_data)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_metrics = [metric_fn(y_test, y_pred) for metric_fn in metric_fns]
        all_fold_metrics.append(fold_metrics)

    return all_fold_metrics
