import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedGroupKFold

from lib.outlier_removal import global_local_outlier_removal
from lib.reproduction import major_oxides


def sort_and_assign_folds(data, group_by, target, n_splits=5, ensure_extremes_in_train=True):
    # Sort the data by the target column
    unique_samples = (
        data.drop_duplicates(subset=[group_by])[[group_by, target]].sort_values(by=target).reset_index(drop=True)
    )

    # Assign fold numbers sequentially
    unique_samples["fold"] = np.arange(len(unique_samples)) % n_splits

    if ensure_extremes_in_train:
        # Identify extreme values (e.g., top and bottom 5%)
        quantiles = unique_samples[target].quantile([0.05, 0.95])
        extremes = unique_samples[
            (unique_samples[target] <= quantiles[0.05]) | (unique_samples[target] >= quantiles[0.95])
        ]

        # Set extreme values to training folds (0 to n_splits-2)
        unique_samples.loc[extremes.index, "fold"] = np.random.randint(0, n_splits - 1, len(extremes))

    # Merge fold assignments back to the original data
    data = data.merge(unique_samples[[group_by, "fold"]], on=group_by, how="left")
    return data


def custom_kfold_cross_validation_new(data, k: int, group_by: str, target: str, random_state=None):
    """
    Perform custom k-fold cross-validation with outlier removal.

    Parameters:
    - data: The input data to be split into train and test sets.
    - k: The number of folds for cross-validation.
    - group_by: The column name to group the data by.
    - target: The target column name for sorting and assigning folds.
    - random_state: The random seed for reproducibility.

    Returns:
    - folds_custom: List of tuples containing train and validation sets for each fold.
    - test_data: The test set.
    """
    # Sort and assign folds
    data = sort_and_assign_folds(data, group_by, target, n_splits=k)

    # Identify the test fold (the one with the highest fold number)
    test_fold_number = k - 1
    test_data = data[data["fold"] == test_fold_number]

    train_data = data[data["fold"] != test_fold_number]
    outlier_mask = global_local_outlier_removal(train_data, drop_cols=["Sample Name", "ID", "fold"] + major_oxides)
    train_data = train_data[~outlier_mask]

    # Perform cross-validation excluding the test fold
    folds_custom = []
    for i in range(k - 1):
        validation_data = train_data[train_data["fold"] == i]
        folds_custom.append((train_data[train_data["fold"] != i], validation_data))

    # Remove the 'fold' column from all folds and test data
    folds_custom = [(train.drop(columns=["fold"]), val.drop(columns=["fold"])) for train, val in folds_custom]
    test_data = test_data.drop(columns=["fold"])

    return folds_custom, test_data


def grouped_train_test_split(data, test_size: float, group_by: str, random_state=None):
    """
    Perform a simple train/test split on the given data.

    Parameters:
    - data: The input data to be split into train and test sets.
    - test_size: The proportion of the dataset to include in the test split.
    - group_by: The column name to group the data by.
    - random_state: The random seed for shuffling the data.

    Returns:
    - train_data: The training set.
    - test_data: The testing set.

    Usage:
        ```python
        train_data, test_data = simple_train_test_split(
            final_df, test_size=0.2, group_by='category'
        )
        # Apply training and validation using train_data and test_data
        ```
    """
    grouped = data.groupby(group_by)
    groups_keys = list(grouped.groups.keys())

    if random_state is not None:
        random.seed(random_state)
    random.shuffle(groups_keys)

    split_idx = int((1 - test_size) * len(groups_keys))
    train_keys = groups_keys[:split_idx]
    test_keys = groups_keys[split_idx:]

    train_data = pd.concat([grouped.get_group(key) for key in train_keys])
    test_data = pd.concat([grouped.get_group(key) for key in test_keys])

    return train_data, test_data


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


def stratified_group_kfold_split(
    data: pd.DataFrame, group_by: str, target: str, num_bins: int = 5, n_splits: int = 5, random_state: int = 42
):
    """
    Perform a Stratified Group K-Fold split on the given data.

    Parameters:
    - data: The dataset to be used.
    - group_by: The column name to group by.
    - target: The target column name for stratification.
    - num_bins: The number of bins for stratification.
    - n_splits: The number of splits for cross-validation.
    - random_state: The random state for reproducibility.

    Returns:
    - List of tuples containing train and test indices for each fold.
    """
    # Step 1: Extract unique sample names and their target values
    unique_samples = data.drop_duplicates(subset=[group_by])[[group_by, target]]

    # Bin the target values into quantiles
    unique_samples[f"{target}_bin"] = pd.qcut(unique_samples[target], q=num_bins, labels=False, duplicates="drop")

    # Step 2: Stratified Group K-Fold Split
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Create empty lists to store train and test indices for each fold
    folds_custom = []

    X = unique_samples[group_by]
    y = unique_samples[f"{target}_bin"]
    groups = unique_samples[group_by]

    for train_idx, test_idx in sgkf.split(X, y, groups):  # type: ignore
        train_samples = unique_samples.iloc[train_idx][group_by]
        test_samples = unique_samples.iloc[test_idx][group_by]

        # Ensure no overlap between train and test samples
        assert not set(train_samples).intersection(set(test_samples)), "Overlap detected between train and test samples"

        train_mask = data[group_by].isin(train_samples)
        test_mask = data[group_by].isin(test_samples)

        # Ensure no overlap between train and test masks
        overlap = data[train_mask & test_mask]
        if not overlap.empty:
            print("Overlap detected between train and test masks:", overlap)
        assert overlap.empty, "Overlap detected between train and test masks"

        train_idx_full = data[train_mask]  # .index
        test_idx_full = data[test_mask]  # .index

        folds_custom.append((train_idx_full, test_idx_full))

    return folds_custom


class StratifiedGroupKFoldSplit:
    def __init__(self, group_by: str, target: str, n_splits: int = 5, shuffle=True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.group_by = group_by
        self.target = target

    def split(self, data):
        """
        Perform stratified group k-fold split.

        Parameters:
        - data: The dataset to be used.
        - group_by: The column name to group by.
        - target: The target column name for stratification.
        - num_bins: The number of bins for stratification.

        Returns:
        - List of tuples containing train and test indices for each fold.
        """
        return stratified_group_kfold_split(
            data,
            self.group_by,
            self.target,
            num_bins=self.n_splits,
            n_splits=self.n_splits,
            random_state=self.random_state,
        )

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def perform_cross_validation(
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    model,
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
    if not hasattr(model, "fit") or not hasattr(model, "predict"):
        raise ValueError("Model must have 'fit' and 'predict' methods.")
    from joblib import Parallel, delayed

    def process_fold(i, train_data, test_data) -> List[float]:
        print(f"Running fold {i+1} with size: {len(train_data)} train and {len(test_data)} test")

        X_train, y_train, X_test, y_test = preprocess_fn(train_data, test_data)

        # Ensure arrays are writable
        X_train = np.array(X_train, copy=True)
        y_train = np.array(y_train, copy=True)
        X_test = np.array(X_test, copy=True)
        y_test = np.array(y_test, copy=True)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_metrics = [metric_fn(y_test, y_pred) for metric_fn in metric_fns]
        return fold_metrics

    all_fold_metrics: List[List[float]] = Parallel(n_jobs=-1)(
        delayed(process_fold)(i, train_data, test_data) for i, (train_data, test_data) in enumerate(folds)
    )

    return all_fold_metrics


@dataclass
class CrossValidationMetrics:
    rmse_cv: float
    std_dev_cv: float
    rmse_cv_folds: Dict[str, float]
    std_dev_cv_folds: Dict[str, float]

    def __init__(self, all_fold_metrics: List[List[float]]):
        # Calculate mean RMSE and Standard Deviation across all folds
        self.rmse_cv = np.mean([rmse for rmse, _ in all_fold_metrics])
        self.std_dev_cv = np.mean([std_dev for _, std_dev in all_fold_metrics])
        # Extract RMSE and Standard Deviation metrics for each fold
        self.rmse_cv_folds = {f"rmse_cv_{i+1}": rmse for i, (rmse, _) in enumerate(all_fold_metrics)}
        self.std_dev_cv_folds = {f"std_dev_cv_{i+1}": std_dev for i, (_, std_dev) in enumerate(all_fold_metrics)}

    def as_dict(self):
        return {
            "rmse_cv": self.rmse_cv,
            "std_dev_cv": self.std_dev_cv,
            **self.rmse_cv_folds,
            **self.std_dev_cv_folds,
        }


def get_cross_validation_metrics(all_fold_metrics: List[List[float]]):
    """Get the cross-validation metrics for a list of fold metrics.

    Args:
        all_fold_metrics (List[List[float]]): A list of lists, each sublist contains computed metrics for each fold.

    Returns:
        CrossValidationMetrics: An object containing the averaged cross-validation metrics.
    """
    return CrossValidationMetrics(all_fold_metrics)


def perform_repeated_cross_validation(
    get_kf: Callable[..., Union[CustomKFoldCrossValidator, StratifiedGroupKFoldSplit]],
    data: pd.DataFrame,
    model,
    preprocess_fn: Callable[[pd.DataFrame, pd.DataFrame], tuple],
    metric_fns: List[Callable[[np.ndarray, np.ndarray], float]],
    n_repeats: int = 5,
    random_state: int = 42,
) -> CrossValidationMetrics:
    """
    Perform repeated cross-validation with different random seeds.

    Parameters:
    - all_fold_metrics: A list of lists, each sublist contains computed metrics for each fold.
    - kf_class: The KFold or similar cross-validator class.
    - data: The dataset to be used.
    - model: The regression model to be trained.
    - preprocess_fn: A function that takes train and test data DataFrames and returns
                     the tuples (X_train, y_train) and (X_test, y_test).
    - metric_fns: List of functions to compute the metrics, each takes two arguments: y_true and y_pred.
    - n_repeats: Number of times to repeat the cross-validation.
    - random_state: The random seed for reproducibility.

    Returns:
        CrossValidationMetrics: An object containing the averaged cross-validation metrics.
    """
    all_metrics = []

    for i in range(n_repeats):
        new_kf_class = get_kf(random_state=random_state + i)
        fold_metrics = perform_cross_validation(
            kf=new_kf_class,
            data=data,
            model=model,
            preprocess_fn=preprocess_fn,
            metric_fns=metric_fns,
        )
        all_metrics.extend(fold_metrics)

    return CrossValidationMetrics(all_metrics)
