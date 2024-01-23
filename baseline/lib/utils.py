import pandas as pd
from sklearn.model_selection import KFold, train_test_split


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

    for train_keys_idx, test_keys_idx in kf.split(groups_keys):
        train_keys = [groups_keys[idx] for idx in train_keys_idx]
        test_keys = [groups_keys[idx] for idx in test_keys_idx]

        train_data = pd.concat([grouped.get_group(key) for key in train_keys])
        test_data = pd.concat([grouped.get_group(key) for key in test_keys])

        yield train_data, test_data


def custom_train_test_split(data, group_by: str, test_size=0.2, random_state=None):
    """
    Custom train-test split function that splits the data based on
    a specified grouping variable.

    Parameters:
        data (pandas.DataFrame): The input data to be split.
        group_by (str): The column name to group the data by.
        test_size (float): The proportion of the data to be used for testing.
        Default is 0.2.
        random_state (int): The random seed for reproducibility.
        Default is None.

    Returns:
        train_data (pandas.DataFrame): The training data.
        test_data (pandas.DataFrame): The testing data.

    Usage:
        ```python
        train_data, test_data = custom_train_test_split(
            final_df, test_size=0.2
        )
        ```
    """
    grouped = data.groupby(group_by)
    groups_keys = list(grouped.groups.keys())

    train_keys, test_keys = train_test_split(
        groups_keys, test_size=test_size, random_state=random_state
    )

    train_data = pd.concat([grouped.get_group(key) for key in train_keys])
    test_data = pd.concat([grouped.get_group(key) for key in test_keys])

    return train_data, test_data


def filter_data_by_compositional_range(data, compositional_range, oxide, oxide_ranges):
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

    _data = data.copy(deep=True)

    _data[oxide] = pd.to_numeric(data[oxide], errors="coerce")
    _data = data.dropna(subset=[oxide])

    # Filter the dataset based on the oxide concentration within the specified range
    filtered_data = _data[(_data[oxide] >= lower_bound) & (_data[oxide] <= upper_bound)]

    # Check if empty now
    if filtered_data.empty:
        print(
            f"WARNING: No data found for {compositional_range} {oxide} ({lower_bound}-{upper_bound})"
        )

    return filtered_data
