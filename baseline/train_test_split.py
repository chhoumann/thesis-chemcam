import os
from enum import Enum
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from lib.config import AppConfig
from lib.data_handling import CompositionData
from lib.reproduction import folder_to_composition_sample_name, major_oxides


class CalibrationDataFilter(Enum):
    Used_2015_Calibration = "Used for 2015 calibration"
    Used_2021_Mn_Calibration = "Used for 2021 Mn calibration"
    Used_2022_Li_Calibration = "Used for 2022 Li calibration"


def get_all_samples(cd: CompositionData, dataset_loc: str):
    samples = os.listdir(dataset_loc)

    sample_compositions = []

    for sample in samples:
        sample_name = folder_to_composition_sample_name.get(sample, sample)
        composition = cd.get_composition_for_sample(sample_name)

        if composition.empty:
            continue

        if composition[major_oxides].isnull().values.any():
            continue

        composition["sample_name"] = sample

        sample_compositions.append(composition)

    samples_df = pd.concat(sample_compositions)

    for oxide in major_oxides:
        samples_df[oxide] = pd.to_numeric(samples_df[oxide], errors="coerce")

    return samples_df


def filter_samples(
    samples: pd.DataFrame, filters: List[CalibrationDataFilter] = []
) -> pd.DataFrame:
    """
    Filter samples based on a list of CalibrationFilters.

    Parameters:
    - samples (pd.DataFrame): The samples dataframe to filter.
    - filters (List[CalibrationFilter]): A list of CalibrationFilter enum members to filter by. If a sample has a value of 1 in any of the columns corresponding to the specified filters, it will be included in the returned dataframe.

    Returns:
    - pd.DataFrame: A dataframe filtered by the specified CalibrationFilters.
    """
    if len(filters) == 0:
        return samples

    # Convert the list of CalibrationFilter enums to a list of their corresponding column names
    filter_columns = [filter.value for filter in filters]

    # Create a mask that is True for rows where any of the specified columns have a value of 1
    mask = samples[filter_columns].any(axis=1)

    # Return the filtered dataframe
    return samples[mask]


def create_train_test_split(
    cd: CompositionData, dataset_loc: str, filter: List[CalibrationDataFilter] = []
):
    """
    Create a randomized train-test split of samples based on the given CompositionData and dataset location.

    Args:
        cd (CompositionData): The CompositionData object containing the samples.
        dataset_loc (str): The location of the dataset.
        filter (List[CalibrationDataFilter], optional): A list of filters to apply on the samples. Defaults to [].

    Returns:
        pd.DataFrame: A DataFrame containing the train-test split of samples.
    """
    samples = get_all_samples(cd, dataset_loc)
    filtered_samples = filter_samples(samples, filter)

    train, test = train_test_split(filtered_samples, test_size=0.2, random_state=42)

    train_df = pd.DataFrame(train, columns=["sample_name"])
    train_df["train_test"] = "train"

    test_df = pd.DataFrame(test, columns=["sample_name"])
    test_df["train_test"] = "test"

    train_test_df = pd.concat([train_df, test_df])
    return train_test_df


def create_train_test_split_with_extremes(
    samples: pd.DataFrame,
    n_extremes: int = 2,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Create a train-test split with extreme samples prioritized to be in the training set.

    Args:
        samples (pd.DataFrame): The input DataFrame containing the samples.
        n_extremes (int, optional): The number of extreme samples to include. Defaults to 2.
        test_size (float, optional): The proportion of samples to include in the test set. Defaults to 0.2.
        random_state (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: The train-test split DataFrame.
    """
    if n_extremes < 1:
        raise ValueError("n_extremes must be at least 1")

    extreme_samples = pd.DataFrame()

    for oxide in major_oxides:
        top_extremes = samples.nlargest(n_extremes, oxide)
        bottom_extremes = samples.nsmallest(n_extremes, oxide)

        extreme_samples = pd.concat([extreme_samples, top_extremes, bottom_extremes])

    extreme_samples.drop_duplicates(inplace=True)

    # Calculate adjusted split size for normal samples
    # to account for the inclusion of extreme samples in the training set
    num_extreme_samples = len(extreme_samples)
    total_samples = len(samples)
    adjusted_test_size = (test_size * total_samples) / (
        total_samples - num_extreme_samples
    )

    # Ensure the adjusted test size does not exceed 1
    adjusted_test_size = min(adjusted_test_size, 1.0)

    # Train test split with normal samples + extremes
    normal_samples = samples.drop(extreme_samples.index)
    normal_train, normal_test = train_test_split(
        normal_samples, test_size=adjusted_test_size, random_state=random_state
    )

    train = pd.concat([normal_train, extreme_samples], ignore_index=True)
    train_df = pd.DataFrame(train, columns=samples.columns)
    train_df["train_test"] = "train"

    # Test set with normal samples only
    test_df = pd.DataFrame(normal_test, columns=samples.columns)
    test_df["train_test"] = "test"

    train_test_df = pd.concat([train_df, test_df])
    return train_test_df


if __name__ == "__main__":
    config = AppConfig()
    dataset_loc = config.data_path
    save_path = config.train_test_split_path

    cd = CompositionData()

    samples = get_all_samples(cd, dataset_loc)
    filtered_samples = filter_samples(samples, [])
    split = create_train_test_split_with_extremes(
        filtered_samples, n_extremes=2, test_size=0.2, random_state=42
    )

    split.to_csv(str(save_path), index=False)
