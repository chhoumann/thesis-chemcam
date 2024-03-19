import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tqdm
import typer
from sklearn.decomposition import FastICA
from ica.data_processing import load_composition_df_for_sample, preprocess, postprocess
from ica.jade import JADE
from ica.train_test import train, test
from lib.data_handling import CompositionData
from lib.config import AppConfig
from lib.norms import Norm
from lib.utils import get_train_test_split

app = typer.Typer()
config = AppConfig()


@app.command(name="run", help="Runs train & test for ICA")
def run():
    # Train
    ica_df_n1, ica_df_n3, compositions_df_n1, compositions_df_n3 = get_data(
        is_test_run=False
    )

    train_experiment = train(
        ica_df_n1, ica_df_n3, compositions_df_n1, compositions_df_n3
    )

    # Test
    ica_df_n1, ica_df_n3, compositions_df_n1, compositions_df_n3 = get_data(
        is_test_run=True
    )

    return test(
        ica_df_n1,
        ica_df_n3,
        compositions_df_n1,
        compositions_df_n3,
        train_experiment.experiment_id,
    )


def get_data(is_test_run: bool):
    exclude_columns_abs = ["ID", "Sample Name"]

    ica_df_n1, compositions_df_n1 = _get_data_for_norm(
        num_components=8, norm=Norm.NORM_1, is_test_run=is_test_run
    )
    temp_df = ica_df_n1.drop(columns=exclude_columns_abs)
    temp_df = temp_df.abs()
    ica_df_n1_abs = pd.concat([ica_df_n1[exclude_columns_abs], temp_df], axis=1)

    ica_df_n3, compositions_df_n3 = _get_data_for_norm(
        num_components=8, norm=Norm.NORM_3, is_test_run=is_test_run
    )
    temp_df = ica_df_n3.drop(columns=exclude_columns_abs)
    temp_df = temp_df.abs()
    ica_df_n3_abs = pd.concat([ica_df_n3[exclude_columns_abs], temp_df], axis=1)

    assert len(ica_df_n1_abs) == len(
        ica_df_n3_abs
    ), "The number of rows in the two DataFrames must be equal."

    assert (
        ica_df_n1_abs["ID"] == ica_df_n1_abs["ID"]
    ).all(), "The IDs of the two DataFrames must be aligned."

    return ica_df_n1_abs, ica_df_n3_abs, compositions_df_n1, compositions_df_n3


def _get_data_for_norm(
    num_components: int, norm: Norm, is_test_run: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    calib_data_path = Path(config.data_path)
    output_dir = Path(
        f"{config.data_cache_dir}/_preformatted_ica/norm{norm.value}{'-test' if is_test_run else ''}"
    )

    ica_df_csv_loc = Path(f"{output_dir}/ica_data.csv")
    compositions_csv_loc = Path(f"{output_dir}/composition_data.csv")

    if ica_df_csv_loc.exists() and compositions_csv_loc.exists():
        print("Preprocessed data found. Loading data...")
        ica_df = pd.read_csv(ica_df_csv_loc)
        compositions_df = pd.read_csv(compositions_csv_loc)
    else:
        print("No preprocessed data found. Creating and saving preprocessed data...")

        output_dir.mkdir(parents=True, exist_ok=True)
        ica_df, compositions_df = _create_processed_data(
            calib_data_path,
            num_components=num_components,
            norm=norm,
            is_test_run=is_test_run,
        )

        ica_df.to_csv(ica_df_csv_loc, index=False)
        compositions_df.to_csv(compositions_csv_loc, index=False)

        print(
            f"Preprocessed data saved to {ica_df_csv_loc} and {compositions_csv_loc}.\n"
        )

    return ica_df, compositions_df


def _create_processed_data(
    calib_data_path: Path,
    ica_model: str = "jade",
    num_components: int = 8,
    norm: Norm = Norm.NORM_3,
    is_test_run: bool = False,
    average_location_datasets: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    composition_data = CompositionData(config.composition_data_path)

    ic_wavelengths_list = []
    ica_df = pd.DataFrame()

    filtered_compositions_list = []
    compositions_df = pd.DataFrame()

    test_train_split_idx = get_train_test_split()

    not_in_set = []
    missing = []

    desired_dataset = "test" if is_test_run else "train"
    for sample_name in tqdm.tqdm(list(os.listdir(calib_data_path))):
        split_info_sample_row = test_train_split_idx[
            test_train_split_idx["sample_name"] == sample_name
        ]["train_test"]

        if split_info_sample_row.empty:
            print(
                f"""No split info found for {
                sample_name}. Likely has missing data or is not used in calib2015."""
            )
            missing.append(sample_name)
            continue

        isSampleNotInSet = split_info_sample_row.values[0] != desired_dataset

        if isSampleNotInSet:
            # print(f"Skipping {sample_name}... Not in {desired_dataset} set.")
            not_in_set.append(sample_name)
            continue

        compositions_df = load_composition_df_for_sample(sample_name, composition_data)

        if compositions_df is None:
            print(f"No composition data found for {sample_name}. Skipping.")
            missing.append(sample_name)
            continue

        dfs = preprocess(sample_name, calib_data_path, average_location_datasets, norm)

        for sample_id, df in dfs:
            # Run ICA for each location in sample and get the estimated sources
            ica_estimated_sources = run_ica(
                df, model=ica_model, num_components=num_components
            )

            # Postprocess the data
            ic_wavelengths, filtered_compositions_df = postprocess(
                df, compositions_df, ica_estimated_sources, sample_id, num_components
            )

            # Add the sample name and ID to the ICA DataFrame
            ic_wavelengths["Sample Name"] = sample_name
            ic_wavelengths["ID"] = sample_id

            ic_wavelengths_list.append(ic_wavelengths)
            filtered_compositions_list.append(filtered_compositions_df)

    ica_df = _merge_preprocessed_dfs(ic_wavelengths_list)
    compositions_df = _merge_preprocessed_dfs(filtered_compositions_list)

    return ica_df, compositions_df


def run_ica(
    df: pd.DataFrame, model: str = "jade", num_components: int = 8
) -> np.ndarray:
    """
    Performs Independent Component Analysis (ICA) on a given dataset using JADE or FastICA algorithms.

    Parameters:
    ----------
    df : pd.DataFrame
        The input dataset for ICA. The DataFrame should have rows as samples and columns as features.

    model : str, optional
        The ICA model to be used. Must be either 'jade' or 'fastica'. Defaults to 'fastica'.

    num_components : int, optional
        The number of independent components to be extracted. Defaults to 8.

    Returns:
    -------
    np.ndarray
        An array of the estimated independent components extracted from the input data.

    Raises:
    ------
    ValueError
        If an invalid model name is specified.

    AssertionError
        If the extracted signals are not independent, indicated by the correlation matrix not being
        close to the identity matrix or the sum of squares of correlations not being close to one.
    """
    estimated_sources = None

    if model == "jade":
        # cols = df.columns
        jade_model = JADE(num_components)
        mixing_matrix = jade_model.fit(X=df)  # noqa: F841
        estimated_sources = jade_model.transform(df)
    elif model == "fastica":
        fastica_model = FastICA(
            n_components=num_components, whiten="unit-variance", max_iter=5000
        )
        estimated_sources = fastica_model.fit_transform(df)
    else:
        raise ValueError("Invalid model specified. Must be 'jade' or 'fastica'.")

    return estimated_sources


def _merge_preprocessed_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(dfs)
    df = df.apply(pd.to_numeric, errors="ignore")

    return df


if __name__ == "__main__":
    app()
