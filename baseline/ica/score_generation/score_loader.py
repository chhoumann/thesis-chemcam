import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from ica.score_generation.postprocess import parallel_postprocess
from ica.score_generation.preprocess import preprocess
from ica.score_generation.run_ica import run_ica
from lib.config import AppConfig
from lib.data_handling import CompositionData
from lib.norms import Norm
from lib.utils import get_train_test_split

config = AppConfig()


def load_scores(is_test_run: bool):
    exclude_columns_abs = ["ID", "Sample Name"]

    ica_df_n1, compositions_df_n1 = _load_scores_for_norm(
        num_components=8, norm=Norm.NORM_1, is_test_run=is_test_run
    )
    temp_df = ica_df_n1.drop(columns=exclude_columns_abs)
    temp_df = temp_df.abs()
    ica_df_n1_abs = pd.concat([ica_df_n1[exclude_columns_abs], temp_df], axis=1)

    ica_df_n3, compositions_df_n3 = _load_scores_for_norm(
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


def _load_scores_for_norm(
    num_components: int, norm: Norm, is_test_run: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    calib_data_path = Path(config.data_path)
    output_dir = Path(
        f"{config.data_cache_dir}/_preformatted_ica/norm{norm.value}{'-test' if is_test_run else ''}"
    )

    ica_df_csv_loc = Path(f"{output_dir}/ica_data.csv")
    compositions_csv_loc = Path(f"{output_dir}/composition_data.csv")

    if ica_df_csv_loc.exists() and compositions_csv_loc.exists():
        print(f"Preprocessed ICA scores found for Norm {norm.value}. Loading data...")

        ica_df = pd.read_csv(ica_df_csv_loc)
        compositions_df = pd.read_csv(compositions_csv_loc)
    else:
        print(
            f"No preprocessed ICA scores found for Norm {norm.value}. Preprocessing data..."
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        ica_df, compositions_df = _compute_scores_for_norm(
            calib_data_path,
            num_components=num_components,
            norm=norm,
            is_test_run=is_test_run,
        )

        ica_df.to_csv(ica_df_csv_loc, index=False)
        compositions_df.to_csv(compositions_csv_loc, index=False)

        print(
            f"Preprocessed ICA scores saved to {ica_df_csv_loc} and {compositions_csv_loc}.\n"
        )

    return ica_df, compositions_df


def _compute_scores_for_norm(
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

    desired_dataset = "test" if is_test_run else "train"

    # Prepare samples for parallel processing
    sample_details_list = []

    for sample_name in tqdm(list(os.listdir(calib_data_path))):
        split_info_sample_row = test_train_split_idx[
            test_train_split_idx["sample_name"] == sample_name
        ]["train_test"]

        if split_info_sample_row.empty:
            print(
                f"No split info found for {sample_name}. Likely has missing data or is not used in calib2015."
            )
            continue

        if split_info_sample_row.values[0] != desired_dataset:
            continue

        compositions_df = load_composition_df_for_sample(sample_name, composition_data)

        if compositions_df is None:
            print(f"No composition data found for {sample_name}. Skipping.")
            continue

        dfs = preprocess(sample_name, calib_data_path, average_location_datasets, norm)

        for sample_id, df in dfs:
            ica_estimated_sources = run_ica(
                df, model=ica_model, num_components=num_components
            )

            sample_details_list.append(
                (
                    df,
                    compositions_df,
                    ica_estimated_sources,
                    sample_name,
                    sample_id,
                    num_components,
                )
            )

    # Post process the data in parallel
    print("Post processing preprocessed data...")

    with tqdm(total=len(sample_details_list)) as pbar:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(parallel_postprocess, detail)
                for detail in sample_details_list
            ]

            for future in as_completed(futures):
                ic_wavelengths, filtered_compositions_df = future.result()
                ic_wavelengths_list.append(ic_wavelengths)
                filtered_compositions_list.append(filtered_compositions_df)
                pbar.update(1)

    ica_df = _concatenate_preprocessed_dfs(ic_wavelengths_list)
    compositions_df = _concatenate_preprocessed_dfs(filtered_compositions_list)

    print(f"Finished processing {len(ica_df)} samples.")

    return ica_df, compositions_df


def _concatenate_preprocessed_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(dfs)
    df = df.apply(pd.to_numeric, errors="ignore")

    return df


def load_composition_df_for_sample(
    sample_name: str, composition_data: CompositionData
) -> Optional[pd.DataFrame]:
    # Check if we have composition data for this sample
    composition_df = composition_data.get_composition_for_sample(sample_name)

    if composition_df.empty:
        print(f"No composition data found for {sample_name}. Skipping...")
        return None

    # Check if the composition data contains NaN values
    if composition_df.isnull().values.any():
        print(f"NaN values found in composition data for {sample_name}. Skipping...")
        return None

    return composition_df
