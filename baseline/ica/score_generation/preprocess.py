from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ica.score_generation.mad import identify_outliers_with_mad_iterative_multidim
from lib.data_handling import WavelengthMaskTransformer, get_preprocessed_sample_data
from lib.norms import Norm, Norm1Scaler, Norm3Scaler
from lib.reproduction import masks


def preprocess(
    sample_name: str,
    calib_data_path: Path,
    average_locations=False,
    norm: Norm = Norm.NORM_1,
) -> List[Tuple[str, pd.DataFrame]]:
    sample_data = get_preprocessed_sample_data(
        sample_name, calib_data_path, average_shots=False
    )

    dfs = []
    preprocessed_dfs = []

    if average_locations:
        dfs.append((sample_name, _average_each_shot_across_locations(sample_data)))
    else:
        for location_name, location_df in sample_data.items():
            dfs.append((location_name, location_df))

    for name, data in dfs:
        sample_id = sample_name if average_locations else f"{sample_name}_{name}"

        # Assuming `identify_outliers_with_mad_iterative_multidim` returns indices of non-outliers.
        non_outlier_indices, iterations = identify_outliers_with_mad_iterative_multidim(
            data.drop("wave", axis=1)
        )

        # Create a full boolean array with False values
        outlier_mask = np.zeros(len(data), dtype=bool)

        # Set True for non-outliers
        outlier_mask[non_outlier_indices] = True

        # Invert the mask to get outliers
        outlier_mask = ~outlier_mask

        # Create a mask for columns to apply zeroing to (all columns except 'wave').
        columns_to_zero = data.columns != "wave"

        # Set the outliers to 0
        data.loc[outlier_mask, columns_to_zero] = 0

        # Apply masking
        wmt = WavelengthMaskTransformer(masks)
        df = wmt.fit_transform(data)

        # set the wave column as the index
        data.set_index("wave", inplace=True)

        # Normalize the data
        scaler = Norm1Scaler() if norm.value == 1 else Norm3Scaler()
        data = pd.DataFrame(scaler.fit_transform(df))

        preprocessed_dfs.append((sample_id, data.transpose()))

    return preprocessed_dfs


def _average_each_shot_across_locations(data):
    # Concatenate all DataFrames along the 'wave' column to calculate the mean for each shot across locations
    all_shots = pd.concat(
        [df.set_index("wave") for df in data.values()], axis=0, keys=range(1, 6)
    )
    all_shots_mean = all_shots.groupby("wave").mean()

    # Reset index to include 'wave' as a column in the final DataFrame
    final_avg_shots_df = all_shots_mean.reset_index()

    return final_avg_shots_df
