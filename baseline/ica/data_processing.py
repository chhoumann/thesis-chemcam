from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from ica.mad import identify_outliers_with_mad_iterative_multidim

from lib.data_handling import (
    CompositionData,
    WavelengthMaskTransformer,
    get_preprocessed_sample_data,
)
from lib.norms import Norm, Norm1Scaler, Norm3Scaler
from lib.reproduction import masks


def average_each_shot_across_locations(data):
    # Concatenate all DataFrames along the 'wave' column to calculate the mean for each shot across locations
    all_shots = pd.concat(
        [df.set_index("wave") for df in data.values()], axis=0, keys=range(1, 6)
    )
    all_shots_mean = all_shots.groupby("wave").mean()

    # Reset index to include 'wave' as a column in the final DataFrame
    final_avg_shots_df = all_shots_mean.reset_index()

    return final_avg_shots_df


class ICASampleProcessor:
    def __init__(self, sample_name: str, num_components: int):
        self.sample_name = sample_name
        self.num_components = num_components
        self.compositions_df = None
        self.dfs = []

    def try_load_composition_df(self, composition_data_loc: str) -> bool:
        # Check if we have composition data for this sample
        composition_data = CompositionData(composition_data_loc)
        composition_df = composition_data.get_composition_for_sample(self.sample_name)

        if composition_df.empty:
            print(f"No composition data found for {self.sample_name}. Skipping...")
            return False

        # Check if the composition data contains NaN values
        if composition_df.isnull().values.any():
            print(
                f"NaN values found in composition data for {self.sample_name}. Skipping..."
            )
            return False

        self.composition_df = composition_df

        return True

    def preprocess(
        self, calib_data_path: Path, average_locations=False, norm: Norm = Norm.NORM_1
    ) -> None:
        sample_data = get_preprocessed_sample_data(
            self.sample_name, calib_data_path, average_shots=False
        )

        dfs = []

        if average_locations:
            dfs.append(
                (self.sample_name, average_each_shot_across_locations(sample_data))
            )
        else:
            for location_name, location_df in sample_data.items():
                dfs.append((location_name, location_df))

        for name, data in dfs:
            sample_id = (
                self.sample_name if average_locations else f"{self.sample_name}_{name}"
            )

            # Assuming `identify_outliers_with_mad_iterative_multidim` returns indices of non-outliers.
            non_outlier_indices, iterations = (
                identify_outliers_with_mad_iterative_multidim(data.drop("wave", axis=1))
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

            self.dfs.append((sample_id, data.transpose()))

    def postprocess(
        self, ica_estimated_sources: np.ndarray, df: pd.DataFrame, sample_id: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        columns = df.columns

        corrcols = [f"IC{i+1}" for i in range(self.num_components)]
        df_ics = pd.DataFrame(
            ica_estimated_sources,
            index=[f"shot{i+6}" for i in range(45)],
            columns=corrcols,
        )

        df = pd.concat([df, df_ics], axis=1)

        # Correlate the loadings
        corrdf, ids = self.__correlate_loadings__(corrcols, columns, df)

        # Create the wavelengths matrix for each component
        ic_wavelengths = pd.DataFrame(columns=columns)

        for i in range(len(ids)):
            ic = ids[i].split(" ")[0]
            component_idx = int(ic[2]) - 1
            wavelength = corrdf.index[i]
            corr = corrdf.iloc[i].iloc[component_idx]

            ic_wavelengths.loc[sample_id, wavelength] = corr

        # Filter the composition data to only include the oxides and their compositions
        filtered_composition_df = self.composition_df.iloc[:, 3:12]
        filtered_composition_df.index = pd.Index([sample_id])

        return ic_wavelengths, filtered_composition_df

    # This is a function that finds the correlation between loadings and a set of columns
    # The idea is to somewhat automate identifying which element the loading corresponds to.
    def __correlate_loadings__(
        self, corrcols: list, icacols: list, df: pd.DataFrame
    ) -> (pd.DataFrame, list):
        corrdf = df.corr().drop(labels=icacols, axis=1).drop(labels=corrcols, axis=0)
        # set all corrdf nans to 0 - they were set to 0 during masking, and
        # .corr() sets values that don't vary to NaN
        corrdf = corrdf.fillna(0)
        ids = []

        for ic_label in icacols:
            tmp = corrdf.loc[ic_label]
            match = tmp.values == np.max(tmp)
            col = corrcols[np.where(match)[0][-1]]

            ids.append(col + " (r=" + str(np.max(tmp)) + ")")

        return corrdf, ids
