from pathlib import Path

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
    all_shots = pd.concat([df.set_index("wave") for df in data.values()], axis=0, keys=range(1, 6))
    all_shots_mean = all_shots.groupby("wave").mean()

    # Reset index to include 'wave' as a column in the final DataFrame
    final_avg_shots_df = all_shots_mean.reset_index()

    return final_avg_shots_df


class ICASampleProcessor:
    def __init__(self, sample_name: str, num_components: int):
        self.sample_name = sample_name
        self.sample_id = None
        self.num_components = num_components
        self.compositions_df = None
        self.df = None
        self.ic_wavelengths = None

    def try_load_composition_df(self, composition_data_loc: str) -> bool:
        # Check if we have composition data for this sample
        composition_data = CompositionData(composition_data_loc)
        composition_df = composition_data.get_composition_for_sample(self.sample_name)

        if composition_df.empty:
            print(f"No composition data found for {self.sample_name}. Skipping...")
            return False

        # Check if the composition data contains NaN values
        if composition_df.isnull().values.any():
            print(f"NaN values found in composition data for {self.sample_name}. Skipping...")
            return False

        self.composition_df = composition_df

        return True

    def preprocess(self, calib_data_path: Path, average_locations=False, norm: Norm = Norm.NORM_1) -> None:
        sample_data = get_preprocessed_sample_data(self.sample_name, calib_data_path, average_shots=False)
        location_name_ss, single_sample = list(sample_data.items())[0]

        self.sample_id = self.sample_name if average_locations else f"{self.sample_name}_{location_name_ss}"

        # Average all of the five location datasets into one single dataset
        final_avg_shots_df = (
            average_each_shot_across_locations(sample_data) if average_locations else single_sample
        )

        # Assuming `identify_outliers_with_mad_iterative_multidim` returns indices of non-outliers.
        non_outlier_indices, iterations = identify_outliers_with_mad_iterative_multidim(final_avg_shots_df.drop("wave", axis=1))

        # Create a full boolean array with False values
        outlier_mask = np.zeros(len(final_avg_shots_df), dtype=bool)

        # Set True for non-outliers
        outlier_mask[non_outlier_indices] = True

        # Invert the mask to get outliers
        outlier_mask = ~outlier_mask

        # Create a mask for columns to apply zeroing to (all columns except 'wave').
        columns_to_zero = final_avg_shots_df.columns != 'wave'

        # Set the outliers to 0
        final_avg_shots_df.loc[outlier_mask, columns_to_zero] = 0

        # Apply masking
        wmt = WavelengthMaskTransformer(masks)
        df = wmt.fit_transform(final_avg_shots_df)

        # set the wave column as the index
        final_avg_shots_df.set_index("wave", inplace=True)

        # Normalize the data
        scaler = Norm1Scaler() if norm.value == 1 else Norm3Scaler()
        final_avg_shots_df = pd.DataFrame(scaler.fit_transform(df))

        self.df = final_avg_shots_df.transpose()

    def postprocess(self, ica_estimated_sources: np.ndarray) -> None:
        columns = self.df.columns

        corrcols = [f"IC{i+1}" for i in range(self.num_components)]
        df_ics = pd.DataFrame(
            ica_estimated_sources,
            index=[f"shot{i+6}" for i in range(45)],
            columns=corrcols,
        )

        self.df = pd.concat([self.df, df_ics], axis=1)

        # Correlate the loadings
        corrdf, ids = self.__correlate_loadings__(corrcols, columns)

        # Create the wavelengths matrix for each component
        self.ic_wavelengths = pd.DataFrame(index=[self.sample_name], columns=columns)

        for i in range(len(ids)):
            ic = ids[i].split(" ")[0]
            component_idx = int(ic[2]) - 1
            wavelength = corrdf.index[i]
            corr = corrdf.iloc[i].iloc[component_idx]

            self.ic_wavelengths.loc[self.sample_name, wavelength] = corr

        # Filter the composition data to only include the oxides and their compositions
        self.composition_df = self.composition_df.iloc[:, 3:12]
        self.composition_df.index = [self.sample_name]

    # This is a function that finds the correlation between loadings and a set of columns
    # The idea is to somewhat automate identifying which element the loading corresponds to.
    def __correlate_loadings__(self, corrcols: list, icacols: list) -> (pd.DataFrame, list):
        corrdf = self.df.corr().drop(labels=icacols, axis=1).drop(labels=corrcols, axis=0)
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
