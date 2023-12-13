import pandas as pd
import numpy as np
from lib.data_handling import CompositionData
from lib.data_handling import get_preprocessed_sample_data, WavelengthMaskTransformer
from lib.reproduction import masks
from lib.norms import Norm1Scaler, Norm3Scaler


class ICASampleProcessor:
    def __init__(self, sample_name: str, num_components: int):
        self.sample_name = sample_name
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


    def preprocess(self, calib_data_path: str, norm: int = 1) -> None:
        if norm != 1 and norm != 3:
            raise ValueError("Invalid Norm value. Must be 1 or 3.")

        sample_data = get_preprocessed_sample_data(
            self.sample_name, calib_data_path, average_shots=False
        )

        # For now, we just use the first of the datasets
        df = sample_data[0]

        # Apply masking
        wmt = WavelengthMaskTransformer(masks)
        df = wmt.fit_transform(df)

        # set the wave column as the index
        df.set_index("wave", inplace=True)

        # Normalize the data
        scaler = Norm1Scaler() if norm == 1 else Norm3Scaler()
        df = pd.DataFrame(scaler.fit_transform(df))

        self.df = df.transpose()


    def postprocess(self, ica_estimated_sources: np.ndarray) -> None:
        columns = self.df.columns

        corrcols = [f'IC{i+1}' for i in range(self.num_components)]
        df_ics = pd.DataFrame(ica_estimated_sources, index=[f'shot{i+6}' for i in range(45)], columns=corrcols)

        self.df = pd.concat([self.df, df_ics], axis=1)

        # Correlate the loadings
        corrdf, ids = self.__correlate_loadings__(corrcols, columns)

        # Create the wavelengths matrix for each component
        self.ic_wavelengths = pd.DataFrame(index=[self.sample_name], columns=columns)

        for i in range(len(ids)):
            ic = ids[i].split(' ')[0]
            component_idx = int(ic[2]) - 1
            wavelength = corrdf.index[i]
            corr = corrdf.iloc[i][component_idx]

            self.ic_wavelengths.loc[self.sample_name, wavelength] = corr

        # Filter the composition data to only include the oxides and their compositions
        self.composition_df = self.composition_df.iloc[:, 3:12]
        self.composition_df.index = [self.sample_name]


    # This is a function that finds the correlation between loadings and a set of columns
    # The idea is to somewhat automate identifying which element the loading corresponds to.
    def __correlate_loadings__(self, corrcols: list, icacols: list) -> (pd.DataFrame, list):
        corrdf = self.df.corr().drop(labels=icacols, axis=1).drop(labels=corrcols, axis=0)
        ids = []

        for ic_label in icacols:
            tmp = corrdf.loc[ic_label]
            match = tmp.values == np.max(tmp)
            col = corrcols[np.where(match)[0][-1]]

            ids.append(col + ' (r=' + str(np.max(tmp)) + ')')

        return corrdf, ids
