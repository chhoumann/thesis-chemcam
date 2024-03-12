import enum

from sklearn.base import BaseEstimator, TransformerMixin
from lib.reproduction import spectral_ranges
import pandas as pd

class Norm(enum.Enum):
    NORM_1 = 1
    NORM_3 = 3


class Norm1Scaler(BaseEstimator, TransformerMixin):
    """
    This class is used to normalize the data in the same way as the
    Norm1Scaler class, but it is used for the reshaped data. This is
    necessary because the reshaped data has a different format than
    the original data.

    The reshaped data has the following format:
    - Each row represents a single shot
    - Each column represents a single wavelength
    - The column names are the wavelengths
    """

    def __init__(self) -> None:
        pass

    def fit(self, df):
        return self

    def transform(self, df):
        """
        Apply norm1 normalization to the DataFrame.
        """
        wavelength_columns = []
        for col in df.columns:
            try:
                float(col)
                wavelength_columns.append(col)
            except ValueError:
                # Ignore columns that cannot be converted to float
                continue

        df_float = df[wavelength_columns]
        row_sums = df_float.sum(axis=1)
        normalized_df = df_float.div(row_sums, axis=0)
        df.update(normalized_df)

        return df


class Norm3Scaler(BaseEstimator, TransformerMixin):
    """
    This class is used to normalize the data in the same way as the
    Norm3Scaler class, but it is used for the reshaped data. This is
    necessary because the reshaped data has a different format than
    the original data.

    The reshaped data has the following format:
    - Each row represents a single shot
    - Each column represents a single wavelength
    - The column names are the wavelengths
    """

    def __init__(self):
        super().__init__()

    def fit(self, df):
        """
        Compute the total intensity for each spectrometer range.
        """
        return self


    def transform(self, df):
        """
        Apply norm3 normalization to the DataFrame.
        """
        spectrometer_start_indices = []
        columns = pd.to_numeric(df.columns, errors="coerce")
        columns = columns[~columns.isna()]

        for spectrometer in spectral_ranges:
            for i, col in enumerate(columns):
                if col >= spectral_ranges[spectrometer][0]:
                    spectrometer_start_indices.append(i)
                    break
                    
        for i in range(len(spectrometer_start_indices)):
            start = spectrometer_start_indices[i]

            if i == len(spectrometer_start_indices) - 1:
                end = len(columns)
            else:
                end = spectrometer_start_indices[i + 1]
                
            spectrometer_df = df.iloc[:, start:end]
            row_sums = spectrometer_df.sum(axis=1)
            normalized_df = spectrometer_df.div(row_sums, axis=0)
            df.update(normalized_df)

        return df