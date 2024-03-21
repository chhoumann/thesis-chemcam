import enum
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from lib.reproduction import spectral_ranges
from lib.utils import get_numeric_col_names
from lib.variance_threshold import VarianceThresholdTrimmer


class Norm(enum.Enum):
    NORM_1 = 1
    NORM_3 = 3


def norm(df: pd.DataFrame) -> pd.DataFrame:
    return df.div(df.sum(axis=1), axis=0)


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
        Apply Norm 1 normalization to the DataFrame.
        """
        wavelength_cols = get_numeric_col_names(df).astype(str)
        df.update(norm(df[wavelength_cols]))

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

    def transform(self, df, ranges: Optional[List[Tuple[float, float]]] = None):
        """
        Apply Norm 3 normalization to the DataFrame.
        """
        if ranges is None:
            ranges = spectral_ranges.values()  # type: ignore

        wavelength_cols = get_numeric_col_names(df)

        for i, (start, end) in enumerate(ranges):
            cols_in_range = wavelength_cols[
                (wavelength_cols >= start) & (wavelength_cols <= end)
            ].astype(str)

            df.update(norm(df[cols_in_range]))

        return df
