import enum
from typing import Dict, Tuple

from sklearn.base import BaseEstimator, TransformerMixin


class Norm(enum.Enum):
    NORM_1 = 1
    NORM_3 = 3


class Norm1Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, reshaped=False):
        self.scaler = (
            Norm1ScalerReshapedData() if reshaped else Norm1ScalerOriginalData()
        )

    def fit(self, df):
        return self.scaler.fit(df)

    def transform(self, df):
        return self.scaler.transform(df)


class Norm1ScalerOriginalData(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.total_ = None
        self.shot_columns_ = None

    def fit(self, df):
        """
        Compute the total intensity across all shots to be used for normalization.
        """
        self.shot_columns_ = df.columns[df.columns.str.startswith("shot")]
        self.total_ = df[self.shot_columns_].sum().sum()
        assert self.total_ > 0, "Total intensity must be greater than zero."
        return self

    def transform(self, df):
        """
        Apply norm1 normalization to the DataFrame.
        """
        if self.total_ is None:
            raise ValueError("The fit method must be called before transform.")

        df[self.shot_columns_] = df[self.shot_columns_].div(self.total_, axis=1)
        return df


class Norm1ScalerReshapedData(BaseEstimator, TransformerMixin):
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
    def __init__(
        self, wavelength_ranges: Dict[str, Tuple[float, float]], reshaped=False
    ):
        self.scaler = (
            Norm3ScalerReshapedData(wavelength_ranges)
            if reshaped
            else Norm3ScalerOriginalData(wavelength_ranges)
        )

    def fit(self, df):
        return self.scaler.fit(df)

    def transform(self, df):
        return self.scaler.transform(df)


class Norm3ScalerOriginalData(BaseEstimator, TransformerMixin):
    def __init__(self, wavelength_ranges: Dict[str, Tuple[float, float]]):
        self.wavelength_ranges = wavelength_ranges
        self.totals = None

    def fit(self, df):
        """
        Compute the total intensity for each spectrometer range.
        """
        self.totals = {}
        shot_columns = df.columns[df.columns.str.startswith("shot")]
        for key, (start, end) in self.wavelength_ranges.items():
            mask = (df["wave"] >= start) & (df["wave"] <= end)
            self.totals[key] = df.loc[mask, shot_columns].sum().sum()
        return self

    def transform(self, df):
        """
        Apply norm3 normalization to the DataFrame.
        """
        if self.totals is None:
            raise ValueError("The fit method must be called before transform.")

        shot_columns = df.columns[df.columns.str.startswith("shot")]
        for key, (start, end) in self.wavelength_ranges.items():
            mask = (df["wave"] >= start) & (df["wave"] <= end)
            df.loc[mask, shot_columns] = df.loc[mask, shot_columns].div(
                self.totals[key], axis=1
            )
        return df


class Norm3ScalerReshapedData(BaseEstimator, TransformerMixin):
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

    def __init__(self, wavelength_ranges: Dict[str, Tuple[float, float]]):
        self.wavelength_ranges = wavelength_ranges
        self.totals = None

    def fit(self, df):
        """
        Compute the total intensity for each spectrometer range.
        """
        self.totals = {}
        for key, (start, end) in self.wavelength_ranges.items():
            # Select columns in the specified range and ignore non-float columns
            selected_columns = []
            for col in df.columns:
                try:
                    if start <= float(col) <= end:
                        selected_columns.append(col)
                except ValueError:
                    # Ignore columns that cannot be converted to float
                    continue

            # Compute the sum of intensities in these columns
            self.totals[key] = df[selected_columns].sum().sum()
        return self

    def transform(self, df):
        """
        Apply norm3 normalization to the DataFrame.
        """
        if self.totals is None:
            raise ValueError("The fit method must be called before transform.")

        for key, (start, end) in self.wavelength_ranges.items():
            # Select columns in the specified range and ignore non-float columns
            selected_columns = []
            for col in df.columns:
                try:
                    if start <= float(col) <= end:
                        selected_columns.append(col)
                except ValueError:
                    # Ignore columns that cannot be converted to float
                    continue

            # Normalize intensities in these columns
            df[selected_columns] = df[selected_columns].div(self.totals[key], axis=0)
        return df
