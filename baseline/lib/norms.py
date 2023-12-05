from typing import Dict, Tuple

from sklearn.base import BaseEstimator, TransformerMixin


class Norm1Scaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.total = None

    def fit(self, df):
        """
        Compute the total intensity across all shots to be used for normalization.
        """
        shot_columns = df.columns[df.columns.str.startswith("shot")]
        self.total = df[shot_columns].sum().sum()
        return self

    def transform(self, df):
        """
        Apply norm1 normalization to the DataFrame.
        """
        if self.total is None:
            raise ValueError("The fit method must be called before transform.")

        shot_columns = df.columns[df.columns.str.startswith("shot")]
        df[shot_columns] = df[shot_columns].div(self.total, axis=1)
        return df


class Norm3Scaler(BaseEstimator, TransformerMixin):
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
