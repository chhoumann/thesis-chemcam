import enum

from sklearn.base import BaseEstimator, TransformerMixin


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
        no_channels = 2048

        # first no_channels columns are the wavelengths for the first spectrometer
        # second no_channels columns are the wavelengths for the second spectrometer
        # third no_channels columns are the wavelengths for the third spectrometer
        channel_1 = df.iloc[:, :no_channels]
        channel_2 = df.iloc[:, no_channels : no_channels * 2]
        channel_3 = df.iloc[:, no_channels * 2 : no_channels * 3]

        # sum the intensities for each channel
        channel_1_sum = channel_1.sum(axis=1)
        channel_2_sum = channel_2.sum(axis=1)
        channel_3_sum = channel_3.sum(axis=1)

        # divide each channel by its total intensity
        channel_1_normalized = channel_1.div(channel_1_sum, axis=0)
        channel_2_normalized = channel_2.div(channel_2_sum, axis=0)
        channel_3_normalized = channel_3.div(channel_3_sum, axis=0)

        # update the dataframe with the normalized values
        df.update(channel_1_normalized)
        df.update(channel_2_normalized)
        df.update(channel_3_normalized)

        return df
