import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold

from lib.utils import get_numeric_col_names


class VarianceThresholdTrimmer(BaseEstimator, TransformerMixin):
    """
    A transformer that reduces the dimensionality of data by removing features with low variance, based on a specified threshold.

    Attributes:
        features_to_keep_ (Index): The features selected to be kept after fitting the model.
        threshold (Union[ThresholdLevel, float]): The variance threshold below which features will be removed.
        selector (VarianceThreshold): The variance threshold selector used to fit the data.

    Parameters:
        threshold: Can be a predefined level from the ThresholdLevel enum or a custom float value. This threshold determines the features to be kept based on their variance.

    Methods:
        fit: Fits the model to the data, determining which features to keep.
        transform: Transforms the data by keeping only the selected features.
    """

    def __init__(self, threshold: float):
        """
        Initializes the VarTrim transformer with a specified variance threshold.

        Parameters:
            threshold (float): The variance threshold below which features will be removed.
        """
        self.threshold = threshold
        self.features_to_keep_ = None
        self.selector = None
        self.non_float_columns_ = None

    def fit(self, df: pd.DataFrame):
        """
        Fits the transformer to the data by determining which features exceed the variance threshold.

        Parameters:
            data (pd.DataFrame): The input data on which the transformer will be fitted.

        Returns:
            self: Returns an instance of self.

        Raises:
            ValueError: If the threshold is not a ThresholdLevel or float.
        """

        numeric_col_names = get_numeric_col_names(df).astype(str)

        numeric_columns = df[numeric_col_names].columns.tolist()
        non_numeric_columns = df.drop(columns=numeric_col_names).columns.tolist()

        numeric_df = df[numeric_columns]

        self.selector = VarianceThreshold(threshold=self.threshold)
        self.selector.fit(numeric_df)
        self.features_to_keep_ = numeric_df.columns[
            self.selector.get_support(indices=True)
        ]
        self.non_float_columns_ = non_numeric_columns

        return self

    def transform(self, data: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
        """
        Transforms the data by keeping only the features that meet the variance threshold.

        Parameters:
            data (pd.DataFrame): The input data to be transformed.

        Returns:
            pd.DataFrame: The transformed data with only the features that meet the variance threshold.

        Raises:
            ValueError: If the input data is not a DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a DataFrame.")

        if self.features_to_keep_ is None or self.non_float_columns_ is None:
            raise ValueError(
                "The transformer has not been fitted. Please fit the transformer before transforming the data."
            )

        all_columns = list(self.features_to_keep_) + list(self.non_float_columns_)

        return data[all_columns]
