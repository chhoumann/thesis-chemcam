import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold


class VarTrim(BaseEstimator, TransformerMixin):
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

    def fit(self, data: pd.DataFrame):
        """
        Fits the transformer to the data by determining which features exceed the variance threshold.

        Parameters:
            data (pd.DataFrame): The input data on which the transformer will be fitted.

        Returns:
            self: Returns an instance of self.

        Raises:
            ValueError: If the threshold is not a ThresholdLevel or float.
        """

        float_columns = data.select_dtypes(
            include=["float64", "float32"]
        ).columns.tolist()

        non_float_columns = data.select_dtypes(
            exclude=["float64", "float32"]
        ).columns.tolist()

        data_float = data[float_columns]

        self.selector = VarianceThreshold(threshold=self.threshold)
        self.selector.fit(data_float)
        self.features_to_keep_ = data_float.columns[
            self.selector.get_support(indices=True)
        ]
        self.non_float_columns_ = non_float_columns

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
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

        all_columns = list(self.features_to_keep_) + list(self.non_float_columns_)

        return data[all_columns]
