from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from enum import Enum
from typing import Union
import numpy as np

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
    class ThresholdLevel(Enum):
        """
        Enumeration of predefined variance threshold levels, adjusted to match the dataset's variance range, with options from 'LEAST' to 'MOST'.
        """
        LEAST = 0.0  # Matches the 0th percentile
        VERY_LOW = 8e-16  # Slightly above the 20th percentile
        LOW = 2e-15  # Slightly above the 30th percentile
        MODERATE_LOW = 4e-15  # Slightly above the 40th percentile
        MODERATE = 7e-15  # Slightly above the 50th percentile
        MODERATE_HIGH = 1.5e-14  # Slightly above the 60th percentile
        HIGH = 3e-14  # Slightly above the 70th percentile
        VERY_HIGH = 6.5e-14  # Slightly above the 80th percentile
        MOST = 1.7e-13  # Slightly above the 90th percentile
    
    def __init__(self, threshold: Union[ThresholdLevel, float]):
        """
        Initializes the VarTrim transformer with a specified variance threshold.

        Parameters:
            threshold (Union[ThresholdLevel, float]): The variance threshold for feature selection, which can be specified either as a float or as a member of the ThresholdLevel enum.
        """
        self.features_to_keep_ = None
        self.threshold = threshold
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
        
        float_columns, non_float_columns = [], []
        for col in data.columns:
            try:
                float(col)
                float_columns.append(col)
            except ValueError:
                non_float_columns.append(col)

        data_float = data[float_columns]

        #variances = np.var(data_float, axis=0)
        #percentiles = [np.percentile(variances, p) for p in range(0, 101, 10)]
        #print(percentiles)
        if isinstance(self.threshold, VarTrim.ThresholdLevel):
            threshold = self.threshold.value
        elif isinstance(self.threshold, float):
            threshold = self.threshold
        else:
            raise ValueError("threshold must be either a ThresholdLevel or a float.")
        
        self.selector = VarianceThreshold(threshold=threshold)
        self.selector.fit(data_float)
        self.features_to_keep_ = data_float.columns[self.selector.get_support(indices=True)]
        self.non_float_columns_ = non_float_columns
        return self
    
    def transform(self, data: pd.DataFrame):
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