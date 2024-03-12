from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from enum import Enum
from typing import Union

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
        Enumeration of predefined variance threshold levels ranging from 'LEAST' to 'MOST', allowing for easy specification of common thresholds.
        """
        LEAST = 1e-10
        VERY_LOW = 1e-9
        LOW = 5e-9
        MODERATE_LOW = 1e-8
        MODERATE = 5e-8
        MODERATE_HIGH = 1e-7
        HIGH = 5e-7
        VERY_HIGH = 1e-6
        MOST = 5e-6

    def __init__(self, threshold: Union[ThresholdLevel, float]):
        """
        Initializes the VarTrim transformer with a specified variance threshold.

        Parameters:
            threshold (Union[ThresholdLevel, float]): The variance threshold for feature selection, which can be specified either as a float or as a member of the ThresholdLevel enum.
        """
        self.features_to_keep_ = None
        self.threshold = threshold
        self.selector = None
        
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
        if isinstance(self.threshold, VarTrim.ThresholdLevel):
            threshold = self.threshold.value
        elif isinstance(self.threshold, float):
            threshold = self.threshold
        else:
            raise ValueError("threshold_spec must be either a ThresholdLevel or a float.")
        
        self.selector = VarianceThreshold(threshold=threshold)
        self.selector.fit(data)
        self.features_to_keep_ = data.columns[self.selector.get_support(indices=True)]
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
        
        return data[self.features_to_keep_]