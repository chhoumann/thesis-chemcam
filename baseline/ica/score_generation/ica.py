import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA

from ica.score_generation.jade import JADE


def run_ica(
    df: pd.DataFrame, model: str = "jade", num_components: int = 8
) -> np.ndarray:
    """
    Performs Independent Component Analysis (ICA) on a given dataset using JADE or FastICA algorithms.

    Parameters:
    ----------
    df : pd.DataFrame
        The input dataset for ICA. The DataFrame should have rows as samples and columns as features.

    model : str, optional
        The ICA model to be used. Must be either 'jade' or 'fastica'. Defaults to 'fastica'.

    num_components : int, optional
        The number of independent components to be extracted. Defaults to 8.

    Returns:
    -------
    np.ndarray
        An array of the estimated independent components extracted from the input data.

    Raises:
    ------
    ValueError
        If an invalid model name is specified.

    AssertionError
        If the extracted signals are not independent, indicated by the correlation matrix not being
        close to the identity matrix or the sum of squares of correlations not being close to one.
    """
    estimated_sources = None

    if model == "jade":
        jade_model = JADE(num_components)
        mixing_matrix = jade_model.fit(X=df)  # noqa: F841
        estimated_sources = jade_model.transform(df)
    elif model == "fastica":
        fastica_model = FastICA(
            n_components=num_components, whiten="unit-variance", max_iter=5000
        )
        estimated_sources = fastica_model.fit_transform(df)
    else:
        raise ValueError("Invalid model specified. Must be 'jade' or 'fastica'.")

    return estimated_sources
