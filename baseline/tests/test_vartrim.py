import numpy as np
import pandas as pd
import pytest

from lib.variance_threshold import VarTrim


@pytest.mark.parametrize(
    "threshold, expected_features",
    [
        # Threshold below the lowest variance keeps all features
        (
            1e-11,  # Slightly below the variance of the low variance feature
            [
                "feature_low_variance",
                "feature_medium_variance",
                "feature_high_variance",
            ],
        ),
        # Threshold that removes the lowest variance feature
        (
            1e-2,  # Above the variance of the low variance feature but below the medium
            ["feature_medium_variance", "feature_high_variance"],
        ),
        # Threshold that removes both low and medium variance features
        (
            0.5,  # Above the variance of both the low and medium variance features
            ["feature_high_variance"],
        ),
    ],
)
def test_vartrim_variance_threshold_with_adjusted_constants(
    threshold, expected_features
):
    np.random.seed(42)  # For reproducibility
    data = pd.DataFrame(
        {
            "feature_low_variance": np.random.normal(0, 1e-10, 100),
            "feature_medium_variance": np.random.normal(0, 0.25, 100),
            "feature_high_variance": np.random.normal(0, 1, 100),
        }
    )

    transformer = VarTrim(threshold=threshold)
    transformed_data = transformer.fit_transform(data)

    transformed_data = pd.DataFrame(
        transformed_data, columns=transformer.features_to_keep_
    )

    assert set(transformed_data.columns) == set(
        expected_features
    ), f"Unexpected features found in the transformed data for threshold {threshold}. Expected: {expected_features}, Found: {transformed_data.columns}"
