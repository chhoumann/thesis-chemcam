import numpy as np
import pandas as pd

from lib.variance_threshold import VarTrim


def test_vartrim_variance_threshold():
    # Setup: Create a synthetic dataset
    np.random.seed(42)  # For reproducibility
    data = pd.DataFrame(
        {
            "feature_low_variance": np.random.normal(0, 1e-10, 100),
            "feature_medium_variance": np.random.normal(0, 1, 100),
            "feature_high_variance": np.random.normal(0, 10, 100),
        }
    )

    # Action: Apply VarTrim with a threshold that should remove only the low variance feature
    transformer = VarTrim(
        threshold=1e-9
    )  # This threshold is between the low and medium variance

    transformer.fit_transform(data)
    transformed_data = pd.DataFrame(transformer.fit_transform(data))

    # Assertion: Check that the correct features are kept
    expected_features = ["feature_medium_variance", "feature_high_variance"]

    assert all(
        [col in transformed_data.columns for col in expected_features]
    ), "VarTrim did not keep the expected features based on the variance threshold"

    assert (
        "feature_low_variance" not in transformed_data.columns
    ), "VarTrim kept a feature with variance lower than the threshold"

    # Optionally, check that the transformer attributes are set correctly
    assert (
        transformer.features_to_keep_ is not None
    ), "features_to_keep_ attribute was not set after fitting"

    assert (
        transformer.selector is not None
    ), "selector attribute was not set after fitting"
