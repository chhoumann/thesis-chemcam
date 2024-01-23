import numpy as np

from lib.norms import Norm1Scaler, Norm3Scaler
from lib.reproduction import spectrometer_wavelength_ranges, training_info


def get_weights(y_full, blend_range_min, blend_range_max):
    """
    Helper function to calculate weights for blending predictions.
    """
    w_upper = (y_full - blend_range_min) / (blend_range_max - blend_range_min)
    w_lower = 1 - w_upper
    return w_lower, w_upper


def norm_data(x, oxide: str, model: str):
    """
    Normalizes the data for the given oxide and model.
    """
    norm = training_info[oxide][model]["normalization"]

    if norm == 1:
        scaler = Norm1Scaler(reshaped=True)
        print(f"Using Norm1Scaler for {oxide} {model}")
        scaled_df = scaler.fit_transform(x.copy(deep=True))
        assert np.isclose(scaled_df.sum(axis=1), 1).all()
        return scaled_df
    elif norm == 3:
        scaler = Norm3Scaler(spectrometer_wavelength_ranges, reshaped=True)
        print(f"Using Norm3Scaler for {oxide} {model}")

        scaled_df = scaler.fit_transform(x.copy(deep=True))
        assert np.isclose(
            scaled_df.sum(axis=1), 3, atol=1e-1
        ).all(), f"Norm3: {scaled_df.sum(axis=1)}"
        return scaled_df
    else:
        raise ValueError(f"Normalization value {norm} not recognized.")


def predict_composition_with_blending(oxide: str, X1, X3, models, ranges):
    """
    Predicts the composition of the given oxide based on the
    full model prediction (y_full) and the optimized blending ranges,
    including blending between Mid-High models as well as Low-Mid models.
    """
    assert len(X1) == len(X3), "X1 and X3 must be the same length"

    predictions = []
    blend_ranges = ["Low-Mid", "Mid-High"]

    for i in range(len(X1)):
        full_norm = training_info[oxide]["Full"]["normalization"]
        X_full_norm_row = X1.iloc[i] if full_norm == 1 else X3.iloc[i]
        y_full = models[oxide]["Full"].predict([X_full_norm_row])[0]

        prediction_made = False

        # Check if y_full is within a single range
        for range_name, (range_min, range_max) in ranges[oxide].items():
            if range_min <= y_full <= range_max and range_name not in blend_ranges:
                range_norm = training_info[oxide][range_name]["normalization"]
                X_range_norm_row = X1.iloc[i] if range_norm == 1 else X3.iloc[i]
                predictions.append(
                    models[oxide][range_name].predict([X_range_norm_row])[0]
                )
                # print(f"y_full: {y_full}, range: {range_name}_{oxide}")
                prediction_made = True
                break

        if prediction_made:
            continue

        # Blend between Low-Mid and Mid-High models
        for blend_range in blend_ranges:
            if blend_range not in ranges[oxide]:
                continue

            blend_range_min, blend_range_max = ranges[oxide][blend_range]

            if blend_range_min <= y_full <= blend_range_max:
                w_lower, w_upper = get_weights(y_full, blend_range_min, blend_range_max)

                lower, upper = blend_range.split("-")

                # if the model has Mid-High but no mid, inference would fail otherwise (K2O and Na2O)
                if lower not in models[oxide] and lower == "Mid":
                    lower = "Low"

                assert (
                    lower in models[oxide] and upper in models[oxide]
                ), f"{lower} or {upper} not in models for {oxide}"

                X_lower_norm_row = (
                    X1.iloc[i]
                    if training_info[oxide][lower]["normalization"] == 1
                    else X3.iloc[i]
                )
                X_upper_norm_row = (
                    X1.iloc[i]
                    if training_info[oxide][upper]["normalization"] == 1
                    else X3.iloc[i]
                )

                y_lower = models[oxide][lower].predict([X_lower_norm_row])[0]
                y_upper = models[oxide][upper].predict([X_upper_norm_row])[0]

                y_final = w_lower * y_lower + w_upper * y_upper

                predictions.append(y_final)
                prediction_made = True
                break

        if not prediction_made:
            raise ValueError(
                f"{i}: y_full value {y_full} for oxide {oxide} is outside defined blending ranges."
            )
            # don't include this sample in the final predictions
            # predictions.append(np.nan)

    return predictions
