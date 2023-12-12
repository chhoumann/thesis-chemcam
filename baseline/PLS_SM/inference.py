from pandas import DataFrame

from lib.norms import Norm1Scaler, Norm3Scaler
from lib.reproduction import spectrometer_wavelength_ranges, training_info


def get_weights(y_full, blend_range_min, blend_range_max):
    """
    Helper function to calculate weights for blending predictions.
    """
    w_upper = (y_full - blend_range_min) / (blend_range_max - blend_range_min)
    w_lower = 1 - w_upper
    return w_lower, w_upper


def norm_data(x: DataFrame, oxide: str, model: str):
    """
    Normalizes the data for the given oxide and model.
    """
    norm = training_info[oxide][model]["normalization"]

    if norm == 1:
        scaler = Norm1Scaler(reshaped=True)
        print(f"Using Norm1Scaler for {oxide} {model}")
        return scaler.fit_transform(x.copy(deep=True))
    elif norm == 3:
        scaler = Norm3Scaler(spectrometer_wavelength_ranges, reshaped=True)
        print(f"Using Norm3Scaler for {oxide} {model}")
        return scaler.fit_transform(x.copy(deep=True))
    else:
        raise ValueError(f"Normalization value {norm} not recognized.")


def predict_composition_with_blending(oxide: str, X: DataFrame, models, ranges):
    """
    Predicts the composition of the given oxide based on the
    full model prediction (y_full) and the optimized blending ranges,
    including blending between Mid-High models as well as Low-Mid models.
    """
    X_full_norm = norm_data(X, oxide, "Full")
    y_full = models[oxide]["Full"].predict(X_full_norm)

    # Check for non-blending range predictions first
    blend_ranges = ["Low-Mid", "Mid-High"]

    for range_name, (range_min, range_max) in ranges[oxide].items():
        if range_min <= y_full <= range_max and range_name not in blend_ranges:
            X_range_norm = norm_data(X, oxide, range_name)
            return models[oxide][range_name].predict(X_range_norm)

    for blend_range in blend_ranges:
        # Check if blend_range is defined for the given oxide
        if blend_range in ranges[oxide]:
            blend_range_min, blend_range_max = ranges[oxide][blend_range]
            print(
                f"Range: {blend_range}, min: {blend_range_min}, max: {blend_range_max}"
            )

            # Check if y_full is within the defined blending range
            if blend_range_min <= y_full <= blend_range_max:
                w_lower, w_upper = get_weights(y_full, blend_range_min, blend_range_max)

                lower, upper = blend_range.split("-")
                X_lower_norm = norm_data(X, oxide, lower)
                X_upper_norm = norm_data(X, oxide, upper)

                y_lower = models[oxide][lower].predict(X_lower_norm)
                y_upper = models[oxide][upper].predict(X_upper_norm)

                y_final = w_lower * y_lower + w_upper * y_upper
                print(f"y_lower: {y_lower}, y_upper: {y_upper}, y_final: {y_final}")
                print(f"w_lower: {w_lower}, w_upper: {w_upper}")

                return y_final

    # Error if y_full is outside any defined range
    raise ValueError(
        f"y_full value {y_full} for oxide {oxide} is outside defined blending ranges."
    )
