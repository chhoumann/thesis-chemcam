from optuna import Trial
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)


def instantiate_robust_scaler(trial: Trial) -> RobustScaler:
    quantile_range_choices = ["25-75", "10-90", "5-95", "35-65", "30-70", "40-60"]
    params = {
        "quantile_range": tuple(
            map(float, trial.suggest_categorical("quantile_range", quantile_range_choices).split("-"))
        ),
        "with_centering": trial.suggest_categorical("with_centering", [True, False]),
    }

    return RobustScaler(**params)


def instantiate_standard_scaler(trial: Trial) -> StandardScaler:
    params = {
        "with_mean": trial.suggest_categorical("with_mean", [True, False]),
        "with_std": trial.suggest_categorical("with_std", [True, False]),
    }

    return StandardScaler(**params)


def instantiate_min_max_scaler(trial: Trial) -> MinMaxScaler:
    feature_range_choices = ["0,1", "-1,1"]
    selected_range = trial.suggest_categorical("feature_range", feature_range_choices)
    feature_range_tuple = tuple(map(int, selected_range.split(',')))

    params = {
        "feature_range": feature_range_tuple,
    }

    return MinMaxScaler(**params)


def instantiate_power_transformer(trial: Trial) -> PowerTransformer:
    params = {
        "method": "yeo-johnson",
        "standardize": trial.suggest_categorical("standardize", [True, False]),
    }

    return PowerTransformer(**params)
