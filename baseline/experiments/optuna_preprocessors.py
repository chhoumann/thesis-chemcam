from optuna import Trial
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


def instantiate_pca(trial: Trial, logger=lambda params: None) -> PCA:
    params = {
        "n_components": trial.suggest_int("pca_n_components", 1, 50),
        "whiten": trial.suggest_categorical("pca_whiten", [True, False]),
    }
    logger(params)
    return PCA(**params)


def instantiate_kernel_pca(trial: Trial, logger=lambda params: None) -> KernelPCA:
    params = {
        "n_components": trial.suggest_int("kernel_pca_n_components", 1, 100),
        "kernel": trial.suggest_categorical("kernel_pca_kernel", ["linear", "poly", "rbf", "sigmoid", "cosine"]),
        "gamma": trial.suggest_float("kernel_pca_gamma", 1e-3, 1e1, log=True),
        "degree": trial.suggest_int("kernel_pca_degree", 1, 5),
    }
    logger(params)
    return KernelPCA(**params)


def instantiate_robust_scaler(trial: Trial, logger=lambda params: None) -> RobustScaler:
    quantile_range_choices = ["25-75", "10-90", "5-95", "35-65", "30-70", "40-60"]
    params = {
        "quantile_range": tuple(
            map(float, trial.suggest_categorical("quantile_range", quantile_range_choices).split("-"))
        ),
        "with_centering": trial.suggest_categorical("with_centering", [True, False]),
    }
    logger(params)
    return RobustScaler(**params)


def instantiate_standard_scaler(trial: Trial, logger=lambda params: None) -> StandardScaler:
    params = {
        "with_mean": trial.suggest_categorical("with_mean", [True, False]),
        "with_std": trial.suggest_categorical("with_std", [True, False]),
    }
    logger(params)
    return StandardScaler(**params)


def instantiate_min_max_scaler(trial: Trial, logger=lambda params: None) -> MinMaxScaler:
    feature_range_choices = ["0,1", "-1,1"]
    selected_range = trial.suggest_categorical("feature_range", feature_range_choices)
    feature_range_tuple = tuple(map(int, selected_range.split(",")))

    params = {
        "feature_range": feature_range_tuple,
    }
    logger(params)
    return MinMaxScaler(**params)


def instantiate_max_abs_scaler(trial: Trial, logger=lambda params: None) -> MaxAbsScaler:
    return MaxAbsScaler()


def instantiate_power_transformer(trial: Trial, logger=lambda params: None) -> PowerTransformer:
    params = {
        "method": "yeo-johnson",
        "standardize": trial.suggest_categorical("standardize", [True, False]),
    }
    logger(params)
    return PowerTransformer(**params)


def instantiate_quantile_transformer(trial: Trial, logger=lambda params: None) -> QuantileTransformer:
    params = {
        "n_quantiles": trial.suggest_int("quantile_transformer_n_quantiles", 100, 2000),
        "output_distribution": trial.suggest_categorical(
            "quantile_transformer_output_distribution", ["uniform", "normal"]
        ),
        "ignore_implicit_zeros": trial.suggest_categorical("quantile_transformer_ignore_implicit_zeros", [True, False]),
        "subsample": trial.suggest_int("quantile_transformer_subsample", 10000, 100000),
        "random_state": 42,
    }
    logger(params)
    return QuantileTransformer(**params)
