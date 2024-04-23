from optuna import Trial
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


def instantiate_gbr(trial: Trial) -> GradientBoostingRegressor:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e0, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }

    return GradientBoostingRegressor(**params)


def instantiate_svr(trial: Trial) -> SVR:
    params = {
        "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-3, 1e1, log=True),
        "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
        "degree": trial.suggest_int("degree", 1, 5),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }

    return SVR(**params)


def instantiate_xgboost(trial: Trial) -> XGBRegressor:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e0, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-3, 1e1, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1e3, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1e3, log=True),
    }

    return XGBRegressor(**params)


def instantiate_extra_trees(trial: Trial) -> ExtraTreesRegressor:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 25),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }

    return ExtraTreesRegressor(**params)


def instantiate_pls(trial: Trial) -> PLSRegression:
    params = {
        "n_components": trial.suggest_int("n_components", 1, 30),
    }

    return PLSRegression(**params)

