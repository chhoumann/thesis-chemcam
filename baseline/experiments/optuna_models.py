from ngboost import NGBRegressor
from ngboost.distns import Exponential, LogNormal, Normal
from ngboost.scores import LogScore
from optuna import Trial
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


def instantiate_gbr(trial: Trial, logger=lambda params: None) -> GradientBoostingRegressor:
    params = {
        "n_estimators": trial.suggest_int("gbr_n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("gbr_learning_rate", 1e-3, 1e0, log=True),
        "max_depth": trial.suggest_int("gbr_max_depth", 3, 10),
        "subsample": trial.suggest_float("gbr_subsample", 0.5, 1.0),
        "max_features": trial.suggest_categorical("gbr_max_features", ["sqrt", "log2"]),
    }
    logger(params)
    return GradientBoostingRegressor(**params)


def instantiate_svr(trial: Trial, logger=lambda params: None) -> SVR:
    params = {
        "C": trial.suggest_float("svr_C", 1e-3, 1e3, log=True),
        "epsilon": trial.suggest_float("svr_epsilon", 1e-3, 1e1, log=True),
        "kernel": trial.suggest_categorical("svr_kernel", ["linear", "poly", "rbf", "sigmoid"]),
        "degree": trial.suggest_int("svr_degree", 1, 5),
        "gamma": trial.suggest_categorical("svr_gamma", ["scale", "auto"]),
        "coef0": trial.suggest_float("svr_coef0", 0, 10),
        "max_iter": 20_000_000,
    }
    logger(params)
    return SVR(**params)


def instantiate_xgboost(trial: Trial, logger=lambda params: None) -> XGBRegressor:
    params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 1e-3, 1e0, log=True),
        "max_depth": trial.suggest_int("xgb_max_depth", 2, 15),
        "subsample": trial.suggest_float("xgb_subsample", 0.3, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("xgb_gamma", 1e-3, 1e1, log=True),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-3, 1e3, log=True),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-3, 1e3, log=True),
    }
    logger(params)
    return XGBRegressor(**params)


def instantiate_extra_trees(trial: Trial, logger=lambda params: None) -> ExtraTreesRegressor:
    params = {
        "n_estimators": trial.suggest_int("et_n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("et_max_depth", 2, 15),
        "min_samples_split": trial.suggest_int("et_min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("et_min_samples_leaf", 1, 25),
        "max_features": trial.suggest_categorical("et_max_features", ["sqrt", "log2"]),
    }
    logger(params)
    return ExtraTreesRegressor(**params)


def instantiate_pls(trial: Trial, logger=lambda params: None) -> PLSRegression:
    params = {
        "n_components": trial.suggest_int("pls_n_components", 1, 30),
    }
    logger(params)
    return PLSRegression(**params)


def instantiate_ngboost(trial: Trial, logger=lambda params: None) -> NGBRegressor:
    distributions = {
        "Normal": (Normal, LogScore),
        "LogNormal": (LogNormal, LogScore),
        "Exponential": (Exponential, LogScore),
    }

    dist_name = trial.suggest_categorical("Dist", list(distributions.keys()))
    Dist, Score = distributions[dist_name]

    default_tree_learner = DecisionTreeRegressor(
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=trial.suggest_int("max_depth", 2, 10),
        splitter="best",
        random_state=None,
    )

    params = {
        "Dist": Dist,
        "Score": Score,
        "Base": default_tree_learner,
        "natural_gradient": trial.suggest_categorical("natural_gradient", [True, False]),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.5),
        "minibatch_frac": trial.suggest_uniform("minibatch_frac", 0.5, 1.0),
        "col_sample": trial.suggest_uniform("col_sample", 0.5, 1.0),
        "tol": trial.suggest_loguniform("tol", 1e-5, 1e-3),
        "random_state": 42,
        "validation_fraction": trial.suggest_uniform("validation_fraction", 0.1, 0.5),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 10, 100),
    }

    logger(params)
    return NGBRegressor(**params)


def instantiate_lasso(trial: Trial, logger=lambda params: None) -> Lasso:
    params = {
        "alpha": trial.suggest_float("lasso_alpha", 1e-3, 1e3, log=True),
    }
    logger(params)
    return Lasso(**params)


def instantiate_ridge(trial: Trial, logger=lambda params: None) -> Ridge:
    params = {
        "alpha": trial.suggest_float("ridge_alpha", 1e-3, 1e3, log=True),
    }
    logger(params)
    return Ridge(**params)


def instantiate_elasticnet(trial: Trial, logger=lambda params: None) -> ElasticNet:
    params = {
        "alpha": trial.suggest_float("elasticnet_alpha", 1e-3, 1e3, log=True),
        "l1_ratio": trial.suggest_float("elasticnet_l1_ratio", 0, 1),
    }
    logger(params)
    return ElasticNet(**params)
