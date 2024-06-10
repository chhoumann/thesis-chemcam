from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVR
from xgboost import XGBRegressor

from lib.norms import Norm3Scaler


def sio2():
    # Define the preprocessing pipelines with the specified parameters
    pls_preprocessor_pipeline = Pipeline(
        [
            ("scaler", MinMaxScaler(feature_range=(0, 1))),
            ("pca", KernelPCA(n_components=100, kernel="cosine", gamma=0.0025184866615475, degree=2)),
        ]
    )

    svr_preprocessor_pipeline = Pipeline([("scaler", MinMaxScaler(feature_range=(-1, 1)))])

    gbr_preprocessor_pipeline = Pipeline([("scaler", Norm3Scaler())])

    # Define the base estimators with the specified parameters
    pls_pipeline = Pipeline([("preprocessor", pls_preprocessor_pipeline), ("pls", PLSRegression(n_components=1))])

    svr_pipeline = Pipeline(
        [
            ("preprocessor", svr_preprocessor_pipeline),
            (
                "svr",
                SVR(
                    C=0.101270914859041,
                    kernel="poly",
                    degree=5,
                    gamma="auto",
                    coef0=5.982869617857073,
                    epsilon=0.1075038351628729,
                    max_iter=20000000,
                ),
            ),
        ]
    )

    gbr_pipeline = Pipeline(
        [
            ("preprocessor", gbr_preprocessor_pipeline),
            (
                "gbr",
                GradientBoostingRegressor(
                    learning_rate=0.0195372695066511,
                    subsample=0.6333895654431646,
                    max_depth=3,
                    max_features="sqrt",
                    n_estimators=932,
                ),
            ),
        ]
    )

    # Combine the pipelines into a final "sio2" pipeline
    return [("pls", pls_pipeline), ("svr", svr_pipeline), ("gbr", gbr_pipeline)]


def tio2():
    # Define the preprocessor pipelines
    tio2_svr_preprocessor_pipeline = Pipeline(
        [("scaler", Norm3Scaler()), ("transformer", PowerTransformer(method="yeo-johnson", standardize=True))]
    )

    tio2_gbr_preprocessor_pipeline = Pipeline(
        [("scaler", Norm3Scaler()), ("transformer", PowerTransformer(method="yeo-johnson", standardize=True))]
    )

    tio2_rf_preprocessor_pipeline = Pipeline(
        [
            ("scaler", Norm3Scaler()),
            (
                "transformer",
                QuantileTransformer(subsample=60746.0, n_quantiles=941, random_state=42, output_distribution="uniform"),
            ),
        ]
    )

    # Define the base estimators with the specified parameters
    tio2_svr_pipeline = Pipeline(
        [
            ("preprocessor", tio2_svr_preprocessor_pipeline),
            (
                "svr",
                SVR(
                    C=0.0092802848242038,
                    kernel="poly",
                    degree=3,
                    gamma="scale",
                    coef0=8.63601100525176,
                    epsilon=0.0028037787477313,
                    max_iter=20000000,
                ),
            ),
        ]
    )

    tio2_gbr_pipeline = Pipeline(
        [
            ("preprocessor", tio2_gbr_preprocessor_pipeline),
            (
                "gbr",
                GradientBoostingRegressor(
                    learning_rate=0.0285922209309325,
                    subsample=0.5585632955924456,
                    max_depth=5,
                    max_features="sqrt",
                    n_estimators=898,
                ),
            ),
        ]
    )

    tio2_rf_pipeline = Pipeline(
        [
            ("preprocessor", tio2_rf_preprocessor_pipeline),
            (
                "rf",
                RandomForestRegressor(
                    min_samples_split=2,
                    n_estimators=136,
                    max_depth=15,
                    min_samples_leaf=4,
                    max_features="sqrt",
                ),
            ),
        ]
    )

    # Combine the pipelines into a final "tio2" pipeline
    return [("svr", tio2_svr_pipeline), ("gbr", tio2_gbr_pipeline), ("rf", tio2_rf_pipeline)]


def al2o3():
    al2o3_xgboost_pipeline = Pipeline(
        [
            ("scaler", Norm3Scaler()),
            ("transformer", PowerTransformer(method="yeo-johnson", standardize=True)),
            (
                "xgboost",
                XGBRegressor(
                    learning_rate=0.0264099388873174,
                    reg_lambda=0.0034397452764153,
                    colsample_bytree=0.5558889734016561,
                    reg_alpha=0.5011206377632488,
                    n_estimators=761.0,
                    max_depth=5.0,
                    subsample=0.7374993842567144,
                    gamma=0.157189926185756,
                ),
            ),
        ]
    )

    al2o3_svr_pipeline = Pipeline(
        [
            ("scaler", MinMaxScaler(feature_range=(0, 1))),
            (
                "transformer",
                QuantileTransformer(
                    subsample=74354,
                    n_quantiles=472,
                    random_state=42,
                    output_distribution="uniform",
                ),
            ),
            (
                "svr",
                SVR(
                    C=0.0338128083620645,
                    kernel="linear",
                    degree=4,
                    gamma="auto",
                    coef0=1.6935801132480222,
                    epsilon=0.0342891570241155,
                    max_iter=20000000,
                ),
            ),
        ]
    )

    al2o3_ridge_pipeline = Pipeline(
        [
            ("scaler", Norm3Scaler()),
            (
                "transformer",
                QuantileTransformer(
                    subsample=15268,
                    n_quantiles=840,
                    random_state=42,
                    output_distribution="uniform",
                ),
            ),
            ("ridge", Ridge(alpha=95.3074800455775)),
        ]
    )

    # Combine the pipelines into a final "al2o3" pipeline
    return [("xgboost", al2o3_xgboost_pipeline), ("svr", al2o3_svr_pipeline), ("ridge", al2o3_ridge_pipeline)]


def feot():
    feot_svr_pipeline = Pipeline(
        [
            ("scaler", Norm3Scaler()),
            (
                "transformer",
                QuantileTransformer(
                    subsample=27549,
                    n_quantiles=665,
                    random_state=42,
                    output_distribution="uniform",
                ),
            ),
            (
                "svr",
                SVR(
                    C=16.477142954470164,
                    kernel="rbf",
                    degree=5,
                    gamma="scale",
                    coef0=6.252397271092422,
                    epsilon=0.0149272838875138,
                    max_iter=20000000,
                ),
            ),
        ]
    )

    feot_pls_pipeline = Pipeline(
        [
            (
                "scaler",
                StandardScaler(with_std=True, with_mean=True),
            ),
            (
                "transformer",
                PowerTransformer(method="yeo-johnson", standardize=False),
            ),
            ("pls", PLSRegression(n_components=30)),
        ]
    )

    feot_ridge_pipeline = Pipeline(
        [
            ("scaler", Norm3Scaler()),
            (
                "transformer",
                QuantileTransformer(
                    subsample=23173,
                    n_quantiles=444,
                    random_state=42,
                    output_distribution="uniform",
                ),
            ),
            ("ridge", Ridge(alpha=55.16248016653623)),
        ]
    )

    # Combine the pipelines into a final "feot" pipeline
    return [("svr", feot_svr_pipeline), ("pls", feot_pls_pipeline), ("ridge", feot_ridge_pipeline)]


def mgo():
    # Define the preprocessing pipelines with the specified parameters
    svr_preprocessor_pipeline = Pipeline(
        [
            ("scaler", RobustScaler(with_centering=False, quantile_range=(10.0, 90.0))),
            ("transformer", PowerTransformer(method="yeo-johnson", standardize=True)),
        ]
    )

    pls_preprocessor_pipeline = Pipeline(
        [
            ("scaler", Norm3Scaler()),
            (
                "pca",
                KernelPCA(
                    n_components=80,
                    gamma=0.0224292044706105,
                    kernel="rbf",
                    degree=2,
                ),
            ),
        ]
    )

    ridge_preprocessor_pipeline = Pipeline(
        [
            ("scaler", RobustScaler(with_centering=True, quantile_range=(35.0, 65.0))),
            ("transformer", PowerTransformer(method="yeo-johnson", standardize=False)),
        ]
    )

    # Define the base estimators with the specified parameters
    svr_pipeline = Pipeline(
        [
            ("preprocessor", svr_preprocessor_pipeline),
            (
                "svr",
                SVR(
                    C=0.0892694115469055,
                    kernel="poly",
                    degree=3,
                    gamma="auto",
                    coef0=9.35850494507051,
                    epsilon=0.0124423713270124,
                    max_iter=20000000,
                ),
            ),
        ]
    )

    pls_pipeline = Pipeline(
        [
            ("preprocessor", pls_preprocessor_pipeline),
            ("pls", PLSRegression(n_components=3)),
        ]
    )

    ridge_pipeline = Pipeline(
        [
            ("preprocessor", ridge_preprocessor_pipeline),
            ("ridge", Ridge(alpha=49.00255709869574)),
        ]
    )

    # Combine the pipelines into a final "mgo" pipeline
    return [("svr", svr_pipeline), ("pls", pls_pipeline), ("ridge", ridge_pipeline)]


def cao():
    cao_svr_pipeline = Pipeline(
        [
            ("scaler", MinMaxScaler(feature_range=(0, 1))),
            (
                "transformer",
                QuantileTransformer(
                    subsample=80879,
                    n_quantiles=692,
                    random_state=42,
                    output_distribution="uniform",
                ),
            ),
            (
                "svr",
                SVR(
                    C=0.0834973126025444,
                    kernel="linear",
                    degree=1,
                    gamma="auto",
                    coef0=3.035393902393877,
                    epsilon=0.0858748187215623,
                    max_iter=20000000,
                ),
            ),
        ]
    )

    cao_pls_pipeline = Pipeline(
        [
            ("scaler", MaxAbsScaler()),
            (
                "transformer",
                QuantileTransformer(
                    subsample=64610,
                    n_quantiles=714,
                    random_state=42,
                    output_distribution="uniform",
                ),
            ),
            ("pls", PLSRegression(n_components=22)),
        ]
    )

    cao_gbr_pipeline = Pipeline(
        [
            ("scaler", Norm3Scaler()),
            (
                "transformer",
                QuantileTransformer(
                    subsample=91846,
                    n_quantiles=395,
                    random_state=42,
                    output_distribution="uniform",
                ),
            ),
            (
                "gbr",
                GradientBoostingRegressor(
                    learning_rate=0.0170321215492168,
                    subsample=0.9991027968583126,
                    max_depth=3,
                    max_features="sqrt",
                    n_estimators=1000,
                ),
            ),
        ]
    )

    # Combine the pipelines into a final "cao" pipeline
    return [("svr", cao_svr_pipeline), ("pls", cao_pls_pipeline), ("gbr", cao_gbr_pipeline)]


def na2o():
    na2o_svr_pipeline = Pipeline(
        [
            ("scaler", Norm3Scaler()),
            (
                "transformer",
                PowerTransformer(method="yeo-johnson", standardize=True),
            ),
            (
                "svr",
                SVR(
                    C=0.0075575590330982,
                    kernel="poly",
                    degree=4,
                    gamma="scale",
                    coef0=7.578932896276198,
                    epsilon=0.0026366262560171,
                    max_iter=20000000,
                ),
            ),
        ]
    )

    na2o_pls_pipeline = Pipeline(
        [
            ("scaler", Norm3Scaler()),
            (
                "transformer",
                PowerTransformer(method="yeo-johnson", standardize=True),
            ),
            ("pls", PLSRegression(n_components=30)),
        ]
    )

    na2o_gbr_pipeline = Pipeline(
        [
            ("scaler", Norm3Scaler()),
            (
                "transformer",
                QuantileTransformer(
                    subsample=61550,
                    n_quantiles=875,
                    random_state=42,
                    output_distribution="normal",
                ),
            ),
            (
                "gbr",
                GradientBoostingRegressor(
                    learning_rate=0.0110063295655558,
                    subsample=0.9527088410971416,
                    max_depth=5,
                    max_features="sqrt",
                    n_estimators=957,
                ),
            ),
        ]
    )

    # Combine the pipelines into a final "na2o" pipeline
    return [("svr", na2o_svr_pipeline), ("pls", na2o_pls_pipeline), ("gbr", na2o_gbr_pipeline)]


def k2o():
    k2o_pls_pipeline = Pipeline(
        [
            ("scaler", Norm3Scaler()),
            ("pls", PLSRegression(n_components=30)),
        ]
    )

    k2o_gbr_pipeline = Pipeline(
        [
            ("scaler", MinMaxScaler(feature_range=(-1, 1))),
            (
                "transformer",
                QuantileTransformer(
                    subsample=51417,
                    n_quantiles=139,
                    random_state=42,
                    output_distribution="uniform",
                ),
            ),
            (
                "gbr",
                GradientBoostingRegressor(
                    learning_rate=0.0358163597255004,
                    subsample=0.6765783752228338,
                    max_depth=4,
                    max_features="sqrt",
                    n_estimators=712,
                ),
            ),
        ]
    )

    k2o_svr_pipeline = Pipeline(
        [
            ("scaler", Norm3Scaler()),
            (
                "transformer",
                QuantileTransformer(
                    subsample=52840,
                    n_quantiles=873,
                    random_state=42,
                    output_distribution="uniform",
                ),
            ),
            (
                "svr",
                SVR(
                    C=446.9244924838029,
                    kernel="rbf",
                    degree=1,
                    gamma="auto",
                    coef0=1.5472933504186566,
                    epsilon=0.0094011000249323,
                    max_iter=20000000,
                ),
            ),
        ]
    )

    # Combine the pipelines into a final "k2o" pipeline
    return [("pls", k2o_pls_pipeline), ("gbr", k2o_gbr_pipeline), ("svr", k2o_svr_pipeline)]


def make_oxide_pipelines():
    return {
        "SiO2": sio2(),
        "TiO2": tio2(),
        "Al2O3": al2o3(),
        "FeOT": feot(),
        "MgO": mgo(),
        "CaO": cao(),
        "Na2O": na2o(),
        "K2O": k2o(),
    }
