from pathlib import Path

import mlflow
import pandas as pd

from lib.reproduction import major_oxides

# Supplementary runs
ngboost_experiments = {
    "SiO2": 167,
    "TiO2": 227,
    "Al2O3": 226,
    "FeOT": 168,
    "MgO": 261,
    "CaO": 171,
    "Na2O": 169,  # [169, 172] - did duplicate experiments
    "K2O": 170,  # [170, 173] - did duplicate experiments
}

random_forest_experiments = {
    "SiO2": 230,
    "TiO2": 245,
    "Al2O3": 244,
    "FeOT": 252,
    "MgO": 251,
    "CaO": 249,
    "Na2O": 250,
    "K2O": 253,
}

lasso_ridge_enet_experiments = {
    "SiO2": 178,
    "TiO2": 204,
    "Al2O3": 203,
    "FeOT": 222,
    "MgO": 205,
    "CaO": 221,
    "Na2O": 216,
    "K2O": 206,
}

# Main runs (larger scale, contains almost all model types)
main_runs = {
    "SiO2": "123",
    "TiO2": "125",
    "Al2O3": "131",
    "FeOT": "130",
    "MgO": "126",
    "CaO": "128",
    "Na2O": "127",
    "K2O": "129",
}


def _construct_experiment_df(experiment_dicts):
    for experiment_dict in experiment_dicts:
        for oxide in major_oxides:
            if oxide not in experiment_dict:
                raise ValueError(f"Missing {oxide} in one of the experiment dictionaries")

    experiment_data = {
        oxide: [experiment_dict[oxide] for experiment_dict in experiment_dicts] for oxide in major_oxides
    }

    return pd.DataFrame(experiment_data)


def _get_runs_across_oxides(df):
    oxide_runs = {}
    for oxide in df.columns:
        experiment_ids = df[oxide].dropna().tolist()
        runs = mlflow.search_runs(experiment_ids=experiment_ids)
        runs['params.oxide'] = oxide  # type: ignore
        oxide_runs[oxide] = runs
    return oxide_runs


def _get_full_runs_df(df, runs_file_path):
    if Path(runs_file_path).exists():
        return pd.read_csv(runs_file_path)

    oxide_runs = _get_runs_across_oxides(df)
    runs = pd.concat(oxide_runs.values(), ignore_index=True)
    runs.to_csv(runs_file_path)

    return runs


def get_full_runs_df(path_to_runs_file: str):
    df = _construct_experiment_df(
        [ngboost_experiments, random_forest_experiments, lasso_ridge_enet_experiments, main_runs]
    )

    return _get_full_runs_df(df, path_to_runs_file)


def clean_experiment_data(runs: pd.DataFrame):
    filtered_runs = runs[~runs["params.model_type"].isna()]  # Remove runs without model type (parent runs)
    filtered_runs = filtered_runs[filtered_runs["status"] != "FAILED"]  # Remove failed runs
    return filtered_runs

PARAMETER_MAPPINGS = {
    "gbr": ["gbr_n_estimators", "gbr_learning_rate", "gbr_max_depth", "gbr_subsample", "gbr_max_features"],
    "svr": ["svr_C", "svr_epsilon", "svr_kernel", "svr_degree", "svr_gamma", "svr_coef0", "max_iter"],
    "xgboost": ["xgb_n_estimators", "xgb_learning_rate", "xgb_max_depth", "xgb_subsample", "xgb_colsample_bytree", "xgb_gamma", "xgb_reg_alpha", "xgb_reg_lambda"],
    "extra_trees": ["et_n_estimators", "et_max_depth", "et_min_samples_split", "et_min_samples_leaf", "et_max_features"],
    "pls": ["pls_n_components"],
    "ngboost": ["Dist", "max_iter", "Score", "Base", "natural_gradient", "n_estimators", "learning_rate", "minibatch_frac", "col_sample", "tol", "random_state", "validation_fraction", "early_stopping_rounds"],
    "lasso": ["lasso_alpha"],
    "ridge": ["ridge_alpha"],
    "elasticnet": ["elasticnet_alpha", "elasticnet_l1_ratio"],
    "random_forest": ["rf_n_estimators", "rf_max_depth", "rf_min_samples_split", "rf_min_samples_leaf", "rf_max_features"],
    "robust_scaler": ["quantile_range", "with_centering"],
    "standard_scaler": ["with_mean", "with_std"],
    "min_max_scaler": ["feature_range"],
    "max_abs_scaler": [],
    "power_transformer": ["method", "standardize"],
    "quantile_transformer": ["n_quantiles", "output_distribution", "subsample", "random_state"],
    "norm3_scaler": [],
    "norm1_scaler": [],
    "pca": ["pca_n_components", "pca_whiten"],
    "kernel_pca": ["kernel_pca_n_components", "kernel_pca_kernel", "kernel_pca_gamma", "kernel_pca_degree"],
}

def pretty_print_params(row):
    model_type = row.get("params.model_type", "Unknown model")
    scaler_type = row.get("params.scaler_type", "Unknown scaler")
    transformer_type = row.get("params.transformer_type", "None")
    pca_type = row.get("params.pca_type", "None")

    model_params = {k.replace("params.", ""): v for k, v in row.items() if k.replace("params.", "") in PARAMETER_MAPPINGS.get(model_type, [])}
    scaler_params = {k.replace("params.", ""): v for k, v in row.items() if k.replace("params.", "") in PARAMETER_MAPPINGS.get(scaler_type, [])}
    transformer_params = {k.replace("params.", ""): v for k, v in row.items() if k.replace("params.", "") in PARAMETER_MAPPINGS.get(transformer_type, [])}
    pca_params = {k.replace("params.", ""): v for k, v in row.items() if k.replace("params.", "") in PARAMETER_MAPPINGS.get(pca_type, [])}

    print(f"Model: {model_type}")
    print("Model Parameters:")
    for param, value in model_params.items():
        print(f"  {param}: {value}")

    print(f"\nScaler: {scaler_type}")
    print("Scaler Parameters:")
    for param, value in scaler_params.items():
        print(f"  {param}: {value}")

    if transformer_type != "none":
        print(f"\nTransformer: {transformer_type}")
        print("Transformer Parameters:")
        for param, value in transformer_params.items():
            print(f"  {param}: {value}")

    if pca_type != "none":
        print(f"\nPCA: {pca_type}")
        print("PCA Parameters:")
        for param, value in pca_params.items():
            print(f"  {param}: {value}")
