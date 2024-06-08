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
