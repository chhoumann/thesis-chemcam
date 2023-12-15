import mlflow
import pandas as pd
from typing import Dict

from lib.reproduction import training_info, major_oxides, weighted_sum_oxide_percentages

client = mlflow.tracking.MlflowClient()

kv_pls_sm = {}

for oxide, v in training_info.items():
    sub = {}
    for comp_range_name, vv in v.items():
        model_name = f"PLS_ALL_{oxide}_{comp_range_name}"
        models = client.search_model_versions(f"name = '{model_name}'")
        model = mlflow.pyfunc.load_model(models[0].source)
        sub[comp_range_name] = model

    kv_pls_sm[oxide] = sub


kv_ica = {}
for oxide in major_oxides:
    model_name = f"ICA_{oxide}"
    models = client.search_model_versions(f"name = '{model_name}'")
    model = mlflow.pyfunc.load_model(models[0].source)
    kv_ica[oxide] = model


def get_ica_prediction(df: pd.DataFrame) -> Dict[str, float]:
    ica_pred = {}
    for oxide, model in kv_ica.items():
        ica_pred[oxide] = model.predict(df)

    return ica_pred


def get_pls_sm_prediction(df: pd.DataFrame) -> Dict[str, float]:
    pls_sm_pred = {}
    for oxide, v in kv_pls_sm.items():
        sub = {}
        for comp_range_name, model in v.items():
            sub[comp_range_name] = model.predict(df)

        pls_sm_pred[oxide] = sub

    return pls_sm_pred


def get_weighted_sum_prediction(df: pd.DataFrame) -> Dict[str, float]:
    oxide_preds = {}
    ica_pred = get_ica_prediction(df)
    pls_sm_pred = get_pls_sm_prediction(df)

    for oxide, distrib in weighted_sum_oxide_percentages.items():
        w_ica = distrib["ICA"]
        w_pls_sm = distrib["PLS1-SM"]

        oxide_preds[oxide] = ica_pred[oxide] * w_ica + pls_sm_pred[oxide] * w_pls_sm

    return oxide_preds
