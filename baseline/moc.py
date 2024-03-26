import logging
from typing import Dict, Mapping, Optional, Tuple

import mlflow
import pandas as pd
import typer
from sklearn.metrics import mean_squared_error

from ica.main import full_run as run_ica
from ica.main import test_run as run_ica_test
from lib.config import AppConfig
from lib.data_handling import CompositionData
from lib.reproduction import major_oxides, weighted_sum_oxide_percentages
from PLS_SM.full_flow import full_run as run_pls
from PLS_SM.full_flow import test_run as run_pls_test

logger = logging.getLogger(__name__)

app = typer.Typer()


def run_ica_pls_for_predictions_on_full_data(
    pls_experiment_id: Optional[str] = None, ica_experiment_id: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Getting predictions for PLS-SM")

    # Run PLS-SM
    pls_tar_pred: pd.DataFrame = (
        run_pls() if pls_experiment_id is None else run_pls_test(pls_experiment_id)
    )

    # Run ICA
    logger.info("Getting predictions for ICA")
    ica_tar_pred: pd.DataFrame = (
        run_ica() if ica_experiment_id is None else run_ica_test(ica_experiment_id)
    )

    logger.info("Finished getting predictions for PLS-SM and ICA")

    return pls_tar_pred, ica_tar_pred


def align_predictions(pls_tar_pred, ica_tar_pred) -> pd.DataFrame:
    logger.info("Aligning predictions")

    return pd.merge(
        ica_tar_pred,
        pls_tar_pred,
        how="inner",
        on=["ID", "Sample Name"],
        suffixes=("_ICA", "_PLS_SM"),
    )


def make_moc_predictions(
    pls_predictions: pd.DataFrame,
    ica_predictions: pd.DataFrame,
    blend_ratios: Mapping[str, Mapping[str, float | int]],
) -> pd.DataFrame:
    logger.info("Making MOC predictions")
    merged_df = align_predictions(
        ica_tar_pred=ica_predictions, pls_tar_pred=pls_predictions
    )

    moc_predictions = pd.DataFrame()

    for oxide, ratio in blend_ratios.items():
        w_ica = ratio["ICA"] / 100
        w_pls_sm = ratio["PLS1-SM"] / 100
        moc_predictions[oxide] = (
            merged_df[oxide + "_ICA"] * w_ica + merged_df[oxide + "_PLS_SM"] * w_pls_sm
        )

    moc_predictions["ID"] = merged_df.index
    moc_predictions["Sample Name"] = merged_df["Sample Name"]

    return moc_predictions


def merge_predictions_with_actual(moc_predictions: pd.DataFrame) -> pd.DataFrame:
    logger.info("Merging predictions with actual data")
    cd = CompositionData()

    merged_data = pd.DataFrame()

    for index, row in moc_predictions.iterrows():
        actual_data = cd.get_composition_for_sample(row["Sample Name"])

        if not actual_data.empty:
            for oxide in major_oxides:
                merged_data.at[index, oxide + "_pred"] = row[oxide]
                merged_data.at[index, oxide + "_actual"] = actual_data[oxide].values[0]
            merged_data.at[index, "Sample Name"] = row["Sample Name"]

    return merged_data


def calculate_moc_rmses(moc_predictions_and_actual: pd.DataFrame) -> Dict[str, float]:
    logger.info("Calculating MOC RMSEs")
    moc_rmses = {}

    for oxide in major_oxides:
        y_actual = moc_predictions_and_actual[oxide + "_actual"]
        y_pred = moc_predictions_and_actual[oxide + "_pred"]
        rmse = mean_squared_error(y_actual, y_pred, squared=False)
        moc_rmses[oxide] = rmse

    return moc_rmses


@app.command(name="run")
def main(pls_id: Optional[str] = None, ica_id: Optional[str] = None):
    logger.info("Running MOC Pipeline")

    pls_tar_pred, ica_tar_pred = run_ica_pls_for_predictions_on_full_data(
        pls_experiment_id=pls_id, ica_experiment_id=ica_id
    )

    timestamp = pd.Timestamp.now().strftime("%m-%d-%y_%H%M%S")
    mlflow_experiment_id = mlflow.create_experiment(f"MOC_{timestamp}")
    mlflow.set_experiment(experiment_id=mlflow_experiment_id)

    with mlflow.start_run(run_name="MOC"):
        moc_predictions = make_moc_predictions(
            pls_predictions=pls_tar_pred,
            ica_predictions=ica_tar_pred,
            blend_ratios=weighted_sum_oxide_percentages,
        )

        merged = merge_predictions_with_actual(moc_predictions)

        rmses = calculate_moc_rmses(merged)
        mlflow.log_metrics(rmses)

    logger.info("MOC Pipeline finished")


if __name__ == "__main__":
    app()
