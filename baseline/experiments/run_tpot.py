import datetime
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import pandas as pd
import typer
from sklearn.metrics import mean_squared_error
from tpot import TPOTRegressor

from lib import full_flow_dataloader
from lib.config import load_config
from lib.norms import Norm1Scaler, Norm3Scaler
from lib.reproduction import major_oxides


def load_and_scale_data(norm: int):
    train_processed, test_processed = full_flow_dataloader.load_full_flow_data()

    train_cols = train_processed.columns
    test_cols = test_processed.columns

    scaler = Norm1Scaler() if norm == 1 else Norm3Scaler()
    train = scaler.fit_transform(train_processed)
    test = scaler.fit_transform(test_processed)

    # turn back into dataframe
    train = pd.DataFrame(train, columns=train_cols)
    test = pd.DataFrame(test, columns=test_cols)

    return train, test


def train_and_log_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    major_oxides: List[str],
    config: Dict,
    drop_cols: List[str],
    tpot_config: Dict,
    norm: int = 3,
    should_output_pipeline: bool = False,
    pipeline_output_dir: Optional[str] = None,
):
    X_train = train.drop(columns=drop_cols)
    y_train = train[major_oxides]
    X_test = test.drop(columns=drop_cols)
    y_test = test[major_oxides]

    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    experiment_name = (
        f'TPOT_Norm{norm}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )
    mlflow.set_experiment(experiment_name)

    tpot = TPOTRegressor(**tpot_config)  # type: ignore

    for oxide in major_oxides:
        with mlflow.start_run(run_name=f"TPOT_{oxide}"):
            # Log experiment configuration
            mlflow.log_params(tpot_config)
            mlflow.set_tags({"oxide": oxide, "model_type": "TPOT"})

            tpot.fit(X_train, y_train[oxide])

            score = tpot.score(X_test, y_test[oxide])
            y_pred = tpot.predict(X_test)
            rmse = sqrt(mean_squared_error(y_test[oxide], y_pred))

            # Log performance metrics
            mlflow.log_metrics({"score": score, "rmse": rmse})

            if should_output_pipeline:
                # Export and log pipeline structure
                pipeline_file = Path(f"{pipeline_output_dir}/tpot_{oxide}_pipeline.py")
                tpot.export(str(pipeline_file))
                mlflow.log_artifact(str(pipeline_file))


app = typer.Typer()


@app.command(add_help_option=True, help="Run TPOT")
def run(
    norm: int = typer.Option(default=3, help="Normalization type: 1 or 3"),
    should_output_pipeline: bool = typer.Option(
        default=False, help="Output the pipeline structure"
    ),
    pipeline_output_dir: Optional[str] = typer.Option(
        default=None,
        help="Output directory for the pipeline structure",
    ),
    generations: int = typer.Option(default=10, help="Number of generations"),
    population_size: int = typer.Option(default=50, help="Population size"),
    n_jobs: int = typer.Option(default=-1, help="Number of jobs"),
    verbosity: int = typer.Option(default=2, help="Verbosity level"),
):
    config = load_config()
    train, test = load_and_scale_data(norm)
    drop_cols = major_oxides + ["ID", "Sample Name"]

    tpot_config = {
        "generations": int(generations),
        "population_size": int(population_size),
        "n_jobs": int(n_jobs),
        "verbosity": int(verbosity),
    }

    train_and_log_model(
        train,
        test,
        major_oxides,
        config,
        drop_cols,
        tpot_config,
        norm,
        should_output_pipeline,
        pipeline_output_dir,
    )


if __name__ == "__main__":
    app()
