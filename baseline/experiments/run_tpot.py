from math import sqrt
from pathlib import Path
from typing import Dict, Optional

import mlflow
import typer
from sklearn.metrics import mean_squared_error
from tpot import TPOTRegressor

from lib.experiment_setup import Experiment


def train_and_log_model(
    tpot_config: Dict,
    should_output_pipeline: bool = False,
    pipeline_output_dir: Optional[str] = None,
):
    tpot_experiment = Experiment(name="TPOT", norm=3)


    def run(X_train, X_test, y_train, y_test, oxide, _):
        tpot = TPOTRegressor(**tpot_config)  # type: ignore
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
        

    tpot_experiment.run_univariate(run)


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
    tpot_config = {
        "generations": int(generations),
        "population_size": int(population_size),
        "n_jobs": int(n_jobs),
        "verbosity": int(verbosity),
    }

    train_and_log_model(
        tpot_config,
        should_output_pipeline,
        pipeline_output_dir,
    )


if __name__ == "__main__":
    app()
