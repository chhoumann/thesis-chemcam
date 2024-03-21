import pandas as pd
import typer

from ica.score_generation.score_loader import get_scores
from ica.train_test import test, train
from lib.config import AppConfig

app = typer.Typer()
config = AppConfig()


@app.command(name="full_run", help="Runs train & test for ICA")
def full_run() -> pd.DataFrame:
    ica_df_n1, ica_df_n3, compositions_df_n1, compositions_df_n3 = get_scores(
        is_test_run=False
    )

    train_experiment = train(
        ica_df_n1, ica_df_n3, compositions_df_n1, compositions_df_n3
    )

    target_predictions = test_run(train_experiment_id=train_experiment.experiment_id)

    return target_predictions


@app.command(name="test_run", help="Runs test for ICA")
def test_run(train_experiment_id: str) -> pd.DataFrame:
    ica_df_n1, ica_df_n3, compositions_df_n1, compositions_df_n3 = get_scores(
        is_test_run=True
    )

    target_predictions = test(
        ica_df_n1,
        ica_df_n3,
        compositions_df_n1,
        compositions_df_n3,
        train_experiment_id,
    )

    return target_predictions


if __name__ == "__main__":
    app()
