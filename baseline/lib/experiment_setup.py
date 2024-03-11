import mlflow
import datetime
import pandas as pd

from sklearn.base import BaseEstimator
from typing import Callable
from lib.full_flow_dataloader import load_train_test_data


class Experiment:
    def __init__(self, name: str, norm: int):
        self.name = name
        self.norm = norm
        self.X_train, self.y_train, self.X_test, self.y_test = load_train_test_data(
            norm=norm
        )

    def run(
        self,
        func: Callable[
            [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, int], None
        ],
        mlflow_set_experiment=True,
    ):
        if mlflow_set_experiment:
            mlflow.set_experiment(
                f'{self.name}_Norm{self.norm}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
            )

        for target in self.y_train.columns:
            with mlflow.start_run(run_name=f"{self.name}_{target}"):
                mlflow.log_param("norm", self.norm)
                mlflow.log_param("target", target)
                func(
                    self.X_train,
                    self.X_test,
                    self.y_train,
                    self.y_test,
                    target,
                    self.norm,
                )

    def mlflow_sklearn_log_model(self, model: BaseEstimator, *args: str):
        name = f"{self.name}_{self.norm}{'_' + '_'.join(args) if args else ''}"
        mlflow.sklearn.log_model(model, name)
