import datetime
from typing import Callable

import mlflow
import pandas as pd
from sklearn.base import BaseEstimator

from lib.config import AppConfig
from lib.full_flow_dataloader import load_train_test_data


class Experiment:
    def __init__(self, name: str, norm: int, create_mlflow_experiment: bool = True):
        self.name = name
        self.norm = norm
        self.X_train, self.y_train, self.X_test, self.y_test = load_train_test_data(
            norm=norm
        )

        mlflow.set_tracking_uri(AppConfig().mlflow_tracking_uri)

        if create_mlflow_experiment:
            self.mlflow_experiment = mlflow.set_experiment(
                f'{self.name}_Norm{self.norm}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
            )

    def run_univariate(
        self,
        func: Callable[
            [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, int], None
        ],
    ):
        """
        Run the specified function for each target variable in a univariate manner.
        This runs iteratively for each target variable.

        Args:
            func (X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, target: str, norm: int): The function to be executed for each target variable.

        Returns:
            None
        """
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

    def run_multivariate(
        self,
        func: Callable[
            [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int], None
        ],
    ):
        """
        Run the specified function for each target variable in a multivariate manner.

        Args:
            func (X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, target: str, norm: int): The function to be executed for each target variable.

        Returns:
            None
        """
        with mlflow.start_run(run_name=self.name):
            mlflow.log_param("norm", self.norm)
            func(
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                self.norm,
            )

    def mlflow_sklearn_log_model(self, model: BaseEstimator, *args: str):
        name = f"{self.name}_{self.norm}{'_' + '_'.join(args) if args else ''}"
        mlflow.sklearn.log_model(model, name)
