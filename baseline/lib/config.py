import os

from dotenv import find_dotenv, load_dotenv

_ENV_VARS = [
    "MLFLOW_TRACKING_URI",
    "DATA_PATH",
    "COMPOSITION_DATA_PATH",
    "TRAIN_TEST_SPLIT_PATH",
    "CCAM_DATA_PATH",
    "CCAM_COMPOSITION_DATA_PATH",
]


class AppConfig:
    def __init__(self):
        load_dotenv(find_dotenv())
        self._config = self._load_config()

    def _load_config(self):
        missing_vars = [var for var in _ENV_VARS if var not in os.environ]
        if missing_vars:
            raise ValueError(
                f"Missing environment variables: {', '.join(missing_vars)}"
            )

        return {var: os.environ[var] for var in _ENV_VARS}

    @property
    def mlflow_tracking_uri(self):
        return self._config["MLFLOW_TRACKING_URI"]

    @property
    def data_path(self):
        return self._config["DATA_PATH"]

    @property
    def composition_data_path(self):
        return self._config["COMPOSITION_DATA_PATH"]

    @property
    def train_test_split_path(self):
        return self._config["TRAIN_TEST_SPLIT_PATH"]

    @property
    def ccam_data_path(self):
        return self._config["CCAM_DATA_PATH"]

    @property
    def ccam_composition_data_path(self):
        return self._config["CCAM_COMPOSITION_DATA_PATH"]
