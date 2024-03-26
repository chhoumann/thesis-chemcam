import os
from enum import Enum
from typing import Optional

from dotenv import find_dotenv, load_dotenv

_ENV_VARS = [
    "MLFLOW_TRACKING_URI",
    "DATA_SOURCE",
    "PDS_DATA_PATH",
    "PDS_COMPOSITION_DATA_PATH",
    "CCAM_DATA_PATH",
    "CCAM_COMPOSITION_DATA_PATH",
    "TRAIN_TEST_SPLIT_PATH",
    "DATA_CACHE_DIR",
]


class DataSource(Enum):
    PDS = "pds"
    CCAM = "ccam"


class AppConfig:
    def __init__(self, data_source: Optional[DataSource] = None):
        load_dotenv(find_dotenv())
        self._config = self._load_config(data_source)

    def _load_config(self, data_source: Optional[DataSource] = None):
        missing_vars = [var for var in _ENV_VARS if var not in os.environ]

        if missing_vars:
            raise ValueError(
                f"Missing environment variables: {', '.join(missing_vars)}"
            )

        vars = {var: os.environ[var] for var in _ENV_VARS}
        data_source_var = (
            vars["DATA_SOURCE"].lower() if data_source is None else data_source.value
        )

        if data_source_var == "pds":
            self._data_source = DataSource.PDS
            self._data_path = vars["PDS_DATA_PATH"]
            self._composition_data_path = vars["PDS_COMPOSITION_DATA_PATH"]
        elif data_source_var == "ccam":
            self._data_source = DataSource.CCAM
            self._data_path = vars["CCAM_DATA_PATH"]
            self._composition_data_path = vars["CCAM_COMPOSITION_DATA_PATH"]
        else:
            raise ValueError(
                f"Invalid data source: {self._data_source}. Must be 'pds' or 'ccam'."
            )

        return vars

    @property
    def mlflow_tracking_uri(self):
        return self._config["MLFLOW_TRACKING_URI"]

    @property
    def data_source(self):
        return self._data_source

    @property
    def data_path(self):
        return self._data_path

    @property
    def composition_data_path(self):
        return self._composition_data_path

    @property
    def train_test_split_path(self):
        return self._config["TRAIN_TEST_SPLIT_PATH"]

    @property
    def data_cache_dir(self):
        return self._config["DATA_CACHE_DIR"]
