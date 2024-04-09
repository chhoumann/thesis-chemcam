import os
from hashlib import md5

from dotenv import find_dotenv, load_dotenv

_ENV_VARS = [
    "MLFLOW_TRACKING_URI",
    "DATA_PATH",
    "COMPOSITION_DATA_PATH",
    "TRAIN_TEST_SPLIT_DIR",
    "DATA_CACHE_DIR",
    "CCAM_MASTER_LIST_FILE_NAME",
]


class AppConfig:
    def __init__(self):
        self._data_hash = None
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
        data_hash = self.data_hash
        dir = self._config["TRAIN_TEST_SPLIT_DIR"]

        return f"{dir}/{data_hash}.csv"

    @property
    def ccam_master_list_file_name(self):
        return self._config["CCAM_MASTER_LIST_FILE_NAME"]

    @property
    def ccam_master_list_file_path(self):
        return os.path.join(self.data_path, self.ccam_master_list_file_name)

    @property
    def data_hash(self):
        if self._data_hash is None:
            with open(self.composition_data_path, "rb") as f:
                file_contents = f.read()

            self._data_hash = md5(file_contents).hexdigest()

        return self._data_hash

    @property
    def data_cache_dir(self):
        return self._config["DATA_CACHE_DIR"]
