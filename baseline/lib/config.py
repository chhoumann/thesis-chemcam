import os
from dotenv import load_dotenv, find_dotenv

EXPECTED_ENV_VARS = [
    "MLFLOW_TRACKING_URI",
    "DATA_PATH",
    "COMPOSITION_DATA_PATH",
    "TRAIN_TEST_SPLIT_PATH",
]

def load_config():
    load_dotenv(find_dotenv())
    for var in EXPECTED_ENV_VARS:
        if var not in os.environ:
            raise ValueError(f"{var} not set in .env")
        
    return {
        "mlflow_tracking_uri": os.environ["MLFLOW_TRACKING_URI"],
        "data_path": os.environ["DATA_PATH"],
        "composition_data_path": os.environ["COMPOSITION_DATA_PATH"],
        "train_test_split_path": os.environ["TRAIN_TEST_SPLIT_PATH"],
    }
