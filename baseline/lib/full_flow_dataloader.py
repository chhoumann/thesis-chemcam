import logging
from pathlib import Path

import pandas as pd

from lib.config import AppConfig
from lib.data_handling import CustomSpectralPipeline, load_split_data
from lib.norms import Norm1Scaler, Norm3Scaler
from lib.reproduction import major_oxides, masks


def load_full_flow_data(load_cache_if_exits: bool = True, average_shots: bool = True):
    """
    Loads the data for the full flow.
    """
    logger = logging.getLogger("train")

    config = AppConfig()
    composition_data_loc = config.composition_data_path
    dataset_loc = config.data_path

    preformatted_data_path = Path(f"{config.data_cache_dir}/_preformatted_sm/")

    data_hash = config.data_hash
    train_path = preformatted_data_path / f"train_{data_hash}.csv"
    test_path = preformatted_data_path / f"test_{data_hash}.csv"

    if (
        load_cache_if_exits
        and preformatted_data_path.exists()
        and train_path.exists()
        and test_path.exists()
    ):
        logger.info(
            "Loading preformatted data from location: %s", preformatted_data_path
        )
        train_processed = pd.read_csv(train_path)
        test_processed = pd.read_csv(test_path)
    else:
        logger.info("Loading data from location: %s", dataset_loc)
        train_data, test_data = load_split_data(
            str(dataset_loc), average_shots=average_shots
        )
        logger.info("Data loaded successfully.")

        logger.info("Initializing CustomSpectralPipeline.")
        pipeline = CustomSpectralPipeline(
            masks=masks,
            composition_data_loc=composition_data_loc,
            major_oxides=major_oxides,
        )
        logger.info("Pipeline initialized. Fitting and transforming data.")
        train_processed = pipeline.fit_transform(train_data)
        test_processed = pipeline.fit_transform(test_data)
        logger.info("Data processing complete.")

        preformatted_data_path.mkdir(parents=True, exist_ok=True)

        train_processed.to_csv(train_path, index=False)
        test_processed.to_csv(test_path, index=False)

    return train_processed, test_processed


def load_and_scale_data(norm: int):
    """
    Loads the data and scales it using the specified normalization method.
    """
    train_processed, test_processed = load_full_flow_data()

    train_cols = train_processed.columns
    test_cols = test_processed.columns

    scaler = Norm1Scaler() if norm == 1 else Norm3Scaler()
    train = scaler.fit_transform(train_processed)
    test = scaler.fit_transform(test_processed)

    # turn back into dataframe
    train = pd.DataFrame(train, columns=train_cols)
    test = pd.DataFrame(test, columns=test_cols)

    return train, test


def load_train_test_data(norm: int, drop_cols: list = ["ID", "Sample Name"]):
    """
    Loads the train and test data and returns the X and y values.
    """
    train, test = load_and_scale_data(norm)

    # Converting train set
    X_train = train.drop(columns=drop_cols)
    y_train = train[major_oxides]

    # Converting test set
    X_test = test.drop(columns=drop_cols)
    y_test = test[major_oxides]

    return X_train, y_train, X_test, y_test
