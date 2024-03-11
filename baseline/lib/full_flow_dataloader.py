import logging
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values

from lib.data_handling import CustomSpectralPipeline, load_split_data
from lib.norms import Norm1Scaler, Norm3Scaler
from lib.reproduction import major_oxides, masks


def load_full_flow_data():
    """
    Loads the data for the full flow.
    """
    logger = logging.getLogger("train")

    env = dotenv_values()
    comp_data_loc = env.get("COMPOSITION_DATA_PATH")
    dataset_loc = env.get("DATA_PATH")

    if not comp_data_loc:
        print("Please set COMPOSITION_DATA_PATH in .env file")
        exit(1)

    if not dataset_loc:
        print("Please set DATA_PATH in .env file")
        exit(1)

    preformatted_data_path = Path("./data/_preformatted_sm/")
    train_path = preformatted_data_path / "train.csv"
    test_path = preformatted_data_path / "test.csv"

    if (
        not preformatted_data_path.exists()
        or not train_path.exists()
        or not test_path.exists()
    ):
        logger.info("Loading data from location: %s", dataset_loc)
        train_data, test_data = load_split_data(
            str(dataset_loc), average_shots=True
        )
        logger.info("Data loaded successfully.")

        logger.info("Initializing CustomSpectralPipeline.")
        pipeline = CustomSpectralPipeline(
            masks=masks,
            composition_data_loc=comp_data_loc,
            major_oxides=major_oxides,
        )
        logger.info("Pipeline initialized. Fitting and transforming data.")
        train_processed = pipeline.fit_transform(train_data)
        test_processed = pipeline.fit_transform(test_data)
        logger.info("Data processing complete.")

        preformatted_data_path.mkdir(parents=True, exist_ok=True)

        train_processed.to_csv(train_path, index=False)
        test_processed.to_csv(test_path, index=False)
    else:
        logger.info(
            "Loading preformatted data from location: %s", preformatted_data_path
        )
        train_processed = pd.read_csv(train_path)
        test_processed = pd.read_csv(test_path)

    return train_processed, test_processed


def load_and_scale_data(norm: int):
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