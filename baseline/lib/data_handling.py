from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from lib.reproduction import ccs_drop_cols, folder_to_composition_sample_name
from lib.utils import get_train_test_split


def get_location_dataset_paths_for_sample(sample_name: str, data_path: Path):
    """Get the (five) location datasets for a sample."""
    sample_path = data_path / sample_name
    return [f for f in sample_path.iterdir() if f.is_file() and f.suffix == ".csv"]


def get_dataset_frame(dataset_path):
    """
    Read a dataset from a CSV file and return a pandas DataFrame.

    Parameters:
    dataset_path (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: The dataset as a DataFrame.
    """
    with open(dataset_path) as f:
        # find index of last line starting with "#" and skip rows until then
        target = 0
        for i, line in enumerate(f):
            if not line.startswith("#"):
                target = i
                break

        # read csv from that line - columns also start wih "#"
        return pd.read_csv(dataset_path, skiprows=target - 1)


def get_preprocessed_sample_data(sample_name: str, data_path: Path, average_shots=True) -> Dict[str, pd.DataFrame]:
    """
    Get preprocessed sample data.

    Args:
        sample_name (str): The name of the sample.
        data_path (Path): The path to the data.
        average_shots (bool, optional): Whether to average shots. Defaults to True.

    Returns:
        list[pd.DataFrame]: A list of preprocessed sample dataframes.
    """

    wavelengths = pd.Series(dtype="float64")

    sample_dataset_paths = get_location_dataset_paths_for_sample(sample_name, data_path)
    sample_spectra = {}

    for i, sample_set in enumerate(sample_dataset_paths):
        df = get_dataset_frame(sample_set)

        # strip whitespace from column names and remove # from column names
        df.columns = df.columns.str.strip().str.replace("# ", "")

        if i == 0:
            wavelengths = df["wave"]
        else:
            assert wavelengths.equals(df["wave"])

        df.drop(ccs_drop_cols, axis=1, inplace=True)

        # re-insert wavelengths to avoid averaging them
        df.insert(0, "wave", wavelengths)

        # add average of all shots and remove individual shots
        if average_shots:
            shot_cols = [col for col in df.columns if "shot" in col]
            shot_avg = df[shot_cols].mean(axis=1)
            df.drop(shot_cols, axis=1, inplace=True)
            df.insert(1, "shot_avg", shot_avg)

        sample_spectra[sample_set.stem] = df

    return sample_spectra


def load_data(
    dataset_loc: str,
    num_samples: Optional[int] = None,
    average_shots=True,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load data from the specified dataset location.

    Parameters:
    - dataset_loc (str): The location of the dataset.
    - num_samples (Optional[int]): The number of samples to load.
    If None, load all samples.

    Returns:
    - Dataset: The loaded dataset. A dictionary with keys as sample names and
    values as lists of pandas DataFrames, each item representing the data for
    a location on the sample.
    """

    # Function for loading sample data in parallel
    def _load_sample_data(sample_name, data_path, average_shots):
        return sample_name, get_preprocessed_sample_data(sample_name, data_path, average_shots)

    data_path = Path(dataset_loc).resolve(strict=True)
    sample_names = [f.name for f in data_path.iterdir() if f.is_dir()]

    take_amount = num_samples if num_samples else len(sample_names)
    sample_data = {}

    with tqdm(total=take_amount, desc="Loading data") as pbar:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(_load_sample_data, sample_name, data_path, average_shots)
                for sample_name in sample_names[:take_amount]
            ]

            for future in as_completed(futures):
                sample_name, data = future.result()
                sample_data[sample_name] = data
                pbar.update()

    return sample_data


def load_split_data(dataset_loc: str, split_loc: Optional[str] = None, average_shots=True):
    sample_data = load_data(dataset_loc, average_shots=average_shots)
    train_test_split_df = get_train_test_split(split_loc)

    train_samples = train_test_split_df.loc[train_test_split_df["train_test"] == "train"]["sample_name"].to_list()
    test_samples = train_test_split_df.loc[train_test_split_df["train_test"] == "test"]["sample_name"].to_list()

    train_sample_data = {sample_name: sample_data[sample_name] for sample_name in train_samples}
    test_sample_data = {sample_name: sample_data[sample_name] for sample_name in test_samples}

    return train_sample_data, test_sample_data


class WavelengthMaskTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to remove values in specified wavelength masks from input data.

    Parameters:
    -----------
    masks : list
        List of tuples representing the wavelength masks.
        Each tuple should contain two values:
        the lower and upper bounds of the mask.

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer to the data.

    transform(X)
        Transform the input data by removing values within the
        specified wavelength masks.

    """

    def __init__(self, masks):
        self.masks = masks

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Apply transformation to the input data.

        Parameters:
        X (pd.DataFrame): The input data to be transformed.

        Returns:
        pd.DataFrame: The transformed data.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input should be a pandas DataFrame.")

        is_pls = "shot_avg" in X.columns

        # if pls, work on shot_avg column
        # otherwise, on shot_6-50
        cols = ["shot_avg"] if is_pls else [f"shot{i}" for i in range(6, 51)]

        for mask in self.masks:
            mask_condition = (X["wave"] >= mask[0]) & (X["wave"] <= mask[1])
            for col in cols:
                X.loc[mask_condition, col] = 0

        return X


def transform_samples(
    sample_data: dict[str, list[pd.DataFrame]],
    transformer: WavelengthMaskTransformer,
) -> dict[str, list[pd.DataFrame]]:
    """
    Transform the input data by removing values within the specified
    wavelength masks.

    Parameters:
    -----------
    X_dict : dict
        Dictionary with keys as sample names and values as lists of pandas
        DataFrames, each item representing the data for a shot on the sample.

    Returns:
    --------
    transformed_data : dict
        Dictionary with keys as sample names and values as lists of
        pandas DataFrames.
    """
    if not isinstance(sample_data, dict):
        raise ValueError(
            "Input should be a dictionary with keys as sample names and" + " values as lists of pandas DataFrames."
        )

    transformed_data = {}
    for sample_name, dfs in sample_data.items():
        transformed_dfs = [transformer.transform(df) for df in dfs]
        transformed_data[sample_name] = transformed_dfs

    return transformed_data


class SpectralDataReshaper(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        wavelength_feature_name: str,
    ):
        self.wave_feature_name = wavelength_feature_name
        self.sample_size_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input should be a pandas DataFrame.")

        self.sample_size_ = len(X.columns) - 1
        return self

    def transform(self, X):
        if self.sample_size_ is None:
            raise RuntimeError("Transformer must be fitted before calling transform.")

        intensity_feature_names = [idx for idx in X.columns if idx != self.wave_feature_name]
        reshaped_values = X[intensity_feature_names].values.reshape(self.sample_size_, -1)
        transformed_df = pd.DataFrame(reshaped_values, columns=X[self.wave_feature_name].unique())

        return transformed_df


class CompositionData:
    """
    A class for handling composition data.

    Methods:
    - __init__():
        Initializes the CompositionData object with the composition data location from the AppConfig.

    - load_composition_data(composition_data_loc: str) -> pd.DataFrame:
        Loads the composition data from the specified file path.

    - get_composition_for_sample(sample_name) -> pd.DataFrame:
        Retrieves the composition data for a specific sample.

    - create_sample_compositions_dict(sample_names) -> dict[str, pd.DataFrame]:
        Creates a dictionary of sample compositions for the given sample names.
    """

    def __init__(
        self,
        composition_data_loc: str,
    ):
        self.composition_data_loc = composition_data_loc
        self.composition_data = self._load_composition_data()

    def get_composition_for_sample(self, sample_name) -> pd.DataFrame:
        _sample_name = folder_to_composition_sample_name.get(sample_name, sample_name)
        sample_name_lower = _sample_name.lower()

        match_condition = (
            self.composition_data[self.match_cols].apply(lambda x: x.str.lower() == sample_name_lower).any(axis=1)
        )

        composition = self.composition_data.loc[match_condition]

        return composition.head(1)

    def create_sample_compositions_dict(self, sample_names) -> dict[str, pd.DataFrame]:
        sample_compositions = {}

        for sample_name in sample_names:
            comp = self.get_composition_for_sample(sample_name)

            if comp.empty:
                print(f"Could not find {sample_name} in labels")
                continue

            sample_compositions[sample_name] = comp

        return sample_compositions

    def _load_composition_data(self) -> pd.DataFrame:
        first_row = pd.read_csv(self.composition_data_loc, nrows=1)
        first_column = first_row.columns[0]

        if first_column.startswith("Target"):
            # PDS
            self.match_cols = ["Spectrum Name", "Sample Name", "Target"]
            df = pd.read_csv(self.composition_data_loc)
        elif first_column.startswith("meta"):
            # CCAM
            self.match_cols = ["Sample Name"]

            df = pd.read_csv(self.composition_data_loc, skiprows=1)

            # Drop the columns that contain the quality of the composition data
            drop_cols = [
                "SiO2 Qual.",
                "TiO2 Qual.",
                "Al2O3 Qual.",
                "FeOT Qual.",
                "MnO Qual.",
                "MgO Qual.",
                "CaO Qual.",
                "Na2O Qual.",
                "K2O Qual.",
            ]

            df.drop(drop_cols, axis=1, inplace=True)

            # Rename the columns to match PDS format
            df.rename(
                columns=lambda x: ("Sample Name" if x.strip() == "Target" else x.replace("(wt%)", "").strip()),
                inplace=True,
            )

            # Clean the data
            for column in df.columns:
                # Replace instances of '<' followed by any number with the number itself
                df[column] = df[column].astype(str).str.replace("<", "")
                # Convert all numbers to floats and errors to NaN (non-numeric values become NaN)
                if column not in self.match_cols:
                    df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            raise ValueError(f'Unknown data source: First column "{first_column}" was not recognized.')

        return df


class NonNegativeTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that sets all negative values in a DataFrame to zero.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        # Create a copy of the DataFrame to avoid modifying the original data
        X_transformed = X.copy()

        X_transformed[X_transformed < 0] = 0

        return X_transformed


class CustomSpectralPipeline(BaseEstimator, TransformerMixin):
    """
    A custom spectral pipeline for processing spectral data.
    Custom to the PLS-SM part.

    Args:
        masks (list): List of masks to be applied to the spectral data.
        major_oxides (list): List of major oxides.
        intensity_feature_name (str, optional): Name of the intensity feature.
            Defaults to "shot_avg".
        wavelength_feature_name (str, optional): Name of the wavelength feature.
            Defaults to "wave".
    """

    def __init__(
        self,
        masks,
        composition_data_loc,
        major_oxides,
    ):
        self.pipeline = Pipeline(
            [
                ("mask_transformer", WavelengthMaskTransformer(masks)),
                ("non_negative_transformer", NonNegativeTransformer()),
                ("data_reshaper", SpectralDataReshaper(wavelength_feature_name="wave")),
            ]
        )

        self.composition_data = CompositionData(composition_data_loc)
        self.major_oxides = major_oxides

    def _attach_major_oxides(
        self,
        transformed_df: pd.DataFrame,
        sample_name: str,
        location_name: str,
    ):
        """
        Process a single sample.

        Args:
            transformed_df (pd.DataFrame): DataFrame containing the spectral data for the
                sample.
            sample_name (str): Name of the sample.

        Returns:
            pd.DataFrame: Processed DataFrame for the sample.
        """
        sample_composition = self.composition_data.get_composition_for_sample(sample_name)

        if sample_composition.empty:
            raise ValueError("sample_composition is empty, cannot attach major oxides")

        oxides = sample_composition[self.major_oxides].iloc[0]
        transformed_df = transformed_df.assign(**oxides)

        transformed_df["Sample Name"] = sample_name
        transformed_df["ID"] = f"{sample_name}_{location_name}"

        return transformed_df

    def fit_transform(self, sample_data: dict[str, Dict[str, pd.DataFrame]]):
        """
        Fit and transform the sample data.

        Args:
            sample_data (dict): Dictionary containing the sample data.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        transformed_samples = []

        for sample_name, sample_location_dfs in tqdm(sample_data.items(), desc="Transforming samples"):
            for _, (location_name, sample_df) in enumerate(sample_location_dfs.items()):
                if self.composition_data.get_composition_for_sample(sample_name=sample_name).empty:
                    continue

                transformed_df = self.pipeline.fit_transform(sample_df)

                transformed_df = self._attach_major_oxides(pd.DataFrame(transformed_df), sample_name, location_name)
                transformed_samples.append(transformed_df)

        df_out = pd.concat(transformed_samples, ignore_index=True).rename(columns=str)

        return df_out
