from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def get_location_dataset_paths_for_sample(sample_name: str, data_path: Path):
    """Get the (five) location datasets for a sample."""
    sample_path = data_path / sample_name
    return [f for f in sample_path.iterdir() if f.is_file() and f.suffix == ".csv"]


def get_dataset_frame(dataset_path):
    with open(dataset_path) as f:
        # find index of last line starting with "#" and skip rows until then
        target = 0
        for i, line in enumerate(f):
            if not line.startswith("#"):
                target = i
                break

        # read csv from that line - columns also start wih "#"
        return pd.read_csv(dataset_path, skiprows=target - 1)


def get_preprocessed_sample_data(
    sample_name: str, data_path: Path
) -> list[pd.DataFrame]:
    exclude_from_avg = ["wave", "mean", "median"]
    first_five_shots = [f"shot{i}" for i in range(1, 6)]

    wavelengths = pd.Series()

    sample_dataset_paths = get_location_dataset_paths_for_sample(sample_name, data_path)
    sample_spectra = []

    for i, sample_set in enumerate(sample_dataset_paths):
        df = get_dataset_frame(sample_set)

        # strip whitespace from column names
        df.columns = df.columns.str.strip()
        # remove # from column names
        df.columns = df.columns.str.replace("# ", "")

        if i == 0:
            wavelengths = df["wave"]
        else:
            assert wavelengths.equals(df["wave"])

        df.drop(exclude_from_avg, axis=1, inplace=True)
        df.drop(first_five_shots, axis=1, inplace=True)

        # re-insert wavelengths to avoid averaging them
        df.insert(0, "wave", wavelengths)

        # add average of all shots and remove individual shots
        shot_cols = [col for col in df.columns if "shot" in col]
        shot_avg = df[shot_cols].mean(axis=1)
        df.drop(shot_cols, axis=1, inplace=True)
        df.insert(1, "shot_avg", shot_avg)

        sample_spectra.append(df)

    return sample_spectra


def load_data(
    dataset_loc: str, num_samples: Optional[int] = None
) -> dict[str, list[pd.DataFrame]]:
    """
    Load data from the specified dataset location.

    Parameters:
    - dataset_loc (str): The location of the dataset.
    - num_samples (Optional[int]): The number of samples to load.
    If None, load all samples.

    Returns:
    - Dataset: The loaded dataset.
    """
    sample_data: dict[str, list[pd.DataFrame]] = {}
    data_path = Path(dataset_loc).resolve(strict=True)
    sample_names = [f.name for f in data_path.iterdir() if f.is_dir()]

    take_amount = num_samples if num_samples else len(sample_names)

    for _sample_name in sample_names[:take_amount]:
        sample_data[_sample_name] = get_preprocessed_sample_data(
            _sample_name, data_path
        )

    return sample_data


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

        for mask in self.masks:
            X = X.loc[~((X["wave"] >= mask[0]) & (X["wave"] <= mask[1]))]

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
            "Input should be a dictionary with keys as sample names and"
            + " values as lists of pandas DataFrames."
        )

    transformed_data = {}
    for sample_name, dfs in sample_data.items():
        transformed_dfs = [transformer.transform(df) for df in dfs]
        transformed_data[sample_name] = transformed_dfs

    return transformed_data


class SpectralDataReshaper(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        intensity_feature_name: str,
        wavelength_feature_name: str,
    ):
        self.intensity_feature_name = intensity_feature_name
        self.wave_feature_name = wavelength_feature_name
        self.sample_size_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input should be a pandas DataFrame.")

        self.sample_size_ = len(X) // len(X[self.wave_feature_name].unique())
        return self

    def transform(self, X):
        if self.sample_size_ is None:
            raise RuntimeError("Transformer must be fitted before calling transform.")

        reshaped_values = X[self.intensity_feature_name].values.reshape(
            self.sample_size_, -1
        )
        transformed_df = pd.DataFrame(
            reshaped_values, columns=X[self.wave_feature_name].unique()
        )

        return transformed_df


def attach_major_oxides(
    transformed_df: pd.DataFrame,
    sample_composition: pd.DataFrame,
    major_oxides: list[str],
):
    if sample_composition.empty:
        raise ValueError("sample_composition is empty, cannot attach major oxides")
    oxides = sample_composition[major_oxides].iloc[0]
    transformed_df = transformed_df.assign(**oxides)

    return transformed_df


class CompositionData:
    def __init__(self, composition_data_loc: str):
        self.composition_data = self.load_composition_data(composition_data_loc)

    @staticmethod
    def load_composition_data(composition_data_loc: str) -> pd.DataFrame:
        return pd.read_csv(composition_data_loc)

    def get_composition_for_sample(self, sample_name) -> pd.DataFrame:
        sample_name_lower = sample_name.lower()
        match_condition = (
            (self.composition_data["Spectrum Name"].str.lower() == sample_name_lower)
            | (self.composition_data["Target"].str.lower() == sample_name_lower)
            | (self.composition_data["Sample Name"].str.lower() == sample_name_lower)
        )
        composition = self.composition_data.loc[match_condition]

        # if composition.empty:
        #     raise ValueError(f"Could not find composition for sample: {sample_name}")

        return composition

    def create_sample_compositions_dict(self, sample_names) -> dict[str, pd.DataFrame]:
        sample_compositions = {}
        for sample_name in sample_names:
            comp = self.get_composition_for_sample(sample_name)
            if comp.empty:
                print(f"Could not find {sample_name} in labels")
                continue
            sample_compositions[sample_name] = comp
        return sample_compositions


class CustomSpectralPipeline:
    def __init__(
        self,
        masks,
        composition_data_loc,
        major_oxides,
        intensity_feature_name="shot_avg",
        wavelength_feature_name="wave",
    ):
        self.mask_transformer = WavelengthMaskTransformer(masks)
        self.data_reshaper = SpectralDataReshaper(
            intensity_feature_name, wavelength_feature_name
        )
        self.composition_data = CompositionData(composition_data_loc)
        self.major_oxides = major_oxides

    def process_sample(
        self,
        sample_df: pd.DataFrame,
        sample_name: str,
    ):
        masked_df = self.mask_transformer.transform(sample_df)
        reshaped_df = self.data_reshaper.fit_transform(masked_df)

        sample_composition = self.composition_data.get_composition_for_sample(
            sample_name
        )
        final_df = attach_major_oxides(
            pd.DataFrame(reshaped_df), sample_composition, self.major_oxides
        )

        final_df["Sample Name"] = sample_name

        return final_df

    def fit_transform(self, sample_data: dict[str, list[pd.DataFrame]]):
        transformed_samples = []
        for sample_name, sample_dfs in sample_data.items():
            for _, sample_df in enumerate(sample_dfs):
                if self.composition_data.get_composition_for_sample(sample_name).empty:
                    continue
                transformed_df = self.process_sample(sample_df, sample_name)
                transformed_samples.append(transformed_df)
        df_out = pd.concat(transformed_samples, ignore_index=True).rename(columns=str)
        return df_out
