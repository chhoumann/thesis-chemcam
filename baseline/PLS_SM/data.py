import pandas as pd
from pathlib import Path
from typing import Optional


def get_location_dataset_paths_for_sample(sample_name: str, data_path: Path):
    """Get the (five) location datasets for a sample."""
    sample_path = data_path / sample_name
    return [
        f for f in sample_path.iterdir() if f.is_file() and f.suffix == ".csv"
    ]


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

    sample_dataset_paths = get_location_dataset_paths_for_sample(
        sample_name, data_path
    )
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
    - num_samples (Optional[int]): The number of samples to load. If None, load all samples.

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


def split_data(
    dataset: pd.DataFrame, split: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Add your implementation here
    pass
