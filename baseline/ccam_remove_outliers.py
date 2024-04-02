import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

from lib.config import AppConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["exclude", "include"],
        default="exclude",
        help='Mode to run the script in: "exclude" or "include". Default is "exclude".',
    )

    return parser.parse_args()


def main(mode: str = "exclude"):
    config = AppConfig()
    data_path = config.data_path

    master_list_file_name = config.ccam_master_list_file_name
    master_list_file_path = os.path.join(data_path, master_list_file_name)

    master_list_df = pd.read_csv(master_list_file_path, skiprows=1)

    outlier_rows = master_list_df.loc[master_list_df["Outlier (Exclude)"] == 1].copy()
    outlier_rows["File"] = outlier_rows["File"].str.strip()

    outliers_dir = Path(config.data_cache_dir) / f"{data_path}-outliers"

    if mode == "exclude":
        exclude_outlier_files(data_path, outliers_dir, outlier_rows)
    elif mode == "include":
        include_outlier_files(data_path, outliers_dir)


def exclude_outlier_files(
    data_path: str, outliers_dir: Path, outlier_rows: pd.DataFrame
):
    samples = [entry.name for entry in os.scandir(data_path) if entry.is_dir()]
    files_to_exclude = []

    for sample in samples:
        sample_path = os.path.join(data_path, sample)

        # Get all files whose file extension is .csv in the directory
        csv_files = [file for file in os.listdir(sample_path) if file.endswith(".csv")]

        if len(csv_files) == 0:
            continue

        for csv_file in csv_files:
            if csv_file in outlier_rows["File"].values:
                file_path = os.path.join(sample_path, csv_file)
                files_to_exclude.append(file_path)

    print(f"Excluding {len(files_to_exclude)} outlier files.")

    outliers_dir.mkdir(exist_ok=True, parents=True)

    for file in files_to_exclude:
        file_path = Path(file)
        relative_path = file_path.relative_to(data_path)
        new_file_path = outliers_dir / relative_path
        new_file_path.parent.mkdir(exist_ok=True, parents=True)

        shutil.move(file, new_file_path)

        print(f"Moved {file_path.name} from {file_path.parent.name} to {new_file_path}")

        files_to_include = list(outliers_dir.rglob("*.csv"))


def include_outlier_files(data_path: str, outliers_dir: Path):
    files_to_include = list(outliers_dir.rglob("*.csv"))

    print(f"Including {len(files_to_include)} outlier files.")

    for file_path in files_to_include:
        if not file_path.is_file():
            continue

        # Calculate the relative path of the file with respect to the outliers directory
        relative_path = file_path.relative_to(outliers_dir)
        # Construct the original file path
        original_file_path = data_path / relative_path
        # Ensure the directory exists
        original_file_path.parent.mkdir(exist_ok=True, parents=True)

        # Move file back to the original location
        shutil.move(file_path, original_file_path)

        print(
            f"Moved {file_path.name} from {file_path.parent.name} to {original_file_path}"
        )


if __name__ == "__main__":
    args = parse_args()
    main(mode=args.mode)
