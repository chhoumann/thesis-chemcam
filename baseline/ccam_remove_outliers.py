import enum
import os
import shutil
from pathlib import Path

import pandas as pd
import typer

from lib.config import AppConfig

app = typer.Typer()

class Mode(enum.Enum):
    exclude = "exclude"
    include = "include"


@app.command()
def main(
    mode: Mode = typer.Option(
        "exclude",
        "--mode",
        "-m",
        help="Mode to run the script in.",
        case_sensitive=False,
    )
):
    config = AppConfig()
    data_path = config.data_path
    master_list_file_path = config.ccam_master_list_file_path

    N_COMMENT_LINES = 1
    master_list_df = pd.read_csv(master_list_file_path, skiprows=N_COMMENT_LINES)

    outlier_rows = master_list_df.loc[master_list_df["Outlier (Exclude)"] == 1].copy()
    outlier_rows["File"] = outlier_rows["File"].str.strip()

    outliers_dir = Path(config.data_cache_dir) / f"{data_path}-outliers"

    if mode == Mode.exclude:
        exclude_outlier_files(data_path, outliers_dir, outlier_rows)
    elif mode == Mode.include:
        include_outlier_files(data_path, outliers_dir)


def exclude_outlier_files(data_path: str, outliers_dir: Path, outlier_rows: pd.DataFrame):
    samples = [entry.name for entry in os.scandir(data_path) if entry.is_dir()]
    files_to_exclude = []

    for sample in samples:
        sample_path = os.path.join(data_path, sample)
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


def include_outlier_files(data_path: str, outliers_dir: Path):
    files_to_include = list(outliers_dir.rglob("*.csv"))

    print(f"Including {len(files_to_include)} outlier files.")

    for file_path in files_to_include:
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(outliers_dir)
        original_file_path = data_path / relative_path
        original_file_path.parent.mkdir(exist_ok=True, parents=True)

        shutil.move(file_path, original_file_path)

        print(f"Moved {file_path.name} from {file_path.parent.name} to {original_file_path}")


if __name__ == "__main__":
    app()
