import pandas as pd
import os
from reproduction import masks


def average_datasets(parent_directory: str) -> pd.DataFrame:
    aggregated_data = []

    for subdirpath, subdirnames, _ in os.walk(parent_directory):
        for subdirname in subdirnames:
            dir_path = os.path.join(subdirpath, subdirname)
            csv_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.csv')]

            for file_path in csv_files:
                df = initial_preprocess(file_path)
                aggregated_data.append(df)

    if not aggregated_data:
        raise ValueError("No data to process.")

    # Concatenate all DataFrames along the rows
    combined_df = pd.concat(aggregated_data, axis=0)

    # Calculate the mean across rows for each wavelength
    # This assumes that the index of each DataFrame is the wavelength
    averaged_df = combined_df.groupby('wave').mean()

    return averaged_df


def initial_preprocess(file_path: str) -> pd.DataFrame:
    df = get_dataset_frame(file_path)

    # Clean up column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("# ", "")

    # Drop specific columns
    exclude = ["mean", "median"]
    first_five_shots = [f"shot{i}" for i in range(1, 6)]
    df.drop(exclude + first_five_shots, axis=1, inplace=True)

    # Apply any masking required
    for mask in masks:
        df = df.loc[~((df["wave"] >= mask[0]) & (df["wave"] <= mask[1]))]

    return df


def get_dataset_frame(dataset_path: str) -> pd.DataFrame:
    with open(dataset_path) as f:
        # Find index of last line starting with "#" and skip rows until then
        for i, line in enumerate(f):
            if not line.startswith("#"):
                break

        # Read CSV from that line - columns also start with "#"
        return pd.read_csv(dataset_path, skiprows=i-1)


def variance_based_selection(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate variances without transposing
    variances = df.var(axis=1)
    threshold = variances.mean()

    # Select wavelengths based on the threshold
    selected_wavelengths = variances[variances > threshold].index
    df_selected = df.loc[selected_wavelengths]

    # Transpose the DataFrame for ICA
    df_transposed = df_selected.transpose()

    return df_transposed


def preprocess_data(parent_directory: str) -> pd.DataFrame:
    data = average_datasets(parent_directory)
    data = variance_based_selection(data)

    return data
