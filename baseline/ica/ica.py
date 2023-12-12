import os
import pandas as pd
import numpy as np

from pathlib import Path
from ica.preprocess import preprocess_data
from ica.postprocess import postprocess_data
from ica.jade import JADE
from sklearn.decomposition import FastICA
from lib.data_handling import CompositionData
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lib.reproduction import major_oxides


def main():
    data_path = Path("./data/calib_2015/1600mm/pls")
    max_runs = 30
    runs = 0
    num_components = 8

    composition_data = CompositionData("./data/data/ccam_calibration_compositions.csv")

    ica_df = pd.DataFrame()
    compositions = pd.DataFrame()

    for sample_name in os.listdir(data_path):
        # Check if we have composition data for this sample
        composition_data_for_sample = composition_data.get_composition_for_sample(sample_name)

        if composition_data_for_sample.empty:
            print(f"No composition data found for {sample_name}. Skipping...")
            continue

        # Check if the composition data contains NaN values
        if composition_data_for_sample.isnull().values.any():
            print(f"NaN values found in composition data for {sample_name}. Skipping...")
            continue

        print(f"Processing {sample_name}...")

        # Preprocess the data
        df = preprocess_data(sample_name, data_path)
        columns = df.columns

        # Run ICA and get the estimated sources
        estimated_sources = run_ica(df, model="jade", num_components=num_components)

        #  Add the estimated sources to the DataFrame for postprocessing
        corrcols = [f'IC{i+1}' for i in range(num_components)]
        df_ics = pd.DataFrame(estimated_sources, index=[f'shot{i+6}' for i in range(45)], columns=corrcols)
        df = pd.concat([df, df_ics], axis=1)

        # Correlate the loadings
        corrdf, ids = correlate_loadings(df, corrcols, columns)

        # Create the wavelengths matrix for each component
        ic_wavelengths = pd.DataFrame(index=[sample_name], columns=columns)

        for i in range(len(ids)):
            ic = ids[i].split(' ')[0]
            component_idx = int(ic[2]) - 1
            wavelength = corrdf.index[i]
            corr = corrdf.iloc[i][component_idx]

            ic_wavelengths.loc[sample_name, wavelength] = corr

        # Aggregate the composition data and the ICA results to their respective DataFrames
        composition_data_for_sample = composition_data_for_sample.iloc[:, 3:12]
        composition_data_for_sample.index = [sample_name]

        compositions = pd.concat([compositions, composition_data_for_sample])
        ica_df = pd.concat([ica_df, ic_wavelengths])

        runs += 1

        if runs >= max_runs:
            break

    print(f"ICA results shape = {ica_df.shape}, composition data shape = {compositions.shape}.\n")

    print(f"ICA results:\n{ica_df}\n")
    print(f"Composition data:\n{compositions}\n")

    # Split the data into training and testing sets
    ica_train, ica_test, comp_train, comp_test = train_test_split(ica_df, compositions, test_size=0.2, random_state=42)

    # Train a linear regression model for each oxide
    for oxide in major_oxides:
        X_train = ica_train
        y_train = comp_train[oxide]

        X_test = ica_test
        y_test = comp_test[oxide]

        model = LinearRegression()
        model.fit(X_train, y_train)

        model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

        print(f"Testing RMSE for {oxide}: {test_rmse}\n")


def run_ica(df: pd.DataFrame, model: str = "fastica", num_components: int = 8) -> np.ndarray:
    """
    Performs Independent Component Analysis (ICA) on a given dataset using JADE or FastICA algorithms.

    Parameters:
    ----------
    df : pd.DataFrame
        The input dataset for ICA. The DataFrame should have rows as samples and columns as features.

    model : str, optional
        The ICA model to be used. Must be either 'jade' or 'fastica'. Defaults to 'fastica'.

    num_components : int, optional
        The number of independent components to be extracted. Defaults to 8.

    Returns:
    -------
    np.ndarray
        An array of the estimated independent components extracted from the input data.

    Raises:
    ------
    ValueError
        If an invalid model name is specified.

    AssertionError
        If the extracted signals are not independent, indicated by the correlation matrix not being
        close to the identity matrix or the sum of squares of correlations not being close to one.
    """
    estimated_sources = None

    if model == "jade":
        # cols = df.columns
        jade_model = JADE(num_components)
        mixing_matrix = jade_model.fit(df)
        estimated_sources = jade_model.transform(df)
    elif model == "fastica":
        fastica_model = FastICA(n_components=num_components, max_iter=5000)
        estimated_sources = fastica_model.fit_transform(df)
    else:
        raise ValueError("Invalid model specified. Must be 'jade' or 'fastica'.")

    return estimated_sources


# This is a function that finds the correlation between loadings and a set of columns
# The idea is to somewhat automate identifying which element the loading corresponds to.
def correlate_loadings(df, corrcols, icacols):
    corrdf = df.corr().drop(labels=icacols, axis=1).drop(labels=corrcols, axis=0)
    ids = []

    for ic_label in icacols:
        tmp = corrdf.loc[ic_label]
        match = tmp.values == np.max(tmp)
        col = corrcols[np.where(match)[0][-1]]

        ids.append(col + ' (r=' + str(np.max(tmp)) + ')')

    return corrdf, ids


if __name__ == "__main__":
    main()
