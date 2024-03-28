from typing import List, Tuple

import numpy as np
import pandas as pd

from lib.reproduction import major_oxides


# Post processing function to be run in parallel
def parallel_postprocess(
    details: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, str, str, int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    (
        df,
        compositions_df,
        ica_estimated_sources,
        sample_name,
        sample_id,
        num_components,
    ) = details

    ic_wavelengths, filtered_compositions_df = _postprocess_df(
        df, compositions_df, ica_estimated_sources, sample_id, num_components
    )
    ic_wavelengths["Sample Name"] = sample_name
    ic_wavelengths["ID"] = sample_id

    return ic_wavelengths, filtered_compositions_df


def _postprocess_df(
    df: pd.DataFrame,
    composition_df: pd.DataFrame,
    ica_estimated_sources: np.ndarray,
    sample_id: str,
    num_components: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    columns = df.columns

    corrcols = [f"IC{i+1}" for i in range(num_components)]
    df_ics = pd.DataFrame(
        ica_estimated_sources,
        index=[f"shot{i+6}" for i in range(45)],
        columns=corrcols,
    )

    df = pd.concat([df, df_ics], axis=1)

    # Correlate the loadings
    corrdf, ids = _correlate_loadings(corrcols, list(columns), df)

    # Create the wavelengths matrix for each component
    ic_wavelengths = pd.DataFrame(columns=columns)

    for i in range(len(ids)):
        ic = ids[i].split(" ")[0]
        component_idx = int(ic[2]) - 1
        wavelength = corrdf.index[i]
        corr = corrdf.iloc[i].iloc[component_idx]

        ic_wavelengths.loc[sample_id, wavelength] = corr

    # Initialize an empty list to store the indices of columns to include
    include_indices = []

    # Iterate over the columns and add the index if the column is in major_oxides
    for i, column in enumerate(composition_df.columns):
        if column in major_oxides:
            include_indices.append(i)

    # Use the list of indices to filter the DataFrame
    filtered_composition_df = composition_df.iloc[:, include_indices]
    filtered_composition_df.index = pd.Index([sample_id])

    return ic_wavelengths, filtered_composition_df


# This is a function that finds the correlation between loadings and a set of columns
# The idea is to somewhat automate identifying which element the loading corresponds to.
def _correlate_loadings(
    corrcols: List, icacols: List, df: pd.DataFrame
) -> Tuple[pd.DataFrame, List]:
    corrdf = df.corr().drop(labels=icacols, axis=1).drop(labels=corrcols, axis=0)
    # set all corrdf nans to 0 - they were set to 0 during masking, and
    # .corr() sets values that don't vary to NaN
    corrdf = corrdf.fillna(0)
    ids = []

    for ic_label in icacols:
        tmp = corrdf.loc[ic_label]
        match = tmp.values == np.max(tmp)
        col = corrcols[np.where(match)[0][-1]]

        ids.append(col + " (r=" + str(np.max(tmp)) + ")")

    return corrdf, ids
