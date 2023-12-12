import pandas as pd
import numpy as np


def postprocess_data(df: pd.DataFrame, composition_df: pd.DataFrame, sample_name: str, estimated_sources: np.ndarray, num_components: int):
    columns = df.columns
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

    # Filter the composition data to only include the oxides and their compositions
    composition_df = composition_df.iloc[:, 3:12]
    composition_df.index = [sample_name]

    return ic_wavelengths, composition_df


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