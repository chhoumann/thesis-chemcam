import pandas as pd


def postprocess_data(sample_name, ica_scores):
    num_ic = ica_scores.shape[1]

    # Generate column names for ICs
    ic_columns = [f'IC{i+1}' for i in range(num_ic)]

    # Create the DataFrame directly from ica_scores
    df = pd.DataFrame(ica_scores, columns=ic_columns)

    # Add 'Shot Number' column
    df['Shot Number'] = range(1, len(df) + 1)

    # Reorder the columns
    columns = ['Shot Number'] + ic_columns
    df = df[columns]

    # Set 'Target ID' as the index
    df.index = [sample_name] * len(df)
    df.index.name = 'Target ID'

    return df
