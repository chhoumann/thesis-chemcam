import pandas as pd


def postprocess_data(target_name, ica_scores):
    num_ic = ica_scores.shape[1]

    # Generate column names for ICs
    ic_columns = [f'IC{i+1}' for i in range(num_ic)]

    # Create the DataFrame directly from ica_scores
    data = pd.DataFrame(ica_scores, columns=ic_columns)

    # Add 'Shot Number' column
    data['Shot Number'] = range(1, len(data) + 1)

    # Reorder the columns
    columns = ['Shot Number'] + ic_columns
    data = data[columns]

    # Set 'Target ID' as the index
    data.index = [target_name] * len(data)
    data.index.name = 'Target ID'

    return data
