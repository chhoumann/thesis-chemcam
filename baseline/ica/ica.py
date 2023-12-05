import os
import pandas as pd
from ica.preprocess import preprocess_data
from ica.postprocess import postprocess_data
from ica.jade import JADE


def main():
    root_dir = "./data/calib_2015/1600mm/pls"
    max_runs = 5
    runs = 0

    # create a pandas dataframe for concatenated data
    df = pd.DataFrame()

    for target_dir_name in os.listdir(root_dir):
        target_dir_path = os.path.join(root_dir, target_dir_name)

        data = preprocess_data(target_dir_path)
        separated_signals = run_jade(data)
        data = postprocess_data(target_dir_name, separated_signals)

        df = df.append(data)

        runs += 1

        if runs >= max_runs:
            break

    df.to_csv("./ica_results.csv")



def run_jade(processed_data):
    num_features = processed_data.shape[1]
    jade_model = JADE(num_components=min(8, num_features),)

    jade_model.fit(processed_data.values)
    separated_signals = jade_model.transform(processed_data.values) # Note: separated signals are "scores"

    # Assuming you want to correlate all original features with the independent components
    corrcols = processed_data.columns.tolist()  # All columns in processed_data
    icacols = ['IC' + str(i) for i in range(1, jade_model.num_components + 1)]  # List of independent components

    # Add the separated signals to the processed data for correlation
    for i, col in enumerate(icacols):
        processed_data[col] = separated_signals[:, i]

    # Perform correlation
    jade_model.correlate_loadings(processed_data, corrcols, icacols)

    # Print or inspect the correlation results
    # print("ICA-JADE Correlations:", jade_model.ica_jade_corr)
    # print("ICA-JADE IDs:", jade_model.ica_jade_ids)

    return separated_signals


if __name__ == "__main__":
    main()
