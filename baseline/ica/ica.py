from ica.preprocess import preprocess_data
from ica.jade import JADE


def main():
    processed_data = preprocess_data("./data/data/calib/calib_2015/1600mm/pls")
    num_features = processed_data.shape[1]
    jade_model = JADE(num_components=min(8, num_features))

    jade_model.fit(processed_data.values)
    separated_signals = jade_model.transform(processed_data.values) # Note: separated signals are "scores"

    print("Separated signals shape:", separated_signals.shape)
    print("Separated signals:", separated_signals)

    # Assuming you want to correlate all original features with the independent components
    corrcols = processed_data.columns.tolist()  # All columns in processed_data
    icacols = ['IC' + str(i) for i in range(1, jade_model.num_components + 1)]  # List of independent components

    # Add the separated signals to the processed data for correlation
    for i, col in enumerate(icacols):
        processed_data[col] = separated_signals[:, i]

    # Perform correlation
    jade_model.correlate_loadings(processed_data, corrcols, icacols)

    # Print or inspect the correlation results
    print("ICA-JADE Correlations:", jade_model.ica_jade_corr)
    print("ICA-JADE IDs:", jade_model.ica_jade_ids)


if __name__ == "__main__":
    main()
