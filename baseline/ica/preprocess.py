import pandas as pd
from lib.data_handling import get_preprocessed_sample_data, WavelengthMaskTransformer
from lib.reproduction import masks
from lib.norms import Norm1Scaler


def preprocess_data(sample_name, data_path):
    sample_data = get_preprocessed_sample_data(
        sample_name, data_path, average_shots=False
    )

    # For now, we just use the first of the datasets
    df = sample_data[0]

    # Apply masking
    wmt = WavelengthMaskTransformer(masks)
    df = wmt.fit_transform(df)

    # set the wave column as the index
    df.set_index("wave", inplace=True)

    # Normalize the data
    scaler = Norm1Scaler()
    df = pd.DataFrame(scaler.fit_transform(df))

    return df