from pathlib import Path

import pandas as pd
from dotenv import dotenv_values
from sklearn.model_selection import train_test_split

from lib.data_handling import CompositionData, load_data
from lib.reproduction import folder_to_composition_sample_name, major_oxides

env = dotenv_values()
comp_data_loc = env.get("COMPOSITION_DATA_PATH")
dataset_loc = env.get("DATA_PATH")

if not comp_data_loc:
    print("Please set COMPOSITION_DATA_PATH in .env file")
    exit(1)

if not dataset_loc:
    print("Please set DATA_PATH in .env file")
    exit(1)

cd = CompositionData(composition_data_loc=comp_data_loc)
cd.composition_data.head()


def get_composition_for_sample(cd, sample_name):
    sample_name_lower = sample_name.lower()
    match_condition = (
        (cd["Spectrum Name"].str.lower() == sample_name_lower)
        | (cd["Target"].str.lower() == sample_name_lower)
        | (cd["Sample Name"].str.lower() == sample_name_lower)
    )
    composition = cd.loc[match_condition]

    return composition.head(1)


if __name__ == "__main__":
    save_path = Path("train_test_split.csv")

    if save_path.exists():
        print("train_test_split.csv already exists. Skipping...")
        exit()

    # quick and easy way to get the list of samples we have
    data = load_data(dataset_loc)

    # get list of samples we have that were used for 2015 calibration
    samples_used_2015 = []
    for sample in data.keys():
        sample_name = folder_to_composition_sample_name.get(sample, sample)
        composition = get_composition_for_sample(cd.composition_data, sample_name)

        if composition.empty:
            continue

        # drop samples with NaNs for any of the oxides
        if composition[major_oxides].isnull().values.any():
            continue

        used_2015 = composition["Used for 2015 calibration"].values[0]
        if used_2015 == 1:
            samples_used_2015.append(sample)

    train, test = train_test_split(samples_used_2015, test_size=0.2, random_state=42)

    train_df = pd.DataFrame(train, columns=["sample_name"])
    train_df["train_test"] = "train"

    test_df = pd.DataFrame(test, columns=["sample_name"])
    test_df["train_test"] = "test"

    train_test_df = pd.concat([train_df, test_df])
    train_test_df.to_csv(str(save_path), index=False)
