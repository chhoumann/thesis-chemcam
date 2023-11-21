from sklearn.model_selection import KFold, train_test_split
import pandas as pd


def custom_kfold_cross_validation(
    data, k: int, group_by: str, random_state=None
):
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    grouped = data.groupby(group_by)
    groups_keys = list(grouped.groups.keys())

    for train_keys_idx, test_keys_idx in kf.split(groups_keys):
        train_keys = [groups_keys[idx] for idx in train_keys_idx]
        test_keys = [groups_keys[idx] for idx in test_keys_idx]

        train_data = pd.concat([grouped.get_group(key) for key in train_keys])
        test_data = pd.concat([grouped.get_group(key) for key in test_keys])

        yield train_data, test_data


def custom_train_test_split(
    data, group_by: str, test_size=0.2, random_state=None
):
    grouped = data.groupby(group_by)
    groups_keys = list(grouped.groups.keys())

    train_keys, test_keys = train_test_split(
        groups_keys, test_size=test_size, random_state=random_state
    )

    train_data = pd.concat([grouped.get_group(key) for key in train_keys])
    test_data = pd.concat([grouped.get_group(key) for key in test_keys])

    return train_data, test_data
