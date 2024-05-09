from typing import List


def get_preprocess_fn(preprocessor, target_col: str, drop_cols: List[str]):
    """
    Creates a preprocessing function configured with a specific preprocessor, target column, and columns to drop.

    This function is designed to be used in a machine learning pipeline where data needs to be preprocessed
    before being fed into a model. It handles the fitting and transformation of training data and the
    transformation of test data using the provided preprocessor.

    Parameters:
    - preprocessor: The preprocessor instance (e.g., a Scikit-learn transformer object).
    - target_col (str): The name of the target column in the dataframe.
    - drop_cols (List[str]): List of column names to be dropped from the dataframe.

    Returns:
    - preprocess_fn: A function that takes training and testing dataframes, drops specified columns,
                     separates the target column, and applies the preprocessor to the feature columns.
    """
    assert target_col in drop_cols, "Target column should be included in the drop columns"

    def preprocess_fn(train, test):
        X_train = train.drop(columns=drop_cols)
        y_train = train[target_col]

        X_test = test.drop(columns=drop_cols)
        y_test = test[target_col]

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        return X_train_transformed, y_train, X_test_transformed, y_test

    return preprocess_fn