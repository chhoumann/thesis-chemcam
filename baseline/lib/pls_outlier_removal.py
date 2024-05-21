from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

from lib.outlier_removal import (
    calculate_leverage_residuals,
    identify_outliers,
    plot_leverage_residuals,
)


def remove_outliers(
    train: pd.DataFrame,
    drop_cols: list[str],
    oxide: str,
    n_components: int,
    outlier_removal: bool,
    outlier_removal_constraint_iteration: int,
    influence_plot_dir: Path,
    experiment_name: str,
    compositional_range: str,
) -> pd.DataFrame:
    outlier_removal_iterations = 0
    pls_OR = PLSRegression(n_components=n_components)
    X_train_OR = train.drop(columns=drop_cols).to_numpy()
    y_train_OR = train[oxide].to_numpy()

    pls_OR.fit(X_train_OR, y_train_OR)

    current_performance = mean_squared_error(y_train_OR, pls_OR.predict(X_train_OR), squared=False)

    mlflow.log_metric(
        "RMSEOR",
        float(current_performance),
        step=outlier_removal_iterations,
    )

    best_or_model = pls_OR

    train_no_outliers = train.copy()

    leverage, Q = None, None

    while True and outlier_removal:
        outlier_removal_iterations += 1

        should_calculate_constraints = (
            outlier_removal_constraint_iteration >= 0
            and outlier_removal_iterations <= outlier_removal_constraint_iteration
        ) or outlier_removal_constraint_iteration < 0
        print(should_calculate_constraints)

        if should_calculate_constraints:
            leverage, Q = calculate_leverage_residuals(pls_OR, X_train_OR)
            print(leverage, Q)

        outliers = identify_outliers(leverage, Q)

        if len(outliers) == 0:
            break

        outliers_indices = np.where(outliers)[0]

        # Plotting the influence plot
        plot_path = Path(
            influence_plot_dir
            / f"{experiment_name}"
            / f"{oxide}_{compositional_range}_{outlier_removal_iterations}.png"
        )
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        plot_leverage_residuals(leverage, Q, outliers, str(plot_path))
        # mlflow.log_artifact(str(plot_path))

        X_train_OR = np.delete(X_train_OR, outliers_indices, axis=0)
        y_train_OR = np.delete(y_train_OR, outliers_indices, axis=0)

        if should_calculate_constraints:
            pls_OR = PLSRegression(n_components=n_components)
            pls_OR.fit(X_train_OR, y_train_OR)

        new_performance = mean_squared_error(y_train_OR, pls_OR.predict(X_train_OR), squared=False)

        mlflow.log_metric(
            "RMSEOR",
            float(new_performance),
            step=outlier_removal_iterations,
        )

        number_of_outliers = np.sum(outliers)  # Counting the number of True values
        mlflow.log_metric(
            "outliers_removed",
            float(number_of_outliers),
            step=outlier_removal_iterations,
        )

        # Check if error has increased: early stop if so
        if float(new_performance) >= float(current_performance):
            break

        # Update to only have best set
        train_no_outliers = train.drop(index=train.index[outliers_indices])

        current_performance = new_performance
        best_or_model = pls_OR

    mlflow.log_metric("outlier_removal_iterations", outlier_removal_iterations)
    # mlflow.sklearn.log_model(best_or_model, f"PLS_OR_{oxide}_{compositional_range}")

    return train_no_outliers
