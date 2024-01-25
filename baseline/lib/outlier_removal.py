import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn.cross_decomposition import PLSRegression


import matplotlib
matplotlib.use('agg')


def calculate_mahalanobis(x, mean, cov):
    x_minus_mu = x - mean
    return np.sqrt(np.dot(np.dot(x_minus_mu, np.linalg.inv(cov)), x_minus_mu.T))


def calculate_leverage_residuals(model: PLSRegression, X: np.ndarray):
    # -- Leverage --
    t = model.x_scores_
    leverage = np.diag(np.dot(t, np.dot(np.linalg.inv(np.dot(t.T, t)), t.T)))

    # -- Residuals --
    # Reconstruct X from scores (t) and loadings (P)
    X_hat = np.dot(t, model.x_loadings_.T)

    # Calculate the residuals (e)
    e = X - X_hat

    # Calculate the spectral residual (Q)
    Q = np.sum(e**2, axis=1)

    return leverage, Q


def identify_outliers(leverage, Q):
    # Stack leverage and residuals into a 2D array for Mahalanobis calculation
    data = np.column_stack((leverage, Q))
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)

    # Calculate Mahalanobis distance for each point
    distances = np.array([calculate_mahalanobis(point, mean, cov) for point in data])

    # Set threshold based on chi-square distribution for a 95% confidence interval
    threshold = stats.chi2.ppf(
        0.80, df=2
    )  # df=2 because we have two dimensions (leverage and Q)

    # Identify outliers
    outliers = distances > threshold

    # Now you can remove these outliers from your dataset and retrain your model
    return outliers


def plot_leverage_residuals(leverage, Q, outliers, plot_file_path=None) -> None:
    """
    Plot the leverage-residuals plot.

    Parameters:
    - leverage (array-like): Array of leverage values.
    - Q (array-like): Array of residuals.
    - outliers (array-like): Array of outlier indices.
    - plot_file_path (str, optional): File path to save the plot. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(leverage, Q, c="none", edgecolors="blue", alpha=0.5)
    ax.scatter(leverage[outliers], Q[outliers], c="red")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Residuals")
    ax.set_title("Leverage-Residuals Plot")
    # plt.show()

    if plot_file_path:
        fig.savefig(plot_file_path)

    plt.close(fig)
