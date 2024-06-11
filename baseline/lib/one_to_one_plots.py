import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def plot_actual_vs_predicted(
    actual_values,
    predicted_values,
    rmse,
    std_dev,
    oxide,
    full_page=False,
    subplot_index=1,
    plot_path="/home/christian/projects/p9/baseline/plots/_one_to_one",
):
    plt.figure(figsize=(12, 24) if full_page else (24, 12))
    plt.subplot(4, 2, subplot_index) if full_page else plt.subplot(2, 4, subplot_index)

    # Create a scatter plot in the subplot
    plt.scatter(
        actual_values, predicted_values, color="black", facecolors="none", edgecolors="black", s=20, label="Predictions"
    )

    # Add a line of perfect predictions
    plt.plot(
        [min(actual_values), max(actual_values)],
        [min(actual_values), max(actual_values)],
        "k-",
        label="Perfect Predictions (1:1)",
    )

    # Fit a regression model and plot the regression line
    model = LinearRegression()
    model.fit(actual_values.to_numpy().reshape(-1, 1), predicted_values)  # Convert to NumPy array and fit model
    line_x = np.linspace(min(actual_values), max(actual_values), 100)
    line_y = model.predict(line_x.reshape(-1, 1))
    plt.plot(line_x, line_y, "r--", label="Regression Line")

    # Calculate R² and add it to the text box
    r_squared = model.score(actual_values.to_numpy().reshape(-1, 1), predicted_values)
    textstr = f"RMSE: {rmse:.4f}\nStd Dev: {std_dev:.4f}\nR²: {r_squared:.4f}"
    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    plt.gca().text(
        0.975,
        0.125,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )

    # Enhancements for each subplot
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Test Set Predictions vs. Certificate Values for {oxide}")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{plot_path}/{oxide}.png")
    plt.show()
