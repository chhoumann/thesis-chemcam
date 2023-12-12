import os
import pandas as pd
import numpy as np

from pathlib import Path
from ica.preprocess import preprocess_data
from ica.postprocess import postprocess_data
from ica.jade import JADE
from sklearn.decomposition import FastICA
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def main():
    data_path = Path("./data/calib/calib_2015/1600mm/pls/")
    max_runs = 4
    runs = 0
    num_components = 15
    all_estimated_sources = []

    df = pd.DataFrame()

    for sample_name in os.listdir(data_path):
        X = preprocess_data(sample_name, data_path)
        estimated_sources = run_ica(X, model="jade", num_components=num_components)
        df = pd.concat([df, postprocess_data(sample_name, estimated_sources)])

        estimated_sources_df = pd.DataFrame(estimated_sources)
        all_estimated_sources.append(estimated_sources_df)

        runs += 1

        if runs >= max_runs:
            break

    combined_sources = pd.concat(all_estimated_sources, ignore_index=True)
    linear_regression(combined_sources)
    df.to_csv("./ica_results.csv")


def run_ica(df: pd.DataFrame, model: str = "fastica", num_components: int = 8) -> np.ndarray:
    """
    Performs Independent Component Analysis (ICA) on a given dataset using JADE or FastICA algorithms.

    Parameters:
    ----------
    df : pd.DataFrame
        The input dataset for ICA. The DataFrame should have rows as samples and columns as features.

    model : str, optional
        The ICA model to be used. Must be either 'jade' or 'fastica'. Defaults to 'fastica'.

    num_components : int, optional
        The number of independent components to be extracted. Defaults to 8.

    Returns:
    -------
    np.ndarray
        An array of the estimated independent components extracted from the input data.

    Raises:
    ------
    ValueError
        If an invalid model name is specified.

    AssertionError
        If the extracted signals are not independent, indicated by the correlation matrix not being
        close to the identity matrix or the sum of squares of correlations not being close to one.
    """
    estimated_sources = None
    oxide_ranges = {
    "SiO2": [(278, 289), (500, 550)],
    "TiO2": [(320, 336), (439, 441), (454, 456)],
    "FeOT": [(247, 276), (404, 406), (425, 425.5), (426.2, 427), (435, 436)],
    "MgO":  [(276, 277), (519, 525)],
    "Al2O3": [(308, 310), (386, 394)],
    "CaO": [(314, 316), (394, 397), (415, 422), (618, 624)],
    "Na2O": [(590, 600), (818, 822)],
    "K2O": [(745, 768)]
    }

    if model == "jade":
        jade_model = JADE(num_components)
        df = df.transpose()
        mixing_matrix = jade_model.fit(df)
        estimated_sources = jade_model.transform(df)
        

        ## _____ All this is just verification ____ ##
        
        # Find correlation:
        feature_indices = df.columns.values
        unfiltered_correlation_matrix = np.corrcoef(df, estimated_sources, rowvar=False)
        number_of_ics = estimated_sources.shape[1] 
        number_of_features = df.shape[1]
        filtered_correlation_matrix = unfiltered_correlation_matrix[-number_of_ics:, :number_of_features]
        final_results = summarize_and_identify_oxides(filtered_correlation_matrix, feature_indices, oxide_ranges)
        #print(final_results)

        # Verification that the output is correct
        #print("transposed: ", mixing_matrix.shape)
        mixing_matrix = mixing_matrix.T
        unmixing_matrix = np.linalg.pinv(mixing_matrix)
        reconstructed = np.dot(estimated_sources, unmixing_matrix)
        #print("Reconstructed shape Jade: ", reconstructed.shape)

        squared_differences = (reconstructed - df) ** 2
        mean_squared_error = squared_differences.values.flatten().mean()
        rmse = np.sqrt(mean_squared_error)
        #print("RMSE for Jade: ", rmse)
        #linear_regression()
        #print(correlation_matrix)
        #print(reconstructed)
        #print(df)
        
        ## ___ Verification End ___ ##

    elif model == "fastica":
        fastica_model = FastICA(n_components=num_components, max_iter=5000)
        df = df.transpose()
        estimated_sources = fastica_model.fit_transform(df)
        print("mixing matrix: ", fastica_model.components_.shape)    
        print("Estimated sources: ", estimated_sources.shape)
        reconstructed = fastica_model.inverse_transform(estimated_sources)
        print("Reconstructed shape FastICA ", reconstructed.shape)

        squared_differences = (reconstructed - df) ** 2
        mean_squared_error = squared_differences.values.flatten().mean()
        rmse = np.sqrt(mean_squared_error)
        print("RMSE for FastICA: ", rmse)
        #print(reconstructed)
        #print(df)
    else:
        raise ValueError("Invalid model specified. Must be 'jade' or 'fastica'.")

    #correlation_matrix = np.corrcoef(estimated_sources.T)

    # Independence check
    #independence = np.allclose(correlation_matrix, np.eye(correlation_matrix.shape[0]), atol=0.1)

    # Sum of squares check
    #sum_of_squares = np.sum(correlation_matrix**2, axis=1)
    #sum_of_squares_close_to_one = np.allclose(sum_of_squares, np.ones(sum_of_squares.shape[0]))

    #assert independence and sum_of_squares_close_to_one, "ICA failed. Extracted signals are not independent because the correlation matrix is not close to the identity matrix."

    return estimated_sources

def summarize_and_identify_oxides(ic_feature_correlation, feature_indices, oxide_ranges):
    n_ics = ic_feature_correlation.shape[0]  # Number of ICs
    ic_identifiers = [f'IC{i+1}' for i in range(n_ics)]
    
    most_corr_wavelengths = []
    corr_coefficients = []

    # Iterate through the ICs to find the most correlated wavelength for each
    for ic_index in range(n_ics):
        ic_correlations = ic_feature_correlation[ic_index, :]  # Get correlations for this IC
        max_corr_index = np.argmax(np.abs(ic_correlations))  # Index of max correlation
        max_corr_value = ic_correlations[max_corr_index]  # Value of max correlation
        max_corr_wavelength = feature_indices[max_corr_index]  # Wavelength with max correlation

        most_corr_wavelengths.append(max_corr_wavelength)
        corr_coefficients.append(max_corr_value)

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'IC Identifier': ic_identifiers,
        'Most Correlated Wavelength': most_corr_wavelengths,
        'Correlation Coefficient': corr_coefficients
    })

    # New DataFrame to store results with identified oxides
    results_df = pd.DataFrame(columns=['IC Identifier', 'Most Correlated Wavelength', 'Correlation Coefficient', 'Identified Oxide'])

    # Iterate over each row in summary_df to identify oxides
    for index, row in summary_df.iterrows():
        wavelength = row['Most Correlated Wavelength']
        oxide = find_oxide(wavelength, oxide_ranges)
        new_row = pd.DataFrame([{
            'IC Identifier': row['IC Identifier'],
            'Most Correlated Wavelength': wavelength,
            'Correlation Coefficient': row['Correlation Coefficient'],
            'Identified Oxide': oxide
        }])
        results_df = pd.concat([results_df, new_row], ignore_index=True)

    return results_df


def find_oxide(wavelength, oxide_ranges):
    for oxide, ranges in oxide_ranges.items():
        for range in ranges:
            if range[0] <= wavelength <= range[1]:
                return oxide
    return "No matching oxide found"

def mad_based_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False otherwise.
    """
    median = np.median(points)
    diff = np.abs(points - median)
    mad = np.median(diff)
    modified_z_score = 0.6745 * diff / mad
    return modified_z_score > thresh

def linear_regression(estimated_sources):
    print("Shape", estimated_sources.shape)
    scores = np.array(estimated_sources)
    composition_data = {
         "SiO2": [57.9, 53.94, 77.94, 72.2], 
         "TiO2": [0.65, 0.9, 0.31, 0.32],
         "Al2O3": [16.6, 26.15, 10.97, 13.07], 
         "FeOT": [7.22, 4.29, 2.76, 3.21],
         "MnO": [0.14, 0.051, 0.02, 0.1],
         "MgO": [3.81, 3.11, 1.18, 0.21],
         "CaO": [4.52, 0.05, 1.33, 0.96], 
         "Na2O": [4.61, 0.59, 2.95, 4.91], 
         "K2O": [1.33, 5.31, 1.6, 3.95] 
         }

    print("Length of composition", len(composition_data))
    print("Length of scores", len(scores))

    composition_df = pd.DataFrame(composition_data)
    epsilon = 0.001  # Small constant to avoid log(0)
    transformed_composition = np.log((composition_df + epsilon)**2)

    models = {}

    for element, values in composition_data.items():
        y = np.log((np.array(values) + epsilon)**2)  # Log-square transformation
        print(y)
        outlier_mask = np.zeros(len(scores), dtype=bool)

        # Iteratively remove outliers
        for component in scores.T:
            outlier_mask |= mad_based_outlier(component)

        # Filter out the outliers
        filtered_scores = scores[~outlier_mask]

        # Create and fit the linear regression model
        model = LinearRegression()
        model.fit(filtered_scores, y)
        models[element] = model

        print(f"Model for {element}:")
        print("Coefficients:", model.coef_)
        print("Intercept:", model.intercept_)
        print()

    # Example of how you might use the model for predictions
    for element, values in composition_data.items():
        model = models[element]

        # Predict using the model
        predictions = model.predict(scores)
        predicted_values = np.sqrt(np.exp(predictions)) - epsilon
        print(f"Predicted values for {element}", predicted_values)

        # Actual values
        actual_values = np.sqrt(np.exp(np.log((np.array(values) + epsilon)**2))) - epsilon

        # Plot actual vs predicted values
        plt.figure()
        plt.scatter(range(len(actual_values)), actual_values, color='blue', label='Actual')
        plt.scatter(range(len(actual_values)), predicted_values, color='red', label='Predicted')
        plt.title(f"Actual vs Predicted for {element}")
        plt.xlabel('Observation')
        plt.ylabel(f'{element} Composition')
        plt.legend()
        plt.savefig(f'{element}_regression_plot.png')
        plt.close()
        
    return models

    predicted_log_square = models["SiO2"].predict(scores)

    # Back-transform predictions
    predicted_original = np.sqrt(np.exp(predicted_log_square)) - epsilon
    
    # Plot the results

if __name__ == "__main__":
    main()
