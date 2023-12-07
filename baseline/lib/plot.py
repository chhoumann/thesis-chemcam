from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from lib.data_handling import WavelengthMaskTransformer, get_preprocessed_sample_data
from lib.reproduction import masks, spectrometer_wavelength_ranges


def plot_spectra(
    sample_name: str,
    data_path: str,
    average_shots=True,
    max_gap_threshold=10,
    fig_name="",
):
    sns.set_style("white")  # Set the Seaborn style
    _data_path = Path(data_path)
    assert _data_path.exists(), f"Data path {data_path} does not exist."

    # Get the preprocessed sample data
    sample_data = get_preprocessed_sample_data(
        sample_name, _data_path, average_shots=average_shots
    )
    first_location = sample_data[0]

    # Transform the data using WavelengthMaskTransformer
    wmt = WavelengthMaskTransformer(masks)
    transformed = wmt.fit_transform(first_location)

    df = pd.DataFrame(
        {"wave": transformed["wave"], "shot_avg": transformed["shot_avg"]}
    )

    # Debug: Print the DataFrame structure
    print("DataFrame columns:", df.columns)

    # Check if 'wave' and 'shot_avg' are in the DataFrame
    if "wave" not in df.columns or "shot_avg" not in df.columns:
        raise ValueError(
            "Expected columns 'wave' and 'shot_avg' not found in the DataFrame."
        )

    # Extract wavelength and intensity data
    wave, intensity = df["wave"], df["shot_avg"]

    plt.figure(figsize=(10, 6))

    # Plot segments with gaps smaller than max_gap_threshold
    start_idx = 0
    for i in range(1, len(wave)):
        if wave.iloc[i] - wave.iloc[i - 1] > max_gap_threshold:
            sns.lineplot(
                x=wave.iloc[start_idx:i], y=intensity.iloc[start_idx:i], color="blue"
            )
            start_idx = i
    sns.lineplot(
        x=wave.iloc[start_idx:], y=intensity.iloc[start_idx:], color="blue"
    )  # plot the last segment

    # Colors for the spectral ranges
    spectral_colors = {
        "UV": "#BA55D3",  # Medium Orchid
        "VIS": "#00FF00",  # Bright Green
        "VNIR": "#8B0000",  # Deep Red
    }

    # Overlaying spectral ranges
    for range_name, (start, end) in spectrometer_wavelength_ranges.items():
        plt.axvspan(
            start, end, alpha=0.3, color=spectral_colors[range_name], label=range_name
        )

    for i, (start, end) in enumerate(masks):
        plt.axvspan(
            start, end, alpha=0.1, color="black", label="Mask" if i == 0 else None
        )

    # Labeling and title
    plt.title("Spectral Analysis of Sample: " + sample_name)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")

    plt.legend(title="Spectral Ranges")
    plt.show()

    if fig_name != "":
        plt.savefig(fig_name, bbox_inches="tight")


def plot_outliers_for_run(mlflow_run_id: str, figure_name=""):
    mlflow.set_tracking_uri("http://localhost:5000")
    client = mlflow.MlflowClient()

    local_dir = Path(f"images/{mlflow_run_id}")
    local_dir.mkdir(parents=True, exist_ok=True)

    artifact_info = client.list_artifacts(mlflow_run_id)
    image_files = [
        artifact.path for artifact in artifact_info if artifact.path.endswith(".png")
    ]
    image_files_sorted = sorted(
        image_files, key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    for image_file in image_files_sorted:
        client.download_artifacts(mlflow_run_id, image_file, str(local_dir))

    num_images = len(image_files_sorted)
    total_height = 5  # Adjust total height as needed
    fig_width = total_height * num_images  # Aesthetic ratio

    # Create a figure with subplots in a horizontal line
    fig, axs = plt.subplots(1, num_images, figsize=(fig_width, total_height))

    # Display each image in its subplot
    for i, img_file in enumerate(image_files_sorted):
        local_file_path = local_dir / img_file
        img = mpimg.imread(local_file_path)  # Load image
        ax = axs[i]
        ax.imshow(img)  # Display image
        ax.axis("off")  # Hide axis
        ax.set_title(f"Iteration {i+1}", fontsize=12)  # Set title with iteration number

    # Adjust the layout to prevent overlap and ensure everything fits nicely
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    # Save the figure if needed
    if figure_name != "":
        save_file_path = local_dir / figure_name
        plt.savefig(save_file_path, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    # Example
    plot_outliers_for_run(
        "16b0b228ada94577b15b658890ea1dc4", "outlier_removal_over_time"
    )

    # Example
    plot_spectra(
        "cadillac",
        "data/data/calib/calib_2015/1600mm/pls",
        fig_name="cadillac_spectra.png",
    )
