from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mlflow
import pandas as pd


def plot_spectra(df: pd.DataFrame):
    pass


def plot_outliers_for_run(mlflow_run_id: str):
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
    save_file_path = local_dir / "outlier_removal_over_time.png"
    plt.savefig(save_file_path, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    # Example usage
    plot_outliers_for_run("16b0b228ada94577b15b658890ea1dc4")
