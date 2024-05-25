import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lib.full_flow_dataloader import load_full_flow_data
from lib.cross_validation import custom_kfold_cross_validation_new, stratified_group_kfold_split
from lib.reproduction import major_oxides

# Set page layout to wide
st.set_page_config(layout="centered")

# Load data
st.title("Custom K-Fold Cross Validation")

@st.cache_data
def load_data():
    train_processed, test_processed = load_full_flow_data()
    data = pd.concat([train_processed, test_processed])
    return train_processed, test_processed, data

@st.cache_data
def get_folds_custom(data, cv_method, group_by, target, random_state, extreme_percentage):
    if cv_method == "Sorted Group K-Fold":
        folds_custom, train_full, test_full = custom_kfold_cross_validation_new(
            data=data, k=5, group_by=group_by, target=target, random_state=random_state, percentile=extreme_percentage
        )
    else:
        folds_custom = stratified_group_kfold_split(
            data, group_by=group_by, target=target, num_bins=5, n_splits=5, random_state=random_state
        )
        train_full = data
        test_full = data
    return folds_custom, train_full, test_full

@st.cache_data
def plot_distribution(folds_custom, test_full, target):
    fold_data_custom = []
    for i, (_, test) in enumerate(folds_custom):
        fold_target = test[target]
        for value in fold_target:
            fold_data_custom.append((i + 1, value))

    test_full_target = test_full[target]
    for value in test_full_target:
        fold_data_custom.append(("Test Full", value))

    fold_df_custom = pd.DataFrame(fold_data_custom, columns=["Fold", target])

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x="Fold", y=target, data=fold_df_custom, ax=ax)
    ax.set_title(f"Custom Stratified Group K-Fold: Distribution of {target} in Each Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel(target)
    return fig

@st.cache_data
def plot_histogram(folds_custom, test_full_target, target):
    fig, ax = plt.subplots(figsize=(15, 10))
    for i, (train_full, test) in enumerate(folds_custom):
        fold_target = test[target]
        sns.histplot(
            fold_target, kde=True, bins=30, label=f"Fold {i+1} ({len(train_full)} / {len(test)})", alpha=0.5, ax=ax
        )

    sns.histplot(
        test_full_target, kde=True, bins=30, label=f"Test Full ({len(test_full)})", alpha=0.5, color="black", ax=ax
    )
    ax.set_title(f"Custom Stratified Group K-Fold: Histogram and KDE of {target} Distribution in Each Fold")
    ax.set_xlabel(target)
    ax.set_ylabel("Count")
    ax.legend()
    return fig

@st.cache_data
def plot_grid_histogram(folds_custom, test_full_target, target):
    num_folds = len(folds_custom)
    num_cols = 3
    num_rows = (num_folds + num_cols - 1) // num_cols

    fig = plt.figure(figsize=(15, 5 * num_rows))

    for i, (train_full, test) in enumerate(folds_custom):
        fold_target = test[target]
        plt.subplot(num_rows, num_cols, i + 1)
        sns.histplot(fold_target, kde=True, bins=30, label=f"Fold {i+1} ({len(train_full)} / {len(test)})", alpha=0.5)
        plt.title(f"Fold {i+1}")
        plt.xlabel(target)
        plt.ylabel("Count")
        plt.legend()

    # Add test_full data in a separate subplot
    plt.subplot(num_rows, num_cols, num_folds + 1)
    sns.histplot(test_full_target, kde=True, bins=30, label=f"Test Full ({len(test_full)})", alpha=0.5, color="black")
    plt.title("Test Full")
    plt.xlabel(target)
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()
    return fig

train_processed, test_processed, data = load_data()
group_by = "Sample Name"
n_splits = 5
random_state = 42

# User selects target
target = st.selectbox("Select target oxide:", major_oxides)

# User selects cross-validation method
cv_method = st.selectbox("Select cross-validation method:", ["Sorted Group K-Fold", "Stratified Group K-Fold"])

# User selects extreme percentage
extreme_percentage = st.slider("Select extreme percentage:", min_value=0.01, max_value=0.25, value=0.05, step=0.01)

# Add links based on selection
if cv_method == "Sorted Group K-Fold":
    st.markdown(
        "[View Sorted K-Fold Code](https://github.com/chhoumann/thesis-chemcam/blob/58822cb4a89426359458eee9c5bb6a0d4ad2af6f/baseline/lib/cross_validation.py#L13)"
    )
else:
    st.markdown(
        "[View Stratified Group K-Fold Code](https://github.com/chhoumann/thesis-chemcam/blob/58822cb4a89426359458eee9c5bb6a0d4ad2af6f/baseline/lib/cross_validation.py#L178)"
    )

folds_custom, train_full, test_full = get_folds_custom(data, cv_method, group_by, target, random_state, extreme_percentage)

# Distribution Plot
st.subheader("Distribution Plot")
fig = plot_distribution(folds_custom, test_full, target)
st.pyplot(fig)

# Histogram Plot
st.subheader("Histogram Plot")
fig = plot_histogram(folds_custom, test_full[target], target)
st.pyplot(fig)

# Grid Layout Histogram Plot
st.subheader("Grid Layout Histogram Plot")
fig = plot_grid_histogram(folds_custom, test_full[target], target)
st.pyplot(fig)
