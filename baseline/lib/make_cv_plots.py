import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lib.cross_validation import custom_kfold_cross_validation_new
from lib.full_flow_dataloader import load_full_flow_data
from lib.reproduction import major_oxides

def create_plots_for_oxide(target):
    train_processed, test_processed = load_full_flow_data()
    data = pd.concat([train_processed, test_processed])
    group_by = 'Sample Name'
    n_splits = 5
    random_state = 42

    folds_custom, train_full, test_full = custom_kfold_cross_validation_new(
        data=data, k=n_splits, group_by=group_by, target=target, random_state=random_state, remove_fold_column=False
    )

    full_data_with_folds = pd.concat([train_full, test_full], ignore_index=True)

    def visualize_combined(data, full_data_with_folds, group_by, target, save_path):
        bin_edges = np.histogram_bin_edges(data[target], bins=30)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(data[target], bins=bin_edges, kde=True)
        plt.title(f"Original Data: Distribution of {target}")

        plt.subplot(1, 2, 2)
        hist_plot = sns.histplot(
            data=full_data_with_folds, x=target, hue="fold", bins=bin_edges, multiple="stack", palette="Set1"
        )
        plt.title(f"Data with Folds Assigned: Histogram of {target} by Fold")

        new_labels = ["Fold 0 (Train)", "Fold 1 (Train)", "Fold 2 (Train)", "Fold 3 (Train)", "Fold 4 (Test)"]
        for t, l in zip(hist_plot.legend_.texts, new_labels):
            t.set_text(l)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'original_and_post_fold.png'))
        plt.close()

    save_path = f"../report_thesis/src/images/{target}"
    os.makedirs(save_path, exist_ok=True)

    visualize_combined(data, full_data_with_folds, group_by, target, save_path)

    fold_data_custom = []

    for i, (_, test) in enumerate(folds_custom):
        fold_target = test[target]
        fold_std = fold_target.std()
        fold_mean = fold_target.mean()
        for value in fold_target:
            fold_data_custom.append((f"Fold {i+1}", value, fold_std, fold_mean))

    train_full_target = train_full[target]
    train_full_std = train_full_target.std()
    train_full_mean = train_full_target.mean()
    for value in train_full_target:
        fold_data_custom.append(('Train\n(1-4 combined)', value, train_full_std, train_full_mean))

    test_full_target = test_full[target]
    test_full_std = test_full_target.std()
    test_full_mean = test_full_target.mean()
    for value in test_full_target:
        fold_data_custom.append(('Test', value, test_full_std, test_full_mean))

    data_full_target = data[target]
    data_full_std = data_full_target.std()
    data_full_mean = data_full_target.mean()
    for value in data_full_target:
        fold_data_custom.append(('Full', value, data_full_std, data_full_mean))

    fold_df_custom = pd.DataFrame(fold_data_custom, columns=['Fold', target, 'Standard Deviation', 'Mean'])

    plt.figure(figsize=(15, 10))
    ax = sns.boxplot(x='Fold', y=target, data=fold_df_custom)
    plt.title(f'Distribution of {target} in data partitions')
    plt.xlabel('Partition')
    plt.ylabel(f"{target} wt. %")

    annotation_y = fold_df_custom[target].max() * 1.05
    for i, fold in enumerate(fold_df_custom['Fold'].unique()):
        fold_data = fold_df_custom[fold_df_custom['Fold'] == fold]
        std = fold_data['Standard Deviation'].iloc[0]
        mean = fold_data['Mean'].iloc[0]
        ax.annotate(f'std: {std:.2f}\nmean: {mean:.2f}', 
                    xy=(i, annotation_y), 
                    xytext=(0, 0), 
                    textcoords='offset points', 
                    ha='center', 
                    fontsize=10, 
                    color='black', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.ylim(top=annotation_y * 1.1)
    plt.savefig(os.path.join(save_path, 'distribution_plot.png'))
    plt.close()

    plt.figure(figsize=(15, 10))

    for i, (train_full, validation) in enumerate(folds_custom):
        fold_target = validation[target]
        sns.histplot(fold_target, kde=True, bins=30, label=f'Fold {i+1} ({len(train_full)} / {len(validation)})', alpha=0.5)

    sns.histplot(test_full_target, kde=True, bins=30, label=f'Test Full ({len(test_full)})', alpha=0.5, color='black')

    plt.title(f'Histogram and KDE of {target} Distribution in Each Fold')
    plt.xlabel(f"{target} wt. %")
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'histogram_kde_plot.png'))
    plt.close()

    num_folds = len(folds_custom)
    num_cols = 3
    num_rows = (num_folds + num_cols - 1) // num_cols

    all_targets = [test[target] for _, test in folds_custom] + [test_full_target]
    x_min = min([min(fold) for fold in all_targets])
    x_max = max([max(fold) for fold in all_targets])

    y_max = 0
    for fold in all_targets:
        hist, bins = np.histogram(fold, bins=30)
        y_max = max(y_max, max(hist))
        y_max += 1

    plt.figure(figsize=(15, 5 * num_rows))

    for i, (train_full, validation) in enumerate(folds_custom):
        fold_target = validation[target]
        plt.subplot(num_rows, num_cols, i + 1)
        sns.histplot(fold_target, kde=True, bins=30, label=f'Fold {i+1} ({len(train_full)} / {len(validation)})')
        plt.title(f'Fold {i+1}')
        plt.xlabel(f"{target} wt. %")
        plt.ylabel('Count')
        plt.xlim(x_min, x_max)
        plt.ylim(0, y_max)
        plt.legend()

    plt.subplot(num_rows, num_cols, num_folds + 1)
    sns.histplot(test_full_target, kde=True, bins=30, label=f'Test Full ({len(test_full)})', color='black')
    plt.title('Test Full')
    plt.xlabel(f"{target} wt. %")
    plt.ylabel('Count')
    plt.xlim(x_min, x_max)
    plt.ylim(0, y_max)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'histogram_grid_plot.png'))
    plt.close()

for oxide in major_oxides:
    create_plots_for_oxide(oxide)
