"""
visualization.py
Functions for data and results visualization
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Styling
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

CLASS_NAMES = ["No Diabetes", "Prediabetes", "Diabetes"]
CLASS_COLORS = [
    "#4575b4",
    "#fee090",
    "#d73027",
]  # Blue → Yellow → Red (ColorBrewer RdYlBu)


def plot_target_distribution(
    y: pd.Series, title: str = "Target Variable Distribution"
) -> None:
    """
    Visualizes the distribution of the target variable.

    Args:
        y: Target variable
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute counts
    y.value_counts().sort_index().plot(kind="bar", ax=axes[0], color=CLASS_COLORS)
    axes[0].set_title(f"{title} – Absolute Counts")
    axes[0].set_xlabel("Diabetes Status")
    axes[0].set_ylabel("Count")
    axes[0].set_xticklabels(CLASS_NAMES, rotation=0)

    # Percentage
    y.value_counts(normalize=True).sort_index().plot(
        kind="bar", ax=axes[1], color=CLASS_COLORS
    )
    axes[1].set_title(f"{title} – Percentage")
    axes[1].set_xlabel("Diabetes Status")
    axes[1].set_ylabel("Share")
    axes[1].set_xticklabels(CLASS_NAMES, rotation=0)
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y * 100:.1f}%")  # pyright: ignore[reportPrivateImportUsage]
    )

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, figsize: tuple = (16, 14)) -> None:
    """
    Visualizes the correlation matrix.

    Args:
        df: DataFrame with features
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    corr = df.corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7},
    )
    plt.title("Correlation Matrix – All Features")
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(
    df: pd.DataFrame, features: list, target: str = "Diabetes_012"
) -> None:
    """
    Visualizes feature distributions grouped by target variable.

    Args:
        df: DataFrame with data
        features: List of features to visualize
        target: Name of the target column
    """
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        df.groupby(target)[feature].mean().plot(
            kind="bar", ax=axes[idx], color=CLASS_COLORS
        )
        axes[idx].set_title(f"{feature} by Diabetes Status")
        axes[idx].set_xlabel("Diabetes Status")
        axes[idx].set_ylabel(f"Mean {feature}")
        axes[idx].set_xticklabels(["No DM", "Pre", "Diabetes"], rotation=0)

    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix") -> None:
    """
    Visualizes the confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()
