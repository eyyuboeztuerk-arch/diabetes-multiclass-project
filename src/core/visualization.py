"""
visualization.py
Functions for data and results visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

CLASS_NAMES  = ['No Diabetes', 'Prediabetes', 'Diabetes']
CLASS_COLORS = ['#4575b4', '#fee090', '#d73027']  # Blue → Yellow → Red (ColorBrewer RdYlBu)


def plot_target_distribution(y: pd.Series, title: str = "Target Variable Distribution") -> None:
    """
    Visualizes the distribution of the target variable.

    Args:
        y: Target variable
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute counts
    y.value_counts().sort_index().plot(
        kind='bar', ax=axes[0], color=CLASS_COLORS
    )
    axes[0].set_title(f"{title} – Absolute Counts")
    axes[0].set_xlabel("Diabetes Status")
    axes[0].set_ylabel("Count")
    axes[0].set_xticklabels(CLASS_NAMES, rotation=0)

    # Percentage
    y.value_counts(normalize=True).sort_index().plot(
        kind='bar', ax=axes[1], color=CLASS_COLORS
    )
    axes[1].set_title(f"{title} – Percentage")
    axes[1].set_xlabel("Diabetes Status")
    axes[1].set_ylabel("Share")
    axes[1].set_xticklabels(CLASS_NAMES, rotation=0)
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y * 100:.1f}%') # pyright: ignore[reportPrivateImportUsage]
    )

    plt.tight_layout()
    plt.show()

    


