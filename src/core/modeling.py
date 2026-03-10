"""
modeling.py
Model training and evaluation functions for multi-class diabetes classification

This module contains functions for:
- Training different machine learning models
- Evaluating model performance
- Visualizing results (confusion matrices, classification reports, ROC curves)
- Comparing multiple models
- Saving/loading trained models
"""

from typing import Dict

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================


def train_logistic_regression(features, target, **kwargs):
    """
    Trains Logistic Regression model for multi-class classification.

    Args:
        features: Training features
        target: Training target
        **kwargs: Additional parameters for LogisticRegression

    Returns:
        Trained Logistic Regression model
    """
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        **kwargs,
    )
    model.fit(features, target)
    print("Logistic Regression trained")
    return model


def train_random_forest(features, target, **kwargs):
    """
    Trains Random Forest model for multi-class classification.

    Args:
        features: Training features
        target: Training target
        **kwargs: Additional parameters for RandomForestClassifier

    Returns:
        Trained Random Forest model
    """
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, **kwargs
    )
    model.fit(features, target)
    print("Random Forest trained")
    return model


def train_xgboost(features, target, **kwargs):
    """
    Trains XGBoost model for multi-class classification.

    Args:
        features: Training features
        target: Training target
        **kwargs: Additional parameters for XGBClassifier

    Returns:
        Trained XGBoost model
    """
    model = XGBClassifier(
        objective="multi:softmax", num_class=3, random_state=42, n_jobs=-1, **kwargs
    )
    model.fit(features, target)
    print("XGBoost trained")
    return model


def train_linear_svm(features, target, **kwargs):
    """
    Trains LinearSVM model (much faster than RBF-SVM for large datasets).

    LinearSVC is wrapped with CalibratedClassifierCV to enable
    probability estimates needed for ROC curves.

    Args:
        features: Training features
        target: Training target
        **kwargs: Additional parameters for LinearSVC

    Returns:
        Trained LinearSVM model (calibrated for probability estimates)
    """
    linear_svm = LinearSVC(max_iter=1000, random_state=42, **kwargs)
    model = CalibratedClassifierCV(linear_svm, cv=3)
    model.fit(features, target)
    print("LinearSVM trained")
    return model


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================


def evaluate_model(model, features, target, model_name: str = "Model") -> Dict:
    """
    Evaluates a trained model and returns metrics dictionary.

    Keys in the returned dictionary match the expected input of compare_models().

    Args:
        model: Trained model
        features: Test features
        target: Test target
        model_name: Name of the model for display

    Returns:
        Dictionary with evaluation metrics:
            - model_name, Accuracy, Precision, Recall,
              F1 (Macro), F1 (Weighted), ROC-AUC,
              y_pred, y_pred_proba

    Medical Context:
        Recall is highlighted as the PRIMARY METRIC because missing
        a diabetes case (false negative) is more critical than a
        false alarm (false positive) in medical screening.
    """
    predictions = model.predict(features)
    target_proba = model.predict_proba(features)

    # Calculate metrics
    accuracy = accuracy_score(target, predictions)
    precision = precision_score(
        target, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(target, predictions, average="weighted", zero_division=0)
    f1_macro = f1_score(target, predictions, average="macro")
    f1_weighted = f1_score(target, predictions, average="weighted")

    # ROC-AUC (One-vs-Rest, macro average)
    target_bin = label_binarize(target, classes=[0, 1, 2])
    roc_auc = roc_auc_score(
        target_bin, target_proba, average="macro", multi_class="ovr"
    )

    report_dict = classification_report(target, predictions, output_dict=True)

    # Print evaluation summary
    print(f"\n{'=' * 60}")
    print(f"{model_name} - EVALUATION")
    print(f"{'=' * 60}")
    print(f"  Accuracy:            {accuracy:.4f}")
    print(f"  Precision:           {precision:.4f}")
    print(f"  Recall (Macro):      {recall:.4f}")
    print(f"  Recall (Diabetes):   {report_dict[2]['recall']:.4f} <- PRIMARY METRIC") # pyright: ignore[reportArgumentType]
    print(f"  F1-Score (Macro):    {f1_macro:.4f}")
    print(f"  F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"  ROC-AUC (Macro):     {roc_auc:.4f}")
    print(f"\nClassification Report:\n{classification_report(target, predictions)}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(target, predictions)}")
    print(f"{'=' * 60}\n")

    return {
        "model_name": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Recall_Diabetes": report_dict[2]["recall"], # type: ignore
        "F1 (Macro)": f1_macro,
        "F1 (Weighted)": f1_weighted,
        "ROC-AUC": roc_auc,
        "y_pred": predictions,
        "y_pred_proba": target_proba,
        "Report": report_dict,
    }


def compare_models(results: dict) -> pd.DataFrame:
    """
    Compares multiple models and returns comparison DataFrame.

    This function aggregates evaluation metrics from multiple models
    into a single DataFrame for easy comparison. Models are sorted
    by Recall_Diabetes (our primary optimization metric).

    Args:
        results: Dictionary of {model_name: metrics_dict}

    Returns:
        DataFrame with model comparison (sorted by Recall descending)

    Medical Context:
        Models are ranked by RECALL_DIABETES (Class 2) because in medical
        screening, missing an actual diabetes case is the most critical error.
        Macro Recall serves as the secondary metric.
    """
    # Convert results dictionary to DataFrame
    comparison = pd.DataFrame(results).T

    # Sort by Recall_Diabetes (descending) - our primary metric
    comparison = comparison.sort_values("Recall_Diabetes", ascending=False)

    return comparison


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_confusion_matrix(model, features, target, model_name: str, ax=None):
    """
    Plots confusion matrix for a trained model.

    Args:
        model: Trained model
        features: Test features
        target: Test target
        model_name: Name of the model for the title
        ax: Matplotlib axis object (optional)

    Returns:
        Matplotlib axis object
    """
    predictions = model.predict(features)
    cm = confusion_matrix(target, predictions)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No Diabetes", "Prediabetes", "Diabetes"]
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    return ax


def plot_classification_report(model, features, target, model_name: str, ax=None):
    """
    Plots classification report as a heatmap.

    Args:
        model: Trained model
        features: Test features
        target: Test target
        model_name: Name of the model for the title
        ax: Matplotlib axis object (optional)

    Returns:
        Matplotlib axis object
    """
    predictions = model.predict(features)
    report = classification_report(
        target,
        predictions,
        target_names=["No Diabetes", "Prediabetes", "Diabetes"],
        output_dict=True,
    )

    classes = ["No Diabetes", "Prediabetes", "Diabetes"]
    df_report = pd.DataFrame(
        [
            [report[c]["precision"], report[c]["recall"], report[c]["f1-score"]]  # pyright: ignore[reportArgumentType]
            for c in classes
        ],  # pyright: ignore[reportArgumentType]
        index=classes,
        columns=["Precision", "Recall", "F1-Score"],
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        df_report,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Score"},
        ax=ax,
    )
    ax.set_title(
        f"Classification Report - {model_name}", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Metrics", fontsize=11)
    ax.set_ylabel("Classes", fontsize=11)

    return ax


def plot_roc_curves(model, features, target, model_name: str, ax=None):
    """
    Plots ROC curves for multi-class classification (One-vs-Rest).

    Args:
        model: Trained model
        features: Test features
        target: Test target
        model_name: Name of the model for the title
        ax: Matplotlib axis object (optional)

    Returns:
        Matplotlib axis object
    """
    target_proba = model.predict_proba(features)
    target_bin = label_binarize(target, classes=[0, 1, 2])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["blue", "orange", "green"]
    class_names = ["No Diabetes", "Prediabetes", "Diabetes"]

    for i, (color, class_name) in enumerate(zip(colors, class_names)):
        fpr, tpr, _ = roc_curve(target_bin[:, i], target_proba[:, i])  # pyright: ignore[reportIndexIssue]
        roc_auc_val = auc(fpr, tpr)
        ax.plot(
            fpr, tpr, color=color, lw=2, label=f"{class_name} (AUC = {roc_auc_val:.3f})"
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    ax.set_xlim([0.0, 1.0])  # pyright: ignore[reportArgumentType]
    ax.set_ylim([0.0, 1.05])  # pyright: ignore[reportArgumentType]
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title(f"ROC Curves - {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    return ax


# =============================================================================
# MODEL PERSISTENCE FUNCTIONS
# =============================================================================


def save_model(model, filepath: str) -> None:
    """
    Saves a trained model to disk.

    Args:
        model: Trained model
        filepath: Target file path (e.g. '../models/rf_model.pkl')
    """
    joblib.dump(model, filepath)
    print(f"Model saved: {filepath}")


def load_model(filepath: str):
    """
    Loads a saved model from disk.

    Args:
        filepath: Path to the saved model file

    Returns:
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"Model loaded: {filepath}")
    return model


def get_feature_importance(model, feature_names, top_n: int = 20) -> pd.DataFrame:
    """
    Extracts feature importance from tree-based models.

    Args:
        model: Trained model (Random Forest or XGBoost)
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with feature importances (sorted descending),
        or None if model does not support feature_importances_
    """
    if hasattr(model, "feature_importances_"):
        importance_df = (
            pd.DataFrame(
                {"Feature": feature_names, "Importance": model.feature_importances_}
            )
            .sort_values("Importance", ascending=False)
            .head(top_n)
        )

        return importance_df
    else:
        print("Model does not have feature_importances_")
        return None  # pyright: ignore[reportReturnType]
