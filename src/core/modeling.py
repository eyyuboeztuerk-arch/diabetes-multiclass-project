"""
Modeling Module
Functions for training, evaluating, and visualizing machine learning models

This module contains all functions needed for model training and evaluation.
All models are optimized for HIGH RECALL to minimize false negatives in
diabetes screening (medical context: missing a diabetes case is more serious
than a false alarm).
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier


def train_logistic_regression(
    features: pd.DataFrame, target: pd.Series
) -> LogisticRegression:
    """
    Trains Logistic Regression model (optimized for recall)

    Logistic Regression is a linear model that predicts class probabilities.
    It's interpretable, fast, and works well as a baseline model.

    Args:
        features: Training features
        target: Training target

    Returns:
        Trained Logistic Regression model

    Model Configuration (Recall-Optimized):
        - class_weight='balanced': Gives more weight to minority classes
          (Prediabetes, Diabetes) to improve their recall
        - max_iter=1000: Ensures convergence for complex datasets
        - multi_class='multinomial': Proper multi-class classification
        - solver='lbfgs': Efficient solver for multi-class problems

    Medical Context:
        class_weight='balanced' is crucial for medical screening because
        it helps the model prioritize detecting minority classes (diabetes
        cases) even at the cost of some false positives.
    """
    # Initialize Logistic Regression with recall-optimized parameters
    model = LogisticRegression(
        class_weight="balanced",  # Prioritize minority classes (HIGH RECALL)
        max_iter=1000,  # Ensure convergence
        multi_class="multinomial",  # Multi-class classification
        solver="lbfgs",  # Efficient solver
        random_state=42,  # Reproducibility
    )

    # Train model on training data
    model.fit(features, target)

    return model


def train_random_forest(
    features: pd.DataFrame, target: pd.Series
) -> RandomForestClassifier:
    """
    Trains Random Forest model (optimized for recall)

    Random Forest is an ensemble of decision trees. It's robust to
    overfitting, handles non-linear relationships well, and provides
    feature importance scores.

    Args:
        features: Training features
        target: Training target

    Returns:
        Trained Random Forest model

    Model Configuration (Recall-Optimized):
        - n_estimators=100: Number of trees (more trees = more stable)
        - class_weight='balanced': Prioritize minority classes
        - max_depth=10: Limit tree depth to prevent overfitting
        - min_samples_split=10: Minimum samples to split a node
        - min_samples_leaf=5: Minimum samples in leaf node

    Medical Context:
        Random Forest with balanced class weights excels at detecting
        minority classes (diabetes cases). The ensemble approach reduces
        variance and improves recall stability.
    """
    # Initialize Random Forest with recall-optimized parameters
    model = RandomForestClassifier(
        n_estimators=100,  # 100 trees in the forest
        class_weight="balanced",  # Prioritize minority classes (HIGH RECALL)
        max_depth=10,  # Prevent overfitting
        min_samples_split=10,  # Minimum samples to split
        min_samples_leaf=5,  # Minimum samples in leaf
        random_state=42,  # Reproducibility
    )

    # Train model on training data
    model.fit(features, target)

    return model


def train_xgboost(features: pd.DataFrame, target: pd.Series) -> XGBClassifier:
    """
    Trains XGBoost model (optimized for recall)

    XGBoost is a gradient boosting algorithm that builds trees sequentially,
    each correcting errors of previous trees. It often achieves the best
    performance on structured/tabular data.

    Args:
        features: Training features
        target: Training target

    Returns:
        Trained XGBoost model

    Model Configuration (Recall-Optimized):
        - n_estimators=100: Number of boosting rounds
        - learning_rate=0.1: Step size for each boosting iteration
        - max_depth=5: Maximum tree depth
        - scale_pos_weight: Calculated to balance classes
        - eval_metric='mlogloss': Multi-class log loss

    Medical Context:
        XGBoost with scale_pos_weight is highly effective for imbalanced
        medical datasets. It learns to prioritize minority classes
        (diabetes cases) through weighted loss function, maximizing recall.
    """
    # Calculate scale_pos_weight for class balancing
    # This gives more weight to minority classes in the loss function
    class_counts = target.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1

    # Initialize XGBoost with recall-optimized parameters
    model = XGBClassifier(
        n_estimators=100,  # 100 boosting rounds
        learning_rate=0.1,  # Step size
        max_depth=5,  # Tree depth
        scale_pos_weight=scale_pos_weight,  # Balance classes (HIGH RECALL)
        eval_metric="mlogloss",  # Multi-class log loss
        use_label_encoder=False,  # Disable deprecated encoder
        random_state=42,  # Reproducibility
    )

    # Train model on training data
    model.fit(features, target)

    return model


def train_svm(features: pd.DataFrame, target: pd.Series) -> SVC:
    """
    Trains Support Vector Machine model (optimized for recall)

    SVM finds optimal hyperplanes to separate classes. It works well
    for high-dimensional data and can capture complex non-linear
    decision boundaries with the RBF kernel.

    Args:
        features: Training features
        target: Training target

    Returns:
        Trained SVM model

    Model Configuration (Recall-Optimized):
        - kernel='rbf': Radial Basis Function for non-linear boundaries
        - class_weight='balanced': Prioritize minority classes
        - C=1.0: Regularization parameter
        - probability=True: Enable probability estimates for ROC curves

    Medical Context:
        SVM with balanced class weights and RBF kernel can capture
        complex patterns in medical data. The balanced weights ensure
        high recall for minority classes (diabetes cases).

    Note:
        SVM can be slow on large datasets. Consider using a subset
        for initial experiments if training time is excessive.
    """
    # Initialize SVM with recall-optimized parameters
    model = SVC(
        kernel="rbf",  # Radial Basis Function kernel
        class_weight="balanced",  # Prioritize minority classes (HIGH RECALL)
        C=1.0,  # Regularization parameter
        probability=True,  # Enable probability estimates
        random_state=42,  # Reproducibility
    )

    # Train model on training data
    model.fit(features, target)

    return model


def evaluate_model(model, features: pd.DataFrame, target: pd.Series) -> dict:
    """
    Evaluates model performance on test data

    This function calculates key classification metrics to assess
    model performance. For medical screening, recall is the most
    important metric (minimize false negatives).

    Args:
        model: Trained model
        features: Test features
        target: Test target (true labels)

    Returns:
        Dictionary with evaluation metrics

    Metrics calculated:
        - Accuracy: Overall correctness (TP+TN)/(TP+TN+FP+FN)
        - Precision: Positive predictive value TP/(TP+FP)
        - Recall: Sensitivity, true positive rate TP/(TP+FN)
        - F1-Score: Harmonic mean of precision and recall

    Medical Context:
        - Recall (Sensitivity): Most critical for screening
          High recall = Few missed diabetes cases (low false negatives)
        - Precision: Positive predictive value
          Lower precision acceptable (false alarms can be verified)
        - F1-Score: Balance between precision and recall
        - Accuracy: Can be misleading with imbalanced classes
    """
    # Make predictions on test data
    predictions = model.predict(features)

    # Calculate evaluation metrics
    # average='weighted': Accounts for class imbalance
    metrics = {
        "Accuracy": accuracy_score(target, predictions),
        "Precision": precision_score(
            target, predictions, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            target, predictions, average="weighted", zero_division=0
        ),  # MOST IMPORTANT
        "F1-Score": f1_score(target, predictions, average="weighted", zero_division=0),
    }

    return metrics


def plot_confusion_matrix(
    model, features: pd.DataFrame, target: pd.Series, model_name: str, ax=None
):
    """
    Plots confusion matrix for a trained model

    Args:
        model: Trained model
        features: Test features
        target: Test target
        model_name: Name of the model for the title
        ax: Matplotlib axis object (optional)
    """
    # Make predictions
    predictions = model.predict(features)

    # Calculate confusion matrix
    cm = confusion_matrix(target, predictions)

    # Create display
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No Diabetes", "Prediabetes", "Diabetes"]
    )

    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    return ax


def plot_classification_report(
    model, features: pd.DataFrame, target: pd.Series, model_name: str, ax=None
):
    """
    Plots classification report as a heatmap

    Args:
        model: Trained model
        features: Test features
        target: Test target
        model_name: Name of the model for the title
        ax: Matplotlib axis object (optional)
    """
    # Make predictions
    predictions = model.predict(features)

    # Generate classification report as dictionary
    report = classification_report(
        target,
        predictions,
        target_names=["No Diabetes", "Prediabetes", "Diabetes"],
        output_dict=True,
    )

    # Extract metrics for each class
    classes = ["No Diabetes", "Prediabetes", "Diabetes"]
    metrics = ["precision", "recall", "f1-score"]

    # Create matrix for heatmap
    data = []
    for cls in classes:
        data.append([report[cls][metric] for metric in metrics])  # pyright: ignore[reportArgumentType]

    # Create DataFrame
    df_report = pd.DataFrame(
        data, index=classes, columns=["Precision", "Recall", "F1-Score"]
    )

    # Create heatmap
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        df_report,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Score"},
        ax=ax,
    )

    ax.set_title(
        f"Classification Report - {model_name}", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Classes", fontsize=12)

    return ax


def plot_roc_curves(
    model, features: pd.DataFrame, target: pd.Series, model_name: str, ax=None
):
    """
    Plots ROC curves for multi-class classification

    Args:
        model: Trained model
        features: Test features
        target: Test target
        model_name: Name of the model for the title
        ax: Matplotlib axis object (optional)
    """
    # Get probability predictions
    if hasattr(model, "predict_proba"):
        target_proba = model.predict_proba(features)
    else:
        # For models without predict_proba (shouldn't happen with our models)
        target_proba = model.decision_function(features)

    # Binarize the target for multi-class ROC
    target_bin = label_binarize(target, classes=[0, 1, 2])
    n_classes = target_bin.shape[1]

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot ROC curve for each class
    colors = ["blue", "orange", "green"]
    class_names = ["No Diabetes", "Prediabetes", "Diabetes"]

    for i, color, class_name in zip(range(n_classes), colors, class_names):
        fpr, tpr, _ = roc_curve(target_bin[:, i], target_proba[:, i])  # pyright: ignore[reportIndexIssue]
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr, tpr, color=color, lw=2, label=f"{class_name} (AUC = {roc_auc:.3f})"
        )

    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")

    ax.set_xlim([0.0, 1.0])  # pyright: ignore[reportArgumentType]
    ax.set_ylim([0.0, 1.05])  # pyright: ignore[reportArgumentType]
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title(f"ROC Curves - {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    return ax


def compare_models(results: dict) -> pd.DataFrame:
    """
    Compares multiple models and returns comparison DataFrame

    This function aggregates evaluation metrics from multiple models
    into a single DataFrame for easy comparison. Models are sorted
    by recall (our primary optimization metric).

    Args:
        results: Dictionary of {model_name: metrics_dict}

    Returns:
        DataFrame with model comparison (sorted by recall)

    Medical Context:
        Models are ranked by RECALL because in medical screening,
        minimizing false negatives (missed diabetes cases) is the
        top priority. A model with highest recall is preferred even
        if it has slightly lower precision or accuracy.
    """
    # Convert results dictionary to DataFrame
    comparison = pd.DataFrame(results).T

    # Sort by Recall (descending) - our primary metric
    comparison = comparison.sort_values("Recall", ascending=False)

    return comparison


def save_model(model, filepath: str):
    """
    Saves trained model to disk using joblib

    This function serializes the trained model to disk for later use.
    Saved models can be loaded for:
    - Production deployment
    - Making predictions on new data
    - Sharing with collaborators
    - Reproducibility

    Args:
        model: Trained model to save
        filepath: Path to save model (should end with .pkl)

    Example:
        >>> save_model(best_model, '../outputs/models/xgboost_smote_recall_optimized.pkl')

    Note:
        joblib is preferred over pickle for scikit-learn models because
        it's more efficient for large numpy arrays (common in ML models).
    """
    # Save model using joblib
    joblib.dump(model, filepath)

    print(f"Model saved: {filepath}")


def load_model(filepath: str):
    """
    Loads trained model from disk

    This function deserializes a saved model from disk for use.

    Args:
        filepath: Path to saved model (.pkl file)

    Returns:
        Loaded model

    Example:
        >>> model = load_model('../outputs/models/xgboost_smote_recall_optimized.pkl')
        >>> predictions = model.predict(new_data)
    """
    # Load model using joblib
    model = joblib.load(filepath)

    print(f"Model loaded: {filepath}")

    return model


def train_linear_svm(features: pd.DataFrame, target: pd.Series):
    """
    Trains LinearSVM model (optimized for recall, much faster than RBF-SVM)

    LinearSVC is used instead of SVC with RBF kernel for computational efficiency.
    It's 10-100x faster and suitable for large datasets.

    Args:
        features: Training features
        target: Training target

    Returns:
        Trained LinearSVM model (calibrated for probability estimates)
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC

    # Train LinearSVC (much faster than SVC with RBF)
    linear_svm = LinearSVC(
        class_weight="balanced",  # Prioritize minority classes
        max_iter=1000,  # Maximum iterations for convergence
        random_state=42,  # For reproducibility
    )

    # Wrap with CalibratedClassifierCV to get probability estimates
    # This is needed for ROC curves and probability-based predictions
    model = CalibratedClassifierCV(linear_svm, cv=3)
    model.fit(features, target)

    return model
