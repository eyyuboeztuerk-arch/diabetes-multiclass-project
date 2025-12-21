"""
Preprocessing Module
Functions for data preprocessing and feature engineering
"""

from typing import List, Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Checks data quality

    Args:
        df: DataFrame with the data

    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        "missing_values": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum(),
        "shape": df.shape,
        "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
    }
    return quality_report


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicates from the dataset

    Args:
        df: DataFrame with the data

    Returns:
        DataFrame without duplicates
    """
    initial_shape = df.shape[0]
    df_clean = df.drop_duplicates()
    removed = initial_shape - df_clean.shape[0]
    print(f"Duplicates removed: {removed}")
    return df_clean


def create_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple:
    """
    Creates train-test split

    Args:
        features: Features
        target: Target
        test_size: Size of test set
        random_state: Random seed

    Returns:
        Tuple (features_train, features_test, target_train, target_test)
    """
    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    print(f"Training Set: {features_train.shape[0]} Samples")
    print(f"Test Set: {features_test.shape[0]} Samples")
    print(f"\nTraining Set Distribution:\n{target_train.value_counts().sort_index()}")
    print(f"\nTest Set Distribution:\n{target_test.value_counts().sort_index()}")

    return features_train, features_test, target_train, target_test


def scale_features(features_train: pd.DataFrame, features_test: pd.DataFrame) -> Tuple:
    """
    Scales features with StandardScaler (all features)

    Note: This function scales ALL features. For selective scaling of only
    continuous features, use scale_continuous_features() instead.

    Args:
        features_train: Training features
        features_test: Test features

    Returns:
        Tuple (features_train_scaled, features_test_scaled, scaler)
    """
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)

    # Convert back to DataFrame
    features_train_scaled = pd.DataFrame(
        features_train_scaled,
        columns=features_train.columns,
        index=features_train.index,
    )
    features_test_scaled = pd.DataFrame(
        features_test_scaled, columns=features_test.columns, index=features_test.index
    )

    return features_train_scaled, features_test_scaled, scaler


def scale_continuous_features(
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    continuous_features: List[str],
) -> Tuple:
    """
    Scales only continuous/quasi-continuous features using StandardScaler

    This function follows industry best practices:
    - Binary features (0/1): No scaling needed
    - Ordinal features (already label-encoded): No scaling needed
    - Continuous features: Apply StandardScaler

    Args:
        features_train: Training features
        features_test: Test features
        continuous_features: List of continuous feature names to scale

    Returns:
        Tuple (features_train_scaled, features_test_scaled, scaler)
    """
    # Separate continuous and other features
    other_features = [
        col for col in features_train.columns if col not in continuous_features
    ]

    # Scale only continuous features
    scaler = StandardScaler()
    train_continuous_scaled = scaler.fit_transform(features_train[continuous_features])
    test_continuous_scaled = scaler.transform(features_test[continuous_features])

    # Convert to DataFrame
    train_continuous_scaled_df = pd.DataFrame(
        train_continuous_scaled, columns=continuous_features, index=features_train.index
    )
    test_continuous_scaled_df = pd.DataFrame(
        test_continuous_scaled, columns=continuous_features, index=features_test.index
    )

    # Concatenate scaled continuous features with other features
    features_train_scaled = pd.concat(
        [train_continuous_scaled_df, features_train[other_features]], axis=1
    )

    features_test_scaled = pd.concat(
        [test_continuous_scaled_df, features_test[other_features]], axis=1
    )

    # Reorder columns to match original order
    features_train_scaled = features_train_scaled[features_train.columns]
    features_test_scaled = features_test_scaled[features_test.columns]

    print(
        f"\n✓ Scaled {len(continuous_features)} continuous features: {continuous_features}"
    )
    print(f"✓ Kept {len(other_features)} binary/ordinal features unchanged")

    return features_train_scaled, features_test_scaled, scaler


def apply_smote(
    features_train: pd.DataFrame, target_train: pd.Series, random_state: int = 42
) -> Tuple:
    """
    Applies SMOTE for class balancing

    Args:
        features_train: Training features
        target_train: Training target
        random_state: Random seed

    Returns:
        Tuple (features_train_resampled, target_train_resampled)
    """
    print("Before SMOTE:")
    print(target_train.value_counts().sort_index())

    smote = SMOTE(random_state=random_state)
    features_train_resampled, target_train_resampled = smote.fit_resample(  # pyright: ignore[reportAssignmentType]
        features_train, target_train
    )

    print("\nAfter SMOTE:")
    print(pd.Series(target_train_resampled).value_counts().sort_index())  # pyright: ignore[reportArgumentType, reportCallIssue]

    return features_train_resampled, target_train_resampled


def save_processed_data(
    df: pd.DataFrame, filepath: str = "data/processed/diabetes_012_processed.csv"
) -> None:
    """
    Saves processed data

    Args:
        df: DataFrame with processed data
        filepath: Target path
    """
    df.to_csv(filepath, index=False)
    print(f"Data saved: {filepath}")
