"""
Feature Engineering Module
Functions for creating new features
"""

import pandas as pd


def create_health_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a composite health risk score

    Args:
        df: DataFrame with features

    Returns:
        DataFrame with new feature
    """
    df = df.copy()

    # Sum risk factors
    risk_features = [
        "HighBP",
        "HighChol",
        "Stroke",
        "HeartDiseaseorAttack",
        "HvyAlcoholConsump",
    ]
    df["HealthRiskScore"] = df[risk_features].sum(axis=1)

    return df


def create_lifestyle_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a lifestyle score (positive health habits)

    Args:
        df: DataFrame with features

    Returns:
        DataFrame with new feature
    """
    df = df.copy()

    # Positive factors
    df["LifestyleScore"] = df["PhysActivity"] + df["Fruits"] + df["Veggies"]

    return df


def create_bmi_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates BMI categories

    Args:
        df: DataFrame with BMI

    Returns:
        DataFrame with BMI categories
    """
    df = df.copy()

    df["BMI_Category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )

    # One-Hot Encoding
    bmi_dummies = pd.get_dummies(df["BMI_Category"], prefix="BMI")
    df = pd.concat([df, bmi_dummies], axis=1)
    df = df.drop("BMI_Category", axis=1)

    return df


def create_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates simplified age groups

    Args:
        df: DataFrame with Age

    Returns:
        DataFrame with age groups
    """
    df = df.copy()

    # Age 1-13 to 3 groups
    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[0, 4, 8, 13], labels=["Young", "Middle", "Senior"]
    )

    # One-Hot Encoding
    age_dummies = pd.get_dummies(df["AgeGroup"], prefix="Age")
    df = pd.concat([df, age_dummies], axis=1)
    df = df.drop("AgeGroup", axis=1)

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates interaction features

    Args:
        df: DataFrame with features

    Returns:
        DataFrame with interaction features
    """
    df = df.copy()

    # BMI x HighBP
    df["BMI_x_HighBP"] = df["BMI"] * df["HighBP"]

    # Age x BMI
    df["Age_x_BMI"] = df["Age"] * df["BMI"]

    # GenHlth x PhysActivity
    df["GenHlth_x_PhysActivity"] = df["GenHlth"] * df["PhysActivity"]

    return df


def apply_all_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering steps

    Args:
        df: DataFrame with original features

    Returns:
        DataFrame with all new features
    """
    df = create_health_risk_score(df)
    df = create_lifestyle_score(df)
    df = create_bmi_categories(df)
    df = create_age_groups(df)
    df = create_interaction_features(df)

    print(f"Feature engineering completed. New shape: {df.shape}")

    return df
