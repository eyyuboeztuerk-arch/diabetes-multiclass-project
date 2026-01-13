"""
Feature Engineering Module
Functions for creating new features to improve model performance

This module creates composite features, categorical encodings, and
interaction terms that may have better predictive power than individual
features alone. Feature engineering is crucial for improving model
performance, especially for complex medical datasets.
"""

import pandas as pd


def create_health_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a composite health risk score

    This function sums multiple binary risk factors to create a single
    composite score. A higher score indicates more health risk factors,
    which is strongly associated with diabetes risk.

    Args:
        df: DataFrame with features

    Returns:
        DataFrame with new 'HealthRiskScore' feature

    Risk factors included:
        - HighBP: High blood pressure (hypertension)
        - HighChol: High cholesterol
        - Stroke: History of stroke
        - HeartDiseaseorAttack: History of heart disease or heart attack
        - HvyAlcoholConsump: Heavy alcohol consumption

    Medical Context:
        These risk factors are known comorbidities with diabetes.
        Patients with multiple risk factors have significantly higher
        diabetes risk. This composite score captures cumulative risk.
    """
    df = df.copy()

    # Define risk factors to sum
    # All are binary (0/1) features
    risk_features = [
        "HighBP",
        "HighChol",
        "Stroke",
        "HeartDiseaseorAttack",
        "HvyAlcoholConsump",
    ]

    # Sum risk factors to create composite score
    # Score ranges from 0 (no risk factors) to 5 (all risk factors)
    df["HealthRiskScore"] = df[risk_features].sum(axis=1)

    return df


def create_lifestyle_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a lifestyle score (positive health habits)

    This function sums positive health behaviors to create a lifestyle
    score. A higher score indicates healthier lifestyle choices, which
    is associated with lower diabetes risk.

    Args:
        df: DataFrame with features

    Returns:
        DataFrame with new 'LifestyleScore' feature

    Positive factors included:
        - PhysActivity: Regular physical activity
        - Fruits: Regular fruit consumption
        - Veggies: Regular vegetable consumption

    Medical Context:
        Healthy lifestyle choices (exercise, healthy diet) are protective
        factors against diabetes. This score captures overall lifestyle
        quality and may help identify low-risk individuals.
    """
    df = df.copy()

    # Define positive lifestyle factors
    # All are binary (0/1) features
    positive_factors = ["PhysActivity", "Fruits", "Veggies"]

    # Sum positive factors to create lifestyle score
    # Score ranges from 0 (no healthy habits) to 3 (all healthy habits)
    df["LifestyleScore"] = df[positive_factors].sum(axis=1)

    return df


def create_bmi_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates BMI categories based on standard thresholds

    This function converts continuous BMI values into categorical groups
    based on medical classification standards. Categories are then
    one-hot encoded for use in machine learning models.

    Args:
        df: DataFrame with BMI feature

    Returns:
        DataFrame with one-hot encoded BMI categories

    BMI Categories (Standard Classification):
        - Underweight: BMI < 18.5
        - Normal: 18.5 ≤ BMI < 25
        - Overweight: 25 ≤ BMI < 30
        - Obese: BMI ≥ 30

    Medical Context:
        BMI categories are clinically meaningful thresholds associated
        with different health risks. Overweight and obesity are major
        risk factors for type 2 diabetes. Categorical encoding may
        capture non-linear relationships better than continuous BMI.
    """
    df = df.copy()

    # Create BMI categories using standard thresholds
    # pd.cut() bins continuous values into discrete categories
    df["BMI_Category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )

    # One-Hot Encoding
    # Converts categorical variable into binary columns (0/1)
    # Each category becomes a separate feature
    bmi_dummies = pd.get_dummies(df["BMI_Category"], prefix="BMI")

    # Add one-hot encoded columns to DataFrame
    df = pd.concat([df, bmi_dummies], axis=1)

    # Drop original categorical column (no longer needed)
    df = df.drop("BMI_Category", axis=1)

    return df


def create_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates simplified age groups from ordinal age variable

    The original Age variable is ordinal (1-13 representing age ranges).
    This function groups ages into broader categories that may capture
    age-related diabetes risk patterns more effectively.

    Args:
        df: DataFrame with Age feature (1-13 scale)

    Returns:
        DataFrame with one-hot encoded age groups

    Age Groups:
        - Young: Age 1-4 (18-44 years)
        - Middle: Age 5-8 (45-64 years)
        - Senior: Age 9-13 (65+ years)

    Medical Context:
        Diabetes risk increases with age, but the relationship may not
        be linear. Grouping ages into clinically meaningful categories
        (young adults, middle-aged, seniors) may improve predictions.
        Type 2 diabetes is most common in middle-aged and older adults.
    """
    df = df.copy()

    # Create age groups from ordinal Age variable (1-13)
    # Original scale: 1=18-24, 2=25-29, ..., 13=80+
    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[0, 4, 8, 13], labels=["Young", "Middle", "Senior"]
    )

    # One-Hot Encoding
    # Converts categorical variable into binary columns
    age_dummies = pd.get_dummies(df["AgeGroup"], prefix="Age")

    # Add one-hot encoded columns to DataFrame
    df = pd.concat([df, age_dummies], axis=1)

    # Drop original categorical column
    df = df.drop("AgeGroup", axis=1)

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates interaction features (products of related features)

    Interaction features capture combined effects of multiple variables.
    The product of two features may have stronger predictive power than
    either feature alone, especially when their effects are synergistic.

    Args:
        df: DataFrame with features

    Returns:
        DataFrame with interaction features

    Interactions created:
        - BMI_x_HighBP: BMI × High Blood Pressure
        - Age_x_BMI: Age × BMI
        - GenHlth_x_PhysActivity: General Health × Physical Activity

    Medical Context:
        1. BMI × HighBP: Obesity combined with hypertension greatly
           increases diabetes risk (synergistic effect)
        2. Age × BMI: Obesity in older adults is particularly risky
        3. GenHlth × PhysActivity: Poor health despite exercise may
           indicate underlying metabolic issues
    """
    df = df.copy()

    # BMI × High Blood Pressure
    # Captures synergistic effect of obesity and hypertension
    df["BMI_x_HighBP"] = df["BMI"] * df["HighBP"]

    # Age × BMI
    # Captures increased risk of obesity in older adults
    df["Age_x_BMI"] = df["Age"] * df["BMI"]

    # General Health × Physical Activity
    # Captures discrepancy between health status and activity level
    df["GenHlth_x_PhysActivity"] = df["GenHlth"] * df["PhysActivity"]

    return df


def apply_all_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering steps in sequence

    This is a convenience function that applies all feature engineering
    transformations in the correct order. It ensures consistent feature
    engineering across training, test, and SMOTE datasets.

    Args:
        df: DataFrame with original features

    Returns:
        DataFrame with all engineered features

    Feature engineering pipeline:
        1. Create HealthRiskScore (composite risk)
        2. Create LifestyleScore (positive habits)
        3. Create BMI categories (one-hot encoded)
        4. Create Age groups (one-hot encoded)
        5. Create interaction features (products)

    Note:
        The order of operations matters. Interaction features are created
        last because they depend on original features (BMI, Age, etc.).
    """
    # Apply all feature engineering functions in sequence
    df = create_health_risk_score(df)
    df = create_lifestyle_score(df)
    df = create_bmi_categories(df)
    df = create_age_groups(df)
    df = create_interaction_features(df)

    # Display summary of feature engineering
    print(f"Feature engineering completed. New shape: {df.shape}")

    return df
