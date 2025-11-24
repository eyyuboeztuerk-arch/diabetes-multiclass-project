"""
Data Loading Module
Functions for loading and initial processing of diabetes data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

def load_raw_data(filepath: str = "data/raw/diabetes_012_health_indicators_BRFSS2015.csv") -> pd.DataFrame:
    """
    Loads the raw diabetes data
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        DataFrame with raw data
    """
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def get_feature_names() -> dict:
    """
    Returns a dictionary with feature names and descriptions
    
    Returns:
        Dictionary with feature descriptions
    """
    features = {
        'Diabetes_012': 'Target variable: 0=no diabetes, 1=prediabetes, 2=diabetes',
        'HighBP': 'High blood pressure',
        'HighChol': 'High cholesterol',
        'CholCheck': 'Cholesterol check in last 5 years',
        'BMI': 'Body Mass Index',
        'Smoker': 'Smoker (at least 100 cigarettes in lifetime)',
        'Stroke': 'Stroke history',
        'HeartDiseaseorAttack': 'Coronary heart disease or myocardial infarction',
        'PhysActivity': 'Physical activity in last 30 days',
        'Fruits': 'Fruit consumption (1+ per day)',
        'Veggies': 'Vegetable consumption (1+ per day)',
        'HvyAlcoholConsump': 'Heavy alcohol consumption',
        'AnyHealthcare': 'Health insurance coverage',
        'NoDocbcCost': 'Could not see doctor due to cost',
        'GenHlth': 'General health status (1-5)',
        'MentHlth': 'Days with poor mental health (0-30)',
        'PhysHlth': 'Days with poor physical health (0-30)',
        'DiffWalk': 'Difficulty walking or climbing stairs',
        'Sex': 'Sex (0=female, 1=male)',
        'Age': 'Age group (1-13)',
        'Education': 'Education level (1-6)',
        'Income': 'Income level (1-8)'
    }
    return features

def get_data_info(df: pd.DataFrame) -> None:
    """
    Prints information about the dataset
    
    Args:
        df: DataFrame with data
    """
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"\nShape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nTarget variable distribution:\n{df['Diabetes_012'].value_counts().sort_index()}")
    print(f"\nTarget variable percentage:\n{df['Diabetes_012'].value_counts(normalize=True).sort_index() * 100}")
    print("=" * 60)

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits features and target variable
    
    Args:
        df: DataFrame with all data
    
    Returns:
        Tuple (features, target)
    """
    features = df.drop('Diabetes_012', axis=1)
    target = df['Diabetes_012']
    return features, target
