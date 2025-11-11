"""
Data loading module
Functions for loading and initial processing of diabetes data
"""

import pandas as pd


def load_raw_data(
    filepath: str = "data/raw/diabetes_012_health_indicators_BRFSS2015.csv",
) -> pd.DataFrame:
    """
    Load the raw diabetes dataset from a CSV file

    Args:
        filepath (str): Path to the CSV file containing the raw data

    Returns:
        pd.DataFrame containing the raw diabetes data
    """
    df = pd.read_csv(filepath)
    print(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
