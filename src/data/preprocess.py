"""
Data preprocessing utilities.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load data from filepath and perform initial cleaning.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        Cleaned DataFrame
    """
    df = pd.read_csv(filepath)
    
    # Basic cleaning operations
    df = df.drop_duplicates()
    df = df.dropna(how='all')
    
    return df


def split_data(df: pd.DataFrame, 
               target_column: str,
               test_size: float = 0.2,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
