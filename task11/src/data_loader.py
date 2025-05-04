import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

def load_and_preprocess_data(random_state: int = 42) -> Dict[str, Any]:
    """
    Load and preprocess the California Housing dataset (as a replacement for Boston).
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing preprocessed data and metadata
    """
    # Load dataset directly from scikit-learn
    try:
        california = fetch_california_housing()
        data = pd.DataFrame(california.data, columns=california.feature_names)
        data['PRICE'] = california.target
        print(f"Dataset loaded with {data.shape[0]} samples and {data.shape[1]} features")
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")
    
    # Split into train and test sets
    X = data.drop('PRICE', axis=1)
    y = data['PRICE']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    print(f"Split into {X_train.shape[0]} training samples and {X_test.shape[0]} test samples")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames to keep column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Print dataset information
    print(f"Features: {', '.join(X_train.columns)}")
    print(f"Target: PRICE")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    # Calculate and print feature statistics
    feature_stats = {}
    for col in X_train.columns:
        feature_stats[col] = {
            'mean': X_train[col].mean(),
            'std': X_train[col].std(),
            'min': X_train[col].min(),
            'max': X_train[col].max()
        }
    
    # Return preprocessed data and metadata
    return {
        'X_train': X_train_scaled_df,
        'y_train': y_train,
        'X_test': X_test_scaled_df,
        'y_test': y_test,
        'feature_names': list(X_train.columns),
        'scaler': scaler,
        'feature_stats': feature_stats
    }

if __name__ == "__main__":
    # Test the data loader
    data = load_and_preprocess_data()
    
    # Print feature statistics
    print("\nFeature Statistics:")
    for feature, stats in data['feature_stats'].items():
        print(f"  {feature}:")
        for stat_name, stat_value in stats.items():
            print(f"    {stat_name}: {stat_value:.4f}")
    
    # Print sample data
    print("\nSample of scaled training data:")
    print(data['X_train'].head())
    
    print("\nSample of training target values:")
    print(data['y_train'].head())