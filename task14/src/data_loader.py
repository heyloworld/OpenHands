"""
Data loader for the Telco Customer Churn dataset.
This module handles loading the dataset, feature engineering, and handling imbalanced data.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import requests
from io import StringIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TelcoChurnDataLoader:
    """
    Class for loading and preprocessing the Telco Customer Churn dataset.
    """
    
    def __init__(self, data_url=None, random_state=42):
        """
        Initialize the data loader.
        
        Args:
            data_url (str, optional): URL to download the dataset from.
                If None, will use the default URL.
            random_state (int, optional): Random state for reproducibility.
        """
        self.data_url = data_url or "https://huggingface.co/datasets/scikit-learn/churn-prediction/raw/main/churn.csv"
        self.random_state = random_state
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.feature_selector = None
        
    def download_data(self):
        """
        Download the dataset from the specified URL.
        
        Returns:
            pandas.DataFrame: The downloaded dataset.
        """
        try:
            logger.info(f"Downloading data from {self.data_url}")
            response = requests.get(self.data_url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Parse the CSV data
            data = pd.read_csv(StringIO(response.text))
            logger.info(f"Successfully downloaded data with shape {data.shape}")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading data: {e}")
            raise
        
    def load_data(self, filepath=None):
        """
        Load the dataset from a file or download it if not available.
        
        Args:
            filepath (str, optional): Path to the dataset file.
                If None, the dataset will be downloaded.
                
        Returns:
            pandas.DataFrame: The loaded dataset.
        """
        try:
            if filepath and os.path.exists(filepath):
                logger.info(f"Loading data from {filepath}")
                self.data = pd.read_csv(filepath)
            elif self.data_url:
                self.data = self.download_data()
                
                # Save a copy of the raw data if filepath is provided
                if filepath:
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    self.data.to_csv(filepath, index=False)
                    logger.info(f"Saved raw data to {filepath}")
            else:
                logger.error("No data file found and no URL provided for download")
                raise FileNotFoundError(f"Data file not found at {filepath} and no URL provided for download")
                
            logger.info(f"Data loaded with shape {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def preprocess_data(self, test_size=0.2):
        """
        Preprocess the data by handling missing values, encoding categorical features,
        and scaling numerical features.
        
        Args:
            test_size (float, optional): Proportion of the dataset to include in the test split.
            
        Returns:
            tuple: Preprocessed training and testing data (X_train, X_test, y_train, y_test).
        """
        if self.data is None:
            logger.error("Data not loaded. Call load_data() first.")
            raise ValueError("Data not loaded. Call load_data() first.")
            
        try:
            logger.info("Preprocessing data...")
            
            # Make a copy of the data to avoid modifying the original
            df = self.data.copy()
            
            # Convert 'TotalCharges' to numeric, handling any errors
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
            # Replace any NaN values in TotalCharges with 0
            df['TotalCharges'].fillna(0, inplace=True)
            
            # Handle missing values
            df.fillna(df.mean(), inplace=True)
            
            # Convert target variable to binary
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
            
            # Drop customer ID column as it's not useful for prediction
            if 'customerID' in df.columns:
                df.drop('customerID', axis=1, inplace=True)
                
            # Identify categorical and numerical columns
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numerical_cols = [col for col in numerical_cols if col != 'Churn']  # Exclude target
            
            logger.info(f"Categorical columns: {categorical_cols}")
            logger.info(f"Numerical columns: {numerical_cols}")
            
            # Create preprocessing pipelines
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            # Combine preprocessing steps
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])
            
            # Split the data
            X = df.drop('Churn', axis=1)
            y = df['Churn']
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            
            logger.info(f"Data split into training ({self.X_train.shape[0]} samples) and testing ({self.X_test.shape[0]} samples)")
            
            # Fit and transform the training data
            self.X_train = self.preprocessor.fit_transform(self.X_train)
            
            # Transform the test data
            self.X_test = self.preprocessor.transform(self.X_test)
            
            logger.info(f"Preprocessed X_train shape: {self.X_train.shape}")
            logger.info(f"Preprocessed X_test shape: {self.X_test.shape}")
            
            return self.X_train, self.X_test, self.y_train, self.y_test
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
            
    def select_features(self, k=10):
        """
        Select the top k features based on ANOVA F-value.
        
        Args:
            k (int, optional): Number of top features to select.
            
        Returns:
            tuple: Feature-selected training and testing data.
        """
        if self.X_train is None or self.X_test is None:
            logger.error("Data not preprocessed. Call preprocess_data() first.")
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
            
        try:
            logger.info(f"Selecting top {k} features...")
            
            # Ensure k is not larger than the number of features
            k = min(k, self.X_train.shape[1])
            
            # Select top k features
            self.feature_selector = SelectKBest(f_classif, k=k)
            X_train_selected = self.feature_selector.fit_transform(self.X_train, self.y_train)
            X_test_selected = self.feature_selector.transform(self.X_test)
            
            logger.info(f"Selected {k} features")
            logger.info(f"Feature-selected X_train shape: {X_train_selected.shape}")
            logger.info(f"Feature-selected X_test shape: {X_test_selected.shape}")
            
            return X_train_selected, X_test_selected
            
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            raise
            
    def handle_imbalanced_data(self, method='smote', sampling_strategy=1.0):
        """
        Handle imbalanced data using oversampling or undersampling techniques.
        
        Args:
            method (str, optional): Method to use for handling imbalanced data.
                Options: 'smote' for oversampling, 'undersample' for undersampling.
            sampling_strategy (float or str, optional): Sampling strategy to use.
                For 'smote', if float, specifies the ratio of the number of samples 
                in the minority class over the number of samples in the majority class.
                For 'undersample', if float, specifies the ratio of the number of samples 
                in the majority class over the number of samples in the minority class.
                
        Returns:
            tuple: Resampled training data and labels.
        """
        if self.X_train is None or self.y_train is None:
            logger.error("Data not preprocessed. Call preprocess_data() first.")
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
            
        try:
            # Check class distribution
            class_counts = np.bincount(self.y_train)
            logger.info(f"Original class distribution: {class_counts}")
            
            if method.lower() == 'smote':
                logger.info(f"Applying SMOTE oversampling with sampling_strategy={sampling_strategy}")
                resampler = SMOTE(sampling_strategy=sampling_strategy, random_state=self.random_state)
            elif method.lower() == 'undersample':
                logger.info(f"Applying random undersampling with sampling_strategy={sampling_strategy}")
                resampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=self.random_state)
            else:
                logger.error(f"Unknown resampling method: {method}")
                raise ValueError(f"Unknown resampling method: {method}. Use 'smote' or 'undersample'.")
                
            X_resampled, y_resampled = resampler.fit_resample(self.X_train, self.y_train)
            
            # Check new class distribution
            new_class_counts = np.bincount(y_resampled)
            logger.info(f"Resampled class distribution: {new_class_counts}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"Error handling imbalanced data: {e}")
            raise
            
    def get_feature_names(self):
        """
        Get the names of the features after preprocessing.
        
        Returns:
            list: Names of the features.
        """
        if self.preprocessor is None:
            logger.error("Preprocessor not fitted. Call preprocess_data() first.")
            raise ValueError("Preprocessor not fitted. Call preprocess_data() first.")
            
        try:
            # Get feature names from the preprocessor
            feature_names = self.preprocessor.get_feature_names_out()
            
            # If feature selection was applied, get the selected feature names
            if self.feature_selector is not None:
                mask = self.feature_selector.get_support()
                feature_names = feature_names[mask]
                
            return feature_names
            
        except Exception as e:
            logger.error(f"Error getting feature names: {e}")
            raise
            
    def get_data(self):
        """
        Get the preprocessed and possibly resampled data.
        
        Returns:
            tuple: Training and testing data (X_train, X_test, y_train, y_test).
        """
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            logger.error("Data not preprocessed. Call preprocess_data() first.")
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
            
        return self.X_train, self.X_test, self.y_train, self.y_test


if __name__ == "__main__":
    # Example usage
    data_loader = TelcoChurnDataLoader()
    data_loader.load_data()
    data_loader.preprocess_data()
    
    # Select features
    X_train_selected, X_test_selected = data_loader.select_features(k=15)
    
    # Handle imbalanced data
    X_resampled, y_resampled = data_loader.handle_imbalanced_data(method='smote')
    
    print(f"Original X_train shape: {data_loader.X_train.shape}")
    print(f"Selected X_train shape: {X_train_selected.shape}")
    print(f"Resampled X_train shape: {X_resampled.shape}")
    print(f"Resampled y_train shape: {y_resampled.shape}")
    
    # Get feature names
    feature_names = data_loader.get_feature_names()
    print(f"Selected feature names: {feature_names}")